try:
    from itertools import izip_longest as zip_longest
except:
    from itertools import zip_longest
import math
import numpy as np
import os
try:
    import Queue
except ImportError:
    import queue as Queue
import threading
from warnings import warn

import gflags
from tqdm import tqdm
import tensorflow as tf

from utils import split_in_chunks, fig2array


def validate(placeholders,
             outs,
             summary_ops,
             reset_cm_op,
             which_set='valid',
             epoch_id=None,
             nthreads=2):

    cfg = gflags.cfg
    if getattr(cfg.valid_params, 'resize_images', False):
        warn('Forcing resize_images to False in evaluation.')
        cfg.valid_params.update({'resize_images': False})

    this_set = cfg.Dataset(
        which_set=which_set,
        **cfg.valid_params)

    # Prepare the threads to save the images
    save_basedir = os.path.join('samples', cfg.model_name,
                                this_set.which_set)
    img_queue = Queue.Queue(maxsize=10)
    sentinel = object()  # Poison pill
    for _ in range(nthreads):
        t = threading.Thread(
            target=save_images,
            args=(img_queue, save_basedir, sentinel))
        t.setDaemon(True)  # Die when main dies
        t.start()
        cfg.sv.coord.register_thread(t)

    # TODO posso distinguere training da valid??
    # summary_writer = tf.summary.FileWriter(logdir=cfg.val_checkpoints_dir,
    #                                        graph=cfg.sess.graph)

    # Re-init confusion matrix
    # cm = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='mean_iou')
    # cfg.sess.run([tf.assign(cm, tf.zeros(tf.shape(cm), dtype=tf.int32))])

    # Begin loop over dataset samples
    tot_loss = 0
    epoch_id_str = 'Ep ' + str(epoch_id+1) + ': ' if epoch_id else ''
    epoch_id = epoch_id if epoch_id else 0
    pbar = tqdm(total=this_set.nbatches,
                bar_format='[' + which_set + '] {n_fmt}/{total_fmt} ' +
                           epoch_id_str + '{percentage:3.0f}%|{bar}| '
                           '[{elapsed}<{remaining},'
                           '{rate_fmt} {postfix}]')

    if cfg.compute_mean_iou:
        prev_subset = None
        per_subset_IoUs = {}
        # Reset Confusion Matrix at the beginning of validation
        cfg.sess.run(reset_cm_op)

    cidx = (epoch_id*this_set.nbatches)
    frame_idx = cidx
    for bidx in range(this_set.nbatches):
        if cfg.sv.should_stop():  # Stop requested
            break

        ret = this_set.next()
        x_batch, y_batch = ret['data'], ret['labels']
        assert all(el == ret['subset'][0] for el in ret['subset'])
        subset = ret['subset'][0]
        f_batch = ret['filenames']
        raw_data_batch = ret['raw_data']

        # Is this batch shorter than batch_size?
        # Check if this batch will not be processed by all the devices.
        # When the sequence is shorter than seq_length or the number of
        # batches is smaller than batch_size, the batch will be smaller
        # than usual. When this happens we might not be able to feed all
        # the CPUs/GPUs altogether. In that case here we compute the
        # number of GPUs that we can use for the current batch
        batch_size = cfg.val_batch_size
        this_len_batch = len(x_batch)
        # Spread the batch over the lowest number of GPUs
        this_num_splits = this_len_batch // batch_size
        if this_len_batch % batch_size != 0:
            this_num_splits += 1
        summary_op = summary_ops[this_num_splits - 1]
        outs_summary = {}
        outs_summary.update(outs)
        outs_summary.update({'summary_op': summary_op})

        # TODO: check the confusion matrix!
        if cfg.compute_mean_iou:
            # Reset the confusion matrix when we switch video
            if this_set.set_has_GT and (not prev_subset or
                                        subset != prev_subset):
                if cfg.summary_per_subset:
                    tf.logging.info('Reset confusion matrix! {} --> {}'.format(
                        prev_subset, subset))
                    cfg.sess.run(reset_cm_op)
                if cfg.stateful_validation:
                    if subset == 'default':
                        raise RuntimeError(
                            'For stateful validation, the validation '
                            'dataset should provide `subset`')
                    # reset_states(model, x_batch.shape)
                prev_subset = subset

        if cfg.seq_length and y_batch.shape[1] > 1:
            x_in = x_batch
            if cfg.target_frame == 'middle':
                y_in = y_batch[:, cfg.seq_length // 2, ...]  # 4D: not one-hot
            if cfg.target_frame == 'last':
                y_in = y_batch[:, cfg.seq_length - 1, ...]
        else:
            x_in = x_batch
            y_in = y_batch

        # if cfg.use_second_path:
        #     x_in = [x_in[..., :3], x_in[..., 3:]]

        x_batch_chunks, y_batch_chunks = split_in_chunks(x_in, y_in,
                                                         this_num_splits)

        # Fill the placeholders with data up to this_num_splits, and
        # then repeat one of the chunks. Note that this will be
        # ignored later on (see comment where placeholders are created)
        [inputs_per_gpu, labels_per_gpu, num_splits,
         num_batches] = placeholders
        in_vals = list(zip_longest(inputs_per_gpu, x_batch_chunks,
                                   fillvalue=x_batch_chunks[0]))
        in_vals.extend(list(zip_longest(labels_per_gpu, y_batch_chunks,
                                        fillvalue=y_batch_chunks[0])))
        in_vals.extend([(num_splits, this_num_splits)])
        in_vals.extend([(num_batches, this_len_batch)])
        feed_dict = {p: v for(p, v) in in_vals}

        if this_set.set_has_GT and cfg.compute_mean_iou:
            # Class balance
            # class_balance_w = np.ones(np.prod(
            #     mini_x.shape[:3])).astype(floatX)
            # class_balance = loss_kwargs.get('class_balance', '')
            # if class_balance in ['median_freq_cost', 'rare_freq_cost']:
            #     w_freq = loss_kwargs.get('w_freq')
            #     class_balance_w = w_freq[y_true.flatten()].astype(floatX)

            # Get the batch pred, the mIoU so far (computed incrementally
            # over the sequences processed so far), the batch loss and
            # potentially the summary
            if cidx % cfg.val_summary_freq == 0:
                    (of_pred_batch, y_pred_batch, y_prob_batch,
                     metrics_out, loss, _, summary_str) = cfg.sess.run(
                         outs + [summary_op], feed_dict=feed_dict)
                    cfg.sv.summary_computed(cfg.sess, summary_str,
                                            global_step=cidx)
            else:
                (of_pred_batch, y_pred_batch, y_prob_batch,
                 metrics_out, loss, _) = cfg.sess.run(
                     outs, feed_dict=feed_dict)

            tot_loss += loss

            if cfg.compute_mean_iou:
                mIoU, per_class_IoU = metrics_out
                # If fg/bg, just consider the foreground class
                if len(per_class_IoU) == 2:
                    per_class_IoU = per_class_IoU[1]

                # Save the IoUs per subset (i.e., video) and their average
                if cfg.summary_per_subset:
                    per_subset_IoUs[subset] = per_class_IoU
                    mIoU = np.mean(per_subset_IoUs.values())

                pbar.set_postfix({
                    'loss': '{:.3f}({:.3f})'.format(loss, tot_loss/(bidx+1)),
                    'mIoU': '{:.3f}'.format(mIoU)})
        else:
            if cidx % cfg.val_summary_freq == 0:
                fetch_dict = cfg.sess.run(outs_summary, feed_dict=feed_dict)

                cfg.sv.summary_computed(cfg.sess,
                                        fetch_dict['summary_op'],
                                        global_step=cidx)
            else:
                fetch_dict = cfg.sess.run(outs, feed_dict=feed_dict)
                mIoU = 0

        # TODO there is no guarantee that this will be processed
        # in order. We could use condition variables, e.g.,
        # http://python.active-venture.com/lib/condition-objects.html
        # Save image summary for learning visualization

        of_pred_batch = fetch_dict.get('of_pred', [None] * len(f_batch))
        y_pred_batch = fetch_dict['pred']
        y_prob_batch = fetch_dict['out_act']
        img_queue.put((frame_idx, this_set, x_batch, y_batch, f_batch, subset,
                       raw_data_batch, of_pred_batch, y_pred_batch,
                       y_prob_batch))
        cidx += 1
        frame_idx += len(x_batch)
        pbar.update(1)
    pbar.close()

    # Kill the threads
    for _ in range(nthreads):
        img_queue.put(sentinel)

    # Write the summaries
    class_labels = this_set.mask_labels[:this_set.non_void_nclasses]
    if cfg.compute_mean_iou:
        if cfg.summary_per_subset:
            # Write the IoUs per subset (i.e., video) and (potentially) class
            # and their average
            write_IoUs_summaries(per_subset_IoUs, step=epoch_id,
                                 class_labels=class_labels)
            write_IoUs_summaries({'mean_per_video': mIoU}, step=epoch_id)
        else:
            # Write the IoUs (potentially per class) and the average IoU over
            # all the sequences
            write_IoUs_summaries({'': per_class_IoU}, step=epoch_id,
                                 class_labels=class_labels)
            write_IoUs_summaries({'global_avg': mIoU}, step=epoch_id)

    img_queue.join()  # Wait for the threads to be done
    this_set.finish()  # Close the dataset
    return mIoU


def write_IoUs_summaries(IoUs, step=None, class_labels=[]):
    '''Write per-video, per-class and global IoU summaries in TensorBoard

    Arguments
    ---------
    IoUs: dictionary
        A dictionary if IoUs per "category". The keys of the dictionary
        can be the subsets (i.e., videos) or more other kinds of
        categories. The values can either be a single IoU scalar
        or a list.
    step: int
        The current cumulative (i.e., global, not limited to this round
        of validation) iteration, used as x coordinate in TensorBoard.
    class_labels: list
        A list of labels for each class in the dataset. When this is
        provided and the per-key values are lists of the same length as
        class_labels, for each subset (or more generally, for each key
        of IoUs) the per-class IoUs will be printed along with the
        average over the classes.
    '''
    cfg = gflags.cfg

    def write_summary(labs, val):
        '''Write a single summary in TensorBoard

        Arguments
        ---------
        labs: list
            A list of labels. The labels will be joined with underscores
            unless None or empty.
        val: int or float or iterable
            The value to be visualized in Tensorboard.
        '''
        assert isinstance(labs, (tuple, list))
        labs = '_'.join([el for el in labs if el not in (None, '')])
        summary = tf.Summary.Value(tag='IoUs/' + labs, simple_value=val)
        summary_str = tf.Summary(value=[summary])
        cfg.sv.summary_computed(cfg.sess, summary_str, global_step=step)

    for labs, vals in IoUs.iteritems():
        # Write per-class IoU if labels are provided and the number of
        # items in IoU is equal to the number of labels
        if len(class_labels) > 2 and len(class_labels) == len(vals):
            cum_IoU = []
            # Write per-subset, per-class IoU
            for class_val, class_label in zip(vals, class_labels):
                write_summary((labs, '{}_IoU'.format(class_label)), class_val)
                cum_IoU.append(class_val)
            # Write avg per-subset
            write_summary((labs, 'class_avg_IoU'), class_val)
        # Write per-subset IoU
        else:
            write_summary((labs, 'IoU'), vals)


def save_images(img_queue, save_basedir, sentinel):
    import matplotlib as mpl
    import seaborn as sns
    from utils import flowToColor
    import cv2
    cfg = gflags.cfg

    while True:
        if cfg.sv.should_stop() and img_queue.empty():  # Stop requested
            tf.logging.debug('Save images thread stopping for sv.should_stop')
            break
        try:
            img = img_queue.get(False)
            if img == sentinel:  # Validation is over, die
                tf.logging.debug('Save images thread stopping for sentinel')
                img_queue.task_done()
                break
            (bidx, this_set, x_batch, y_batch, f_batch, subset,
             raw_data_batch, of_pred_batch, y_pred_batch, y_prob_batch) = img

            cfg = gflags.cfg

            # Initialize variables
            nclasses = this_set.nclasses
            seq_length = this_set.seq_length
            if nclasses is not None:
                try:
                    cmap = this_set.cmap
                except AttributeError:
                    cmap = [el for el in sns.hls_palette(this_set.nclasses)]
                cmap = mpl.colors.ListedColormap(cmap)
            else:
                if x_batch.shape[-1] == 1:
                    cmap = 'gray'
                else:
                    cmap = None
            labels = this_set.mask_labels

            if not np.all(of_pred_batch):
                lengths = (len(x_batch), len(y_batch), len(f_batch),
                           len(y_pred_batch), len(y_prob_batch),
                           len(raw_data_batch))
                assert all(el == lengths[0] for el in lengths), (
                    'x_batch: {}\ny_batch: {}\nf_batch: {}\ny_pred_batch: {}'
                    '\ny_prob_batch: {}\nraw_data_batch: {}'.format(
                          *lengths))
            else:
                lengths = (len(x_batch), len(y_batch), len(f_batch),
                           len(y_pred_batch), len(y_prob_batch),
                           len(of_pred_batch), len(raw_data_batch))
                assert all(el == lengths[0] for el in lengths), (
                    'x_batch: {}\ny_batch: {}\nf_batch: {}\ny_pred_batch: {}'
                    '\ny_prob_batch: {}\nof_pred_batch: {}'
                    '\nraw_data_batch: {}'.format(*lengths))

            zip_list = (x_batch, y_batch, f_batch, of_pred_batch,
                        y_pred_batch, y_prob_batch, raw_data_batch)

            # Save samples, iterating over each element of the batch
            for el in zip(*zip_list):
                (x, y, f, of_pred, y_pred, y_prob, raw_data) = el

                # y = np.expand_dims(y, -1)
                # y_pred = np.expand_dims(y_pred, -1)
                if len(x.shape) == 4:
                    seq_length = x_batch.shape[1]
                    if cfg.target_frame == 'middle':
                        which_frame = seq_length // 2
                    elif cfg.target_frame == 'last':
                        which_frame = seq_length - 1
                else:
                    which_frame = 0

                f = f[which_frame]
                if not isinstance(f, int):
                    f = f[:-4]  # strip .jpg
                    f = f + '.png'
                else:
                    f = str(f) + '.png'

                # Retrieve the optical flow channels
                if x.shape[-1] == 5:
                    of = x[which_frame, ..., 3:]
                    # ang, mag = of
                    hsv = np.zeros_like(x[which_frame, ..., :3],
                                        dtype='uint8')
                    hsv[..., 0] = of[..., 0] * 255
                    hsv[..., 1] = 255
                    hsv[..., 2] = cv2.normalize(of[..., 1] * 255, None, 0, 255,
                                                cv2.NORM_MINMAX)
                    of = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                else:
                    of = None

                if of_pred is not None:
                    of_rgb = flowToColor(of_pred)
                    of_rgb = cv2.resize(of_rgb, (y_pred.shape[1],
                                                 y_pred.shape[0]))
                else:
                    of_rgb = None

                if raw_data.ndim == 4:
                    # Show only the middle frame
                    heat_map_in = raw_data[which_frame, ..., :3]
                else:
                    heat_map_in = raw_data

                # PRINT THE HEATMAP
                if cfg.show_heatmaps_summaries and nclasses is not None:
                    # do not pass optical flow
                    save_heatmap_fn(heat_map_in, of, y_prob, labels,
                                    nclasses, save_basedir, subset, f, bidx)

                # PRINT THE SAMPLES
                # Keep most likely prediction only
                # y = y.argmax(2)
                # y_pred = y_pred.argmax(2)

                # Save image and append frame to animations sequence
                if (cfg.save_gif_frames_on_disk or
                        cfg.show_samples_summaries or cfg.save_gif_on_disk):
                    if raw_data.ndim == 4:
                        sample_in = raw_data[which_frame]
                        if y.shape[0] > 1:
                            y_in = y[which_frame]
                        else:
                            y_in = y[0]
                    else:
                        sample_in = raw_data
                        y_in = y
                    save_samples_and_animations(sample_in, of, of_rgb, y_pred,
                                                y_in, cmap, nclasses, labels,
                                                subset, save_basedir, f, bidx)
                bidx += 1  # Make sure every batch is in a separate frame
            img_queue.task_done()
        except Queue.Empty:
            continue
        except Exception as e:
            # Do not crash for errors during image saving
            # cfg.sv.coord.request_stop(e)
            # raise
            # break
            from traceback import format_exc
            tf.logging.error('Error in save_images!!\n' + format_exc(e))
            img_queue.task_done()
            continue


def save_heatmap_fn(x, of, y_prob, labels, nclasses, save_basedir, subset,
                    f, bidx):
    '''Save an image of the probability of each class

    Save the image and the heatmap of the probability of each class'''
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import AxesGrid
    from StringIO import StringIO

    cfg = gflags.cfg

    fig = plt.figure(dpi=300)
    # Remove whitespace from around the image
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # We will plot the image, each channel/class separately and
    # potentially the optical flow. Let's spread them evenly in a square
    nclasses = cfg.nclasses
    num_extra_frames = 1 if of is None else 2
    ncols = int(math.ceil(math.sqrt(nclasses + num_extra_frames)))
    nrows = int(math.ceil((nclasses + num_extra_frames) / ncols))

    grid = AxesGrid(fig, 111,
                    nrows_ncols=(nrows, ncols),
                    axes_pad=0.25,
                    share_all=True,
                    label_mode="L",
                    cbar_location="right",
                    cbar_mode="single")
    sh = x.shape
    for ax in grid:
        ax.set_xticks([sh[1]])
        ax.set_yticks([sh[0]])

    # image
    grid[0].imshow(x)
    grid[1].set_title('Prediction')
    # optical flow: cmap is ignored for 3D
    if of is not None:
        grid[1].imshow(of, vmin=0, vmax=1, interpolation='nearest')
        grid[1].set_title('Optical flow')
    # heatmaps
    for l, pred, ax in zip(labels[:nclasses-1], y_prob.transpose(2, 0, 1),
                           grid[num_extra_frames:]):
        im = ax.imshow(pred, cmap='hot', vmin=0, vmax=1,
                       interpolation='nearest')
        ax.set_title(l)
    # set the colorbar to match
    grid.cbar_axes[0].colorbar(im)
    for cax in grid.cbar_axes:
        cax.toggle_label(False)
    # Uncomment to save the heatmaps on disk
    # fpath = os.path.join(save_basedir, 'heatmaps', subset, f)
    # if not os.path.exists(os.path.dirname(fpath)):
    #     os.makedirs(os.path.dirname(fpath))
    # plt.savefig(fpath)  # save 3 subplots

    sio = StringIO()
    plt.imsave(sio, fig2array(fig), format='png')
    size = fig.get_size_inches()*fig.dpi  # size in pixels
    heatmap_img = tf.Summary.Image(encoded_image_string=sio.getvalue(),
                                   height=int(size[0]),
                                   width=int(size[1]))
    heatmap_img_summary = tf.Summary.Value(tag='Heatmaps/' + subset,
                                           image=heatmap_img)
    summary_str = tf.Summary(value=[heatmap_img_summary])
    cfg.sv.summary_computed(cfg.sess, summary_str, global_step=bidx)

    plt.close('all')


def save_samples_and_animations(raw_data, of, of_pred, y_pred, y, cmap,
                                nclasses, labels, subset, save_basedir,
                                f, bidx):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import AxesGrid
    from StringIO import StringIO

    cfg = gflags.cfg

    fig = plt.figure(dpi=300)
    # Remove whitespace from around the image
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0, top=1)

    # Set number of rows
    if cfg.model_returns_of:
        n_rows = 2
        n_cols = 3
    else:
        n_rows = 2
        n_cols = 2

    grid = AxesGrid(fig, 111,
                    nrows_ncols=(n_rows, n_cols),
                    axes_pad=0.50,
                    share_all=True,
                    label_mode="L",
                    cbar_location="right",
                    cbar_mode="single")
    sh = raw_data.shape
    for ax in grid:
        ax.set_xticks([sh[1]])
        ax.set_yticks([sh[0]])

    # This element is not used
    if cfg.model_returns_of:
        # grid[5].set_visible(False)
        abs_diff = np.squeeze(np.abs(raw_data-y_pred))
        grid[5].imshow(abs_diff, vmin=0, vmax=1, interpolation='nearest')
    # image
    if raw_data.shape[-1] == 1:
        raw_data_cmap = 'gray'
    else:
        raw_data_cmap = None
    grid[0].imshow(np.squeeze(raw_data), cmap=raw_data_cmap)
    grid[0].set_title('Image')
    # prediction
    grid[2].imshow(np.squeeze(y_pred), cmap=cmap, vmin=0, vmax=nclasses)
    grid[2].set_title('Prediction')
    im = None
    # OF
    if of is not None:
        grid[3].imshow(of, vmin=0, vmax=1, interpolation='nearest')
        grid[3].set_title('Optical flow')
    else:
        grid[3].set_visible(False)
    if of_pred is not None:
        grid[4].imshow(of_pred)
        grid[4].set_title('Predicted OF')
    # GT
    if y is not None:
        im = grid[1].imshow(np.squeeze(y), cmap=cmap, vmin=0, vmax=nclasses)
        grid[1].set_title('Ground truth')
    else:
        grid[1].set_visible(False)
    # set the colorbar to match GT or prediction
    grid.cbar_axes[0].colorbar(im)
    for cax in grid.cbar_axes:
        cax.toggle_label(True)  # show labels
        cax.set_yticks(np.arange(len(labels)) + 0.5)
        cax.set_yticklabels(labels)

    # TODO: Labels 45 gradi

    if cfg.save_gif_frames_on_disk:
        fpath = os.path.join(save_basedir, 'segmentations', subset, f)
        if not os.path.exists(os.path.dirname(fpath)):
            os.makedirs(os.path.dirname(fpath))
        plt.savefig(fpath)  # save 3 subplots

    if cfg.show_samples_summaries:
        sio = StringIO()
        plt.imsave(sio, fig2array(fig), format='png')
        # size = fig.get_size_inches()*fig.dpi  # size in pixels
        seq_img = tf.Summary.Image(encoded_image_string=sio.getvalue())
        seq_img_summary = tf.Summary.Value(tag='Predictions/' + subset,
                                           image=seq_img)

        summary_str = tf.Summary(value=[seq_img_summary])
        cfg.sv.summary_computed(cfg.sess, summary_str, global_step=bidx)

    if cfg.save_gif_on_disk:
        save_animation_frame(fig2array(fig), subset, save_basedir)
    plt.close('all')

    # save predictions
    if cfg.save_raw_predictions_on_disk:
        # plt.imshow(y_pred, vmin=0, vmax=nclasses)
        # fpath = os.path.join('samples', model_name, 'predictions',
        #                      f)
        # if not os.path.exists(os.path.dirname(fpath)):
        #     os.makedirs(os.path.dirname(fpath))
        # plt.savefig(fpath)
        from PIL import Image
        img = Image.fromarray(y_pred.astype('uint8'))
        fpath = os.path.join(save_basedir, 'raw_predictions', subset, f)
        if not os.path.exists(os.path.dirname(fpath)):
            os.makedirs(os.path.dirname(fpath))
        img.save(fpath)
        del(img)


def save_animation_frame(frame, video_name, save_basedir):
    import imageio
    f = os.path.join(save_basedir, 'animations', video_name + '.gif')
    if not os.path.exists(os.path.dirname(f)):
        os.makedirs(os.path.dirname(f))
    imageio.imwrite(f, frame, duration=0.7)
