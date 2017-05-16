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

from utils import compute_chunk_size, fig2array


def validate(placeholders,
             eval_outs,
             val_summary_op,
             val_reset_cm_op,
             which_set='valid',
             epoch_id=None):

    cfg = gflags.cfg
    if getattr(cfg.valid_params, 'resize_images', False):
        warn('Forcing resize_images to False in evaluation.')
        cfg.valid_params.update({'resize_images': False})

    cfg.valid_params['batch_size'] *= cfg.num_splits
    this_set = cfg.Dataset(
        which_set=which_set,
        **cfg.valid_params)
    save_basedir = os.path.join('samples', cfg.model_name,
                                this_set.which_set)
    img_queue = Queue.Queue(maxsize=10)
    for _ in range(2):
        t = threading.Thread(
            target=save_images,
            args=(img_queue, save_basedir))
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
    prev_subset = None
    mIoUs = {}
    # Reset Confusion Matrix at the beginning of validation
    cfg.sess.run(val_reset_cm_op)

    for bidx in range(this_set.nbatches):
        if cfg.sv.should_stop():
            break
        cidx = (epoch_id*this_set.nbatches) + bidx

        ret = this_set.next()
        x_batch, y_batch = ret['data'], ret['labels']
        subset = ret['subset'][0]
        f_batch = ret['filenames']
        raw_data_batch = ret['raw_data']

        # Reset the confusion matrix if we are switching video
        if (this_set.set_has_GT and
           (not prev_subset or subset != prev_subset) and
           cfg.summary_per_subset):
                tf.logging.info('Reset confusion matrix! {} --> {}'.format(
                    prev_subset, subset))
                cfg.sess.run(val_reset_cm_op)
                if cfg.stateful_validation:
                    if subset == 'default':
                        raise RuntimeError(
                            'For stateful validation, the validation '
                            'dataset should provide `subset`')
                    pass
                    # reset_states(model, x_batch.shape)
                prev_subset = subset

        # TODO remove duplication of code
        # Compute the shape of the input chunk for each GPU
        split_dim, lab_split_dim = compute_chunk_size(
            x_batch.shape[0], np.prod(this_set.data_shape[:2]))

        if cfg.seq_length and cfg.seq_length > 1:
            x_in = x_batch
            y_in = y_batch[:, cfg.seq_length // 2, ...]  # 4D: not one-hot
        else:
            x_in = x_batch
            y_in = y_batch

        # if cfg.use_second_path:
        #     x_in = [x_in[..., :3], x_in[..., 3:]]
        y_in = y_in.flatten()
        in_values = [x_in, y_in, split_dim, lab_split_dim]
        feed_dict = {p: v for (p, v) in zip(placeholders, in_values)}

        if this_set.set_has_GT:
            # Class balance
            # class_balance_w = np.ones(np.prod(
            #     mini_x.shape[:3])).astype(floatX)
            # class_balance = loss_kwargs.get('class_balance', '')
            # if class_balance in ['median_freq_cost', 'rare_freq_cost']:
            #     w_freq = loss_kwargs.get('w_freq')
            #     class_balance_w = w_freq[y_true.flatten()].astype(floatX)

            # Get batch pred, mIoU so far, batch loss and potentially the
            # summary
            if cidx % cfg.val_summary_freq == 0:
                (y_pred_batch, y_soft_batch, mIoU, per_class_IoU,
                 loss, _, summary_str) = cfg.sess.run(
                 eval_outs + [val_summary_op], feed_dict=feed_dict)
                cfg.sv.summary_computed(cfg.sess, summary_str,
                                        global_step=cidx)
            else:
                    (y_pred_batch, y_soft_batch, mIoU, per_class_IoU,
                     loss, _) = cfg.sess.run(eval_outs, feed_dict=feed_dict)
            tot_loss += loss
            # mIoU is computed incrementally, so we just need the
            # last value

            # When we have videos we want the IoU per subset and
            # then average them
            if cfg.summary_per_subset:
                # we just consider foreground class
                mIoUs[subset] = mIoU = per_class_IoU[1]

            pbar.set_postfix({
                'val loss': '{:.3f}({:.3f})'.format(loss,
                                                    tot_loss/(bidx+1)),
                'mIoU': '{:.3f}'.format(mIoU)})
        else:
            y_pred_batch, y_soft_batch, summary_str = cfg.sess.run(
                eval_outs[:2] + [val_summary_op], feed_dict=feed_dict)
            mIoU = 0
            summary_str = cfg.sess.run(val_summary_op, feed_dict=feed_dict)
            cfg.sv.summary_computed(cfg.sess, summary_str, global_step=cidx)
        pbar.update(1)
        # TODO there is no guarantee that this will be processed
        # in order. We could use condition variables, e.g.,
        # http://python.active-venture.com/lib/condition-objects.html
        #
        # Save image summary for learning visualization
        img_queue.put((cidx, this_set, x_batch, y_batch, f_batch, subset,
                       raw_data_batch, y_pred_batch, y_soft_batch))
    pbar.close()

    if cfg.summary_per_subset:
        # Write the last video summary
        write_subset_summary(mIoUs, step=epoch_id)
        # Compute the aggregate metrics
        mean_IoU = np.mean(mIoUs.values())
        mIou_summary = tf.Summary.Value(tag='mIoUs/mean_per_video_IoU',
                                        simple_value=mean_IoU)
        summary_str = tf.Summary(value=[mIou_summary])
        cfg.sv.summary_computed(cfg.sess, summary_str, global_step=epoch_id)

    else:
        # Write meanIou summary
        mIou_summary = tf.Summary.Value(tag='mIoUs/mIoU',
                                        simple_value=mIoU)
        summary_str = tf.Summary(value=[mIou_summary])
        cfg.sv.summary_computed(cfg.sess, summary_str, global_step=epoch_id)

        # Per class IoU summaries
        for iou, label_name in zip(
         per_class_IoU, this_set.mask_labels[:this_set.nclasses]):

            per_class_iou_summary = tf.Summary.Value(
                tag='mIoUs/per_class_IoU_{}'.format(label_name),
                simple_value=iou)
            summary_str = tf.Summary(value=[per_class_iou_summary])
            cfg.sv.summary_computed(cfg.sess, summary_str,
                                    global_step=epoch_id)
        mean_IoU = mIoU

    this_set.finish()
    return mean_IoU


def write_subset_summary(mIoUs, subset=None, step=None):
    cfg = gflags.cfg
    if subset:
        mIou_summary = tf.Summary.Value(tag='mIoUs/' + subset,
                                        simple_value=mIoUs[subset])
        summary_str = tf.Summary(value=[mIou_summary])
        cfg.sv.summary_computed(cfg.sess, summary_str, global_step=step)
    else:
        for subset, mIoU in mIoUs.iteritems():
            mIou_summary = tf.Summary.Value(tag='mIoUs/' + subset,
                                            simple_value=mIoU)
            summary_str = tf.Summary(value=[mIou_summary])
            cfg.sv.summary_computed(cfg.sess, summary_str, global_step=step)


def save_images(img_queue, save_basedir):
    import matplotlib as mpl
    import seaborn as sns
    cfg = gflags.cfg

    while True:
        if cfg.sv.should_stop() and img_queue.empty():
            break
        try:
            (bidx, this_set, x_batch, y_batch, f_batch, subset,
             raw_data_batch, y_pred_batch, y_soft_batch) = img_queue.get(False)

            cfg = gflags.cfg

            # Initialize variables
            nclasses = this_set.nclasses
            seq_length = this_set.seq_length
            try:
                cmap = this_set.cmap
            except AttributeError:
                cmap = [el for el in sns.hls_palette(this_set.nclasses)]
            cmap = mpl.colors.ListedColormap(cmap)
            labels = this_set.mask_labels

            assert len(x_batch) == len(y_batch) == len(f_batch) == \
                len(y_pred_batch) == len(y_soft_batch) == len(raw_data_batch)
            # Save samples, iterating over each element of the batch
            for x, y, f, y_pred, y_soft_pred, raw_data in zip(
                    x_batch,
                    y_batch,
                    f_batch,
                    y_pred_batch,
                    y_soft_batch,
                    raw_data_batch):
                # y = np.expand_dims(y, -1)
                # y_pred = np.expand_dims(y_pred, -1)
                if x.shape[-1] == 5:
                    # Keep only middle frame name and save as png
                    seq_length = x_batch.shape[1]
                    f = f[seq_length // 2]
                    f = f[:-4]  # strip .jpg
                    f = f + '.png'
                else:
                    f = f[0]
                    f = f[:-4]
                    f = f + '.png'
                # Retrieve the optical flow channels
                if x.shape[-1] == 5:
                    of = x[seq_length // 2, ..., 3:]
                    # ang, mag = of
                    import cv2
                    hsv = np.zeros_like(x[seq_length // 2, ..., :3],
                                        dtype='uint8')
                    hsv[..., 0] = of[..., 0] * 255
                    hsv[..., 1] = 255
                    hsv[..., 2] = cv2.normalize(of[..., 1] * 255, None, 0, 255,
                                                cv2.NORM_MINMAX)
                    of = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                else:
                    of = None

                if raw_data.ndim == 4:
                    # Show only the middle frame
                    heat_map_in = raw_data[seq_length // 2, ..., :3]
                else:
                    heat_map_in = raw_data

                # PRINT THE HEATMAP
                if cfg.show_heatmaps_summaries:
                    # do not pass optical flow
                    save_heatmap_fn(heat_map_in, of, y_soft_pred, labels,
                                    nclasses, save_basedir, subset, f, bidx)

                # PRINT THE SAMPLES
                # Keep most likely prediction only
                # y = y.argmax(2)
                # y_pred = y_pred.argmax(2)

                # Save image and append frame to animations sequence
                if (cfg.save_gif_frames_on_disk or
                        cfg.show_samples_summaries or cfg.save_gif_on_disk):
                    if raw_data.ndim == 4:
                        sample_in = raw_data[seq_length // 2]
                        y_in = y[seq_length // 2]
                    else:
                        sample_in = raw_data
                        y_in = y
                    save_samples_and_animations(sample_in, of, y_pred, y_in,
                                                cmap, nclasses, labels, subset,
                                                save_basedir, f, bidx)
            img_queue.task_done()
        except Queue.Empty:
            continue
        except Exception as e:
            # Do not crash for errors during image saving
            # cfg.sv.coord.request_stop(e)
            # raise
            # break
            tf.logging.error('Error in save_images!! ' + str(e))
            continue


def save_heatmap_fn(x, of, y_soft_pred, labels, nclasses, save_basedir, subset,
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
    for l, pred, ax in zip(labels[:nclasses-1], y_soft_pred.transpose(2, 0, 1),
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


def save_samples_and_animations(raw_data, of, y_pred, y, cmap, nclasses,
                                labels, subset, save_basedir, f, bidx):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import AxesGrid
    from StringIO import StringIO

    cfg = gflags.cfg

    fig = plt.figure(dpi=300)
    # Remove whitespace from around the image
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0, top=1)

    # Set number of rows
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

    # image
    grid[0].imshow(raw_data)
    grid[0].set_title('Image')
    # prediction
    grid[2].imshow(y_pred, cmap=cmap, vmin=0, vmax=nclasses)
    grid[2].set_title('Prediction')
    im = None
    # OF
    if of is not None:
        im = grid[3].imshow(of, vmin=0, vmax=1, interpolation='nearest')
        grid[3].set_title('Optical flow')
    else:
        grid[3].set_visible(False)
    # GT
    if y is not None:
        im = grid[1].imshow(y, cmap=cmap, vmin=0, vmax=nclasses)
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
