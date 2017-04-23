import numpy as np
import os
import time
from warnings import warn

import gflags
from tqdm import tqdm
import tensorflow as tf

from main_utils import compute_chunk_size


# Print prediction hotmap
def save_heatmap_fn(x, of, y_soft_pred, labels, nclasses, save_basedir,
                    subset, f, epoch_id, summary_writer=None):
    '''Save an image of the probability of each class

    Save the image and the heatmap of the probability of each class'''
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import AxesGrid
    from StringIO import StringIO

    cfg = gflags.cfg

    num_non_hot = 1 if of is None else 2
    nframes = y_soft_pred.shape[-1]
    fig = plt.figure(dpi=300)
    # Remove whitespace from around the image
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    grid = AxesGrid(fig, 111,
                    nrows_ncols=(4, max(num_non_hot, nframes // 4+1)),
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
                           grid[num_non_hot:]):
        im = ax.imshow(pred, cmap='hot', vmin=0, vmax=1,
                       interpolation='nearest')
        ax.set_title(l)
    # set the colorbar to match
    grid.cbar_axes[0].colorbar(im)
    for cax in grid.cbar_axes:
        cax.toggle_label(False)
    if cfg.save_images_on_disk:
        fpath = os.path.join(save_basedir, 'heatmaps', subset, f)
        if not os.path.exists(os.path.dirname(fpath)):
            os.makedirs(os.path.dirname(fpath))
        plt.savefig(fpath)  # save 3 subplots

    sio = StringIO()
    plt.imsave(sio, fig2array(fig), format='png')
    size = fig.get_size_inches()*fig.dpi  # size in pixels
    heatmap_img = tf.Summary.Image(encoded_image_string=sio.getvalue(),
                                   height=int(size[0]),
                                   width=int(size[1]))
    heatmap_img_summary = tf.Summary.Value(tag='Heatmaps',
                                           image=heatmap_img)
    summary = tf.Summary(value=[heatmap_img_summary])
    summary_writer.add_summary(summary, epoch_id)
    summary_writer.flush()

    plt.close('all')


def save_sample_and_fill_sequence_fn(raw_data, of, y_pred, y, cmap, nclasses,
                                     labels, subset, animations, save_basedir,
                                     f, epoch_id, summary_writer=None):
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

    if cfg.save_images_on_disk:
        fpath = os.path.join(save_basedir, 'segmentations', subset, f)
        if not os.path.exists(os.path.dirname(fpath)):
            os.makedirs(os.path.dirname(fpath))
        plt.savefig(fpath)  # save 3 subplots

    sio = StringIO()
    plt.imsave(sio, fig2array(fig), format='png')
    # size = fig.get_size_inches()*fig.dpi  # size in pixels
    seq_img = tf.Summary.Image(encoded_image_string=sio.getvalue())
    seq_img_summary = tf.Summary.Value(tag='GT/Predictions',
                                       image=seq_img)

    summary = tf.Summary(value=[seq_img_summary])
    summary_writer.add_summary(summary, epoch_id)
    summary_writer.flush()

    animations.setdefault(subset, []).append(fig2array(fig))
    plt.close('all')

    # save predictions
    if cfg.save_raw_predictions:
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


def save_images(this_set, x_batch, y_batch, f_batch, y_pred_batch,
                y_soft_batch, subset_batch, raw_data_batch, animations,
                save_basedir, epoch_id, summary_writer):
    import matplotlib as mpl
    import seaborn as sns

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

    # Save samples, iterating over each element of the batch
    for x, y, f, y_pred, y_soft_pred, subset, raw_data in zip(x_batch,
                                                              y_batch,
                                                              f_batch,
                                                              y_pred_batch,
                                                              y_soft_batch,
                                                              subset_batch,
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
        if cfg.save_heatmap:
            # do not pass optical flow
            save_heatmap_fn(heat_map_in, of, y_soft_pred,
                            labels, nclasses,
                            save_basedir, subset,
                            f, epoch_id,
                            summary_writer)

        # PRINT THE SAMPLES
        # Keep most likely prediction only
        # y = y.argmax(2)
        # y_pred = y_pred.argmax(2)

        # Save image and append frame to animations sequence
        if cfg.save_samples:
            if raw_data.ndim == 4:
                sample_in = raw_data[seq_length // 2]
                y_in = y[seq_length // 2]
            else:
                sample_in = raw_data
                y_in = y
            save_sample_and_fill_sequence_fn(sample_in, of, y_pred, y_in, cmap,
                                             nclasses, labels, subset,
                                             animations, save_basedir,
                                             f, epoch_id,
                                             summary_writer)
        return animations


def save_animations(animations, save_basedir):
    import imageio
    for k, v in animations.iteritems():
        f = os.path.join(save_basedir, 'animations', k + '.gif')
        if not os.path.exists(os.path.dirname(f)):
            os.makedirs(os.path.dirname(f))
        imageio.mimsave(f, v, duration=0.7)


def validate(placeholders,
             eval_outs,
             val_summary_op,
             sess,
             epoch_id,
             which_set='valid'):

        # from rec_conv_deconv import reset_states
        import tensorflow as tf

        cfg = gflags.cfg
        if getattr(cfg.valid_params, 'resize_images', False):
            warn('Forcing resize_images to False in evaluation.')
            cfg.valid_params.update({'resize_images': False})

        cfg.valid_params['batch_size'] *= cfg.num_gpus
        this_set = cfg.Dataset(
            which_set=which_set,
            **cfg.valid_params)
        save_basedir = os.path.join('samples', cfg.model_name,
                                    this_set.which_set)

        summary_writer = tf.summary.FileWriter(logdir=cfg.val_checkpoints_dir,
                                               graph=sess.graph)

        # Re-init confusion matrix
        cm = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES,
                               scope='mean_iou')
        sess.run([tf.variables_initializer(cm)])

        # Begin loop over dataset samples
        eval_cost = 0
        animations = {}
        pbar = tqdm(total=this_set.nbatches)
        for bidx in range(this_set.nbatches):
            start_val_time = time.time()

            ret = this_set.next()
            x_batch, y_batch = ret['data'], ret['labels']
            subset_batch = ret['subset']
            f_batch = ret['filenames']
            raw_data_batch = ret['raw_data']
            # Reset the state when we switch to a new video
            # if cfg.stateful_validation:
            #     if any(s == 'default' for s in subset_batch):
            #         raise RuntimeError(
            #             'For stateful validation, the validation dataset '
            #             'should provide `subset`')
            #     if any(last_subset != s for s in subset_batch):
            #         reset_states(model, x_batch.shape)
            #         last_subset = subset_batch[-1]
            # else:
            #     reset_states(model, x_batch.shape)

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

            if this_set.set_has_GT:
                # Class balance
                # class_balance_w = np.ones(np.prod(
                #     mini_x.shape[:3])).astype(floatX)
                # class_balance = loss_kwargs.get('class_balance', '')
                # if class_balance in ['median_freq_cost', 'rare_freq_cost']:
                #     w_freq = loss_kwargs.get('w_freq')
                #     class_balance_w = w_freq[y_true.flatten()].astype(floatX)

                # Create dictionary to feed the input placeholders:
                # and get batch pred, mIoU so far, batch loss
                in_values = [x_in, y_in, split_dim, lab_split_dim]
                feed_dict = {p: v for (p, v) in zip(placeholders, in_values)}
                (y_pred_batch, y_soft_batch, mIoU,
                 loss, _) = sess.run(eval_outs, feed_dict=feed_dict)
                # TODO summaries should not be repeated at each batch I
                # guess..
                summary_str = sess.run(val_summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, epoch_id)
                summary_writer.flush()
                # TODO valuta come fare aggregati sul loop in modo
                # simbolico per metterlo nei summary (o come mettere
                # robe nei summary a runtime)
                # i.e., vedi cosa restituisce summary_str
                eval_cost += loss

                eval_iter_el_time = time.time() - start_val_time
                pbar.set_description('Time: %f, Loss: %f, Mean IoU: %f' % (
                    eval_iter_el_time, loss, mIoU))
                pbar.update(1)

                # Save image summary for learning visualization
                if bidx % cfg.img_summaries_freq == 0:

                    animations = save_images(this_set, x_batch, y_batch,
                                             f_batch, y_pred_batch,
                                             y_soft_batch, subset_batch,
                                             raw_data_batch,
                                             animations, save_basedir,
                                             epoch_id, summary_writer)

            else:
                in_values = [x_in, y_in, split_dim, lab_split_dim]
                feed_dict = {p: v for (p, v) in zip(placeholders, in_values)}
                y_pred_batch = sess.run([eval_outs[0]], feed_dict=feed_dict)
                # TODO is the summary working in this case? I might not
                # be able to compute the e.g., metrics. Should I remove
                # that computation from the graph in build_graph?
                summary_str = sess.run(val_summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, epoch_id)
                summary_writer.flush()
                pbar.set_description('Time: %f' % (eval_iter_el_time))
                pbar.update(1)

            # animations = save_images(this_set, x_batch, y_batch, f_batch,
            #                          y_pred_batch, subset_batch,
            #                          raw_data_batch, animations,
            #                          save_basedir,
            #                          save_heatmap, save_samples,
            #                          save_raw_predictions)
        # Once all the batches have been processed, save animations
        # save_animations(animations, save_basedir)
        if cfg.save_animations:
            save_animations(animations, save_basedir)


def fig2array(fig):
    """Convert a Matplotlib figure to a 4D numpy array

    Params
    ------
    fig:
        A matplotlib figure

    Return
    ------
        A numpy 3D array of RGBA values

    Modified version of: http://www.icare.univ-lille1.fr/node/1141
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (h, w, 3)

    return buf
