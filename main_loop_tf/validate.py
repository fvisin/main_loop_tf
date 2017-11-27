try:
    from itertools import izip_longest as zip_longest
except:
    from itertools import zip_longest
import math
import numpy as np
import os
import shutil
try:
    import Queue
except ImportError:
    import queue as Queue
import threading
from warnings import warn

import gflags
from tqdm import tqdm
import tensorflow as tf
import json

from utils import split_in_chunks, fig2array
from evaluation import db_eval


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
        which_set='valid',
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

    if (epoch_id + 1) % 10 == 0 and epoch_id > 10:
        print('\nMemory limit reached. Deleting previous videos..')
        for i in range(epoch_id - 19, epoch_id - 9):
            if cfg.mask_refinement != '' and cfg.save_ref_videos:
                video_ref_dir = os.path.join(cfg.checkpoints_dir,
                                             'videos_ref',
                                             str(i))
                if os.path.exists(video_ref_dir):
                    shutil.rmtree(video_ref_dir)
            if cfg.save_rec_videos:
                video_rec_dir = os.path.join(cfg.checkpoints_dir,
                                             'videos_rec',
                                             str(i))
                if os.path.exists(video_rec_dir):
                    shutil.rmtree(video_rec_dir)
            if cfg.save_segm_videos:
                video_segm_dir = os.path.join(cfg.checkpoints_dir,
                                              'videos_segm',
                                              str(i))
                if os.path.exists(video_segm_dir):
                    shutil.rmtree(video_segm_dir)
            if cfg.save_of_videos:
                of_dir = os.path.join(cfg.checkpoints_dir,
                                      'videos_of',
                                      str(i))
                if os.path.exists(of_dir):
                    shutil.rmtree(of_dir)
            if (cfg.objectness_path or
                    cfg.warp_prev_objectness) and cfg.save_obj_videos:
                video_obj_dir = os.path.join(cfg.checkpoints_dir,
                                             'videos_obj',
                                             str(i))
                if os.path.exists(video_obj_dir):
                    shutil.rmtree(video_obj_dir)

    video_rec = []
    video_segm = []
    video_of = []
    video_obj = []
    video_ref = []
    video_ref_dir = os.path.join(cfg.checkpoints_dir, 'videos_ref',
                                 str(epoch_id))
    video_rec_dir = os.path.join(cfg.checkpoints_dir, 'videos_rec',
                                 str(epoch_id))
    video_segm_dir = os.path.join(cfg.checkpoints_dir, 'videos_segm',
                                  str(epoch_id))
    of_dir = os.path.join(cfg.checkpoints_dir, 'videos_of', str(epoch_id))
    video_obj_dir = os.path.join(cfg.checkpoints_dir, 'videos_obj',
                                 str(epoch_id))
    if cfg.mask_refinement != '' and cfg.save_ref_videos:
        if not os.path.exists(video_ref_dir):
            os.makedirs(video_ref_dir)
    if cfg.save_rec_videos:
        if not os.path.exists(video_rec_dir):
            os.makedirs(video_rec_dir)
    if cfg.save_segm_videos:
        if not os.path.exists(video_segm_dir):
            os.makedirs(video_segm_dir)
    if cfg.save_of_videos:
        if not os.path.exists(of_dir):
            os.makedirs(of_dir)
    if (cfg.objectness_path or
            cfg.warp_prev_objectness) and cfg.save_obj_videos:
        if not os.path.exists(video_obj_dir):
            os.makedirs(video_obj_dir)

    if cfg.eval_metrics or (epoch_id + 1) % cfg.metrics_freq == 0:
        subsets_list = []
        per_subset_segmentations = {}
        per_subset_annotations = {}

    if cfg.compute_mean_iou:
        prev_subset = None
        per_subset_IoUs = {}
        # Reset Confusion Matrix at the beginning of validation
        cfg.sess.run(reset_cm_op)

    cidx = (epoch_id*this_set.nbatches)
    frame_idx = cidx

    def save_videos(video_ref, video_rec, video_segm,
                    video_obj, video_of, prev_subset):
        if cfg.mask_refinement != '' and cfg.save_ref_videos:
            # write segmentation videos
            frames = np.array(video_ref)
            sdx = 2 if frames.ndim == 5 else 1
            frames = frames.reshape([-1] + list(frames.shape[sdx:]))
            # frames = (frames * 255.0).astype('uint8')
            fname = os.path.join(video_ref_dir, prev_subset + '.mp4')
            write_video(frames, fname, 15, codec='X264', mask=True)
        if cfg.save_rec_videos:
            # write reconstruction videos
            frames = np.array(video_rec)
            sdx = 2 if frames.ndim == 5 else 1
            frames = frames.reshape([-1] + list(frames.shape[sdx:]))
            fname = os.path.join(video_rec_dir, prev_subset + '.mp4')
            write_video(frames, fname, 15, codec='X264')

        if cfg.save_segm_videos:
            # write segmentation videos
            frames = np.array(video_segm)
            sdx = 2 if frames.ndim == 5 else 1
            frames = frames.reshape([-1] + list(frames.shape[sdx:]))
            # frames = (frames * 255.0).astype('uint8')
            fname = os.path.join(video_segm_dir, prev_subset + '.mp4')
            write_video(frames, fname, 15, codec='X264', mask=True)

        if (cfg.objectness_path or
                cfg.warp_prev_objectness) and cfg.save_obj_videos:
            # write segmentation videos
            frames = np.array(video_obj)
            sdx = 2 if frames.ndim == 5 else 1
            frames = frames.reshape([-1] + list(frames.shape[sdx:]))
            # frames = (frames * 255.0).astype('uint8')
            fname = os.path.join(video_obj_dir, prev_subset + '.mp4')
            write_video(frames, fname, 15, codec='X264', mask=True)

        if cfg.save_of_videos:
            # write OF videos
            of_frames = np.array(video_of)
            sdx = 2 if of_frames.ndim == 5 else 1
            of_frames = of_frames.reshape(
                [-1] + list(of_frames.shape[sdx:]))
            fname = os.path.join(of_dir, prev_subset + '.mp4')
            write_video(of_frames, fname, 15, codec='X264',
                        flowRGB=True)

    for bidx in range(this_set.nbatches):
        if cfg.sv.should_stop():  # Stop requested
            break

        ret = this_set.next()
        x_batch, y_batch = ret['data'], ret['labels']
        assert all(el == ret['subset'][0] for el in ret['subset'])
        subset = ret['subset']
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
            # Reset the first mask to be warped
            if not prev_subset or subset[0] != prev_subset:
                # Initialize annotations dictionary
                if cfg.eval_metrics or (epoch_id + 1) % cfg.metrics_freq == 0:
                    per_subset_annotations[subset[0]] = []
                if cfg.summary_per_subset:
                    tf.logging.info('Reset confusion matrix! {} --> {}'.format(
                        prev_subset, subset[0]))
                    cfg.sess.run(reset_cm_op)
                if cfg.stateful_validation:
                    if subset[0] == 'default':
                        raise RuntimeError(
                            'For stateful validation, the validation '
                            'dataset should provide `subset`')
                    # reset_states(model, x_batch.shape)
                # Get the first mask to be warped
                if cfg.target_frame == 'middle':
                    if cfg.valid_params['return_middle_frame_only']:
                        prev_pred_mask = y_batch
                    else:
                        prev_pred_mask = y_batch[:,
                                                 (cfg.seq_length // 2) - 1,
                                                 ...]
            if prev_subset is not None and subset[0] != prev_subset:
                # Write videos for each subset
                # ----------------------------
                save_videos(video_ref, video_rec, video_segm,
                            video_obj, video_of, prev_subset)
                # Save per-subset segmentations for evaluation
                if (cfg.eval_metrics or
                        (epoch_id + 1) % cfg.metrics_freq == 0):
                    if cfg.mask_refinement != '':
                        per_subset_segmentations[prev_subset] = video_ref
                    else:
                        per_subset_segmentations[prev_subset] = video_segm

                video_rec = []
                video_ref = []
                video_segm = []
                video_of = []
                video_obj = []
            prev_subset = subset[0]

        x_in = x_batch
        if cfg.seq_length:
            if cfg.target_frame == 'middle':
                if cfg.valid_params['return_middle_frame_only']:
                    y_in = np.zeros(shape=x_in.shape[:-1], dtype='int32')
                    y_in[:, cfg.seq_length // 2, ...] = y_batch

                    # Save per-subset annotations for evaluation
                    if (cfg.eval_metrics or
                            (epoch_id + 1) % cfg.metrics_freq == 0):
                        per_subset_annotations[subset[0]].append(
                            np.expand_dims(y_batch, axis=-1))
                else:
                    y_in = y_batch
                y_in[:, (cfg.seq_length // 2) - 1, ...] = prev_pred_mask
        else:
            y_in = y_batch

        x_batch_chunks, y_batch_chunks = split_in_chunks(x_in, y_in,
                                                         this_num_splits)

        # Save subsets name for the evalutation
        if cfg.eval_metrics or (epoch_id + 1) % cfg.metrics_freq == 0:
            if not subset[0] in subsets_list:
                subsets_list.append(subset[0])

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

        if cidx % cfg.val_summary_freq == 0:
            fetch_dict = cfg.sess.run(outs_summary, feed_dict=feed_dict)

            cfg.sv.summary_computed(cfg.sess,
                                    fetch_dict['summary_op'],
                                    global_step=cidx)
        else:
            fetch_dict = cfg.sess.run(outs, feed_dict=feed_dict)
            mIoU = 0

        if cfg.compute_mean_iou:
            mIoU = fetch_dict['m_iou']
            per_class_IoU = fetch_dict['per_class_iou']
            # If fg/bg, just consider the foreground class
            if len(per_class_IoU) == 2:
                per_class_IoU = per_class_IoU[1]

            # Save the IoUs per subset (i.e., video) and their average
            if cfg.summary_per_subset:
                per_subset_IoUs[subset[0]] = per_class_IoU
                mIoU = np.mean(per_subset_IoUs.values())

            pbar.set_postfix({'mIoU': '{:.3f}'.format(mIoU)})

        # TODO there is no guarantee that this will be processed
        # in order. We could use condition variables, e.g.,
        # http://python.active-venture.com/lib/condition-objects.html
        # Save image summary for learning visualization

        of_pred_fw_batch = fetch_dict.get('of_pred_fw', [None] * len(f_batch))
        of_pred_bw_batch = fetch_dict.get('of_pred_bw', [None] * len(f_batch))
        y_pred_batch = fetch_dict['pred']
        if cfg.objectness_path or cfg.warp_prev_objectness:
            obj_pred_batch = fetch_dict['obj_pred']
        y_pred_fw_batch = fetch_dict['pred_fw']
        y_pred_bw_batch = fetch_dict['pred_bw']
        y_pred_mask_batch = fetch_dict['pred_mask']
        # y_pred_mask_batch[np.where(y_pred_mask_batch > 0.5)] = 1
        # y_pred_mask_batch[np.where(y_pred_mask_batch < 1)] = 0
        if cfg.mask_refinement != '':
            y_refined_batch = fetch_dict['refined_mask']
        if cfg.masks_linear_interpolation or cfg.masks_interp_conv_layer:
            y_pred_mask_fw_batch = fetch_dict['pred_mask_fw']
            y_pred_mask_fw_batch[np.where(y_pred_mask_fw_batch > 0.5)] = 1
            y_pred_mask_fw_batch[np.where(y_pred_mask_fw_batch < 1)] = 0

            # Save the predicted mask to be warped in the next run
            if cfg.prev_segm_in_input == 'warped':
                prev_pred_mask = np.squeeze(y_pred_mask_fw_batch, axis=-1)
            elif cfg.prev_segm_in_input == 'interpolated':
                prev_pred_mask = np.squeeze(y_pred_mask_batch, axis=-1)
            else:
                raise NotImplementedError()
        elif cfg.mask_refinement != '':
            prev_pred_mask = np.squeeze(y_refined_batch, axis=-1)
        else:
            prev_pred_mask = np.squeeze(y_pred_mask_batch, axis=-1)
        blend_batch = fetch_dict['blend']
        y_prob_batch = fetch_dict['out_act']
        if cfg.mask_refinement != '' and cfg.save_ref_videos:
            video_ref.append(y_refined_batch)
        if cfg.save_rec_videos:
            video_rec.append(y_pred_batch)
        if (cfg.save_segm_videos or
                cfg.eval_metrics or (epoch_id + 1) % cfg.metrics_freq == 0):
            video_segm.append(y_pred_mask_batch)
        if cfg.save_of_videos:
            video_of.append(of_pred_fw_batch)
        if (cfg.objectness_path or
                cfg.warp_prev_objectness) and cfg.save_obj_videos:
            video_obj.append(obj_pred_batch)
        if cfg.show_image_summaries_validation:
            img_queue.put((frame_idx, this_set, x_batch, y_in, f_batch, subset,
                           raw_data_batch, of_pred_fw_batch, of_pred_bw_batch,
                           y_pred_batch, y_pred_fw_batch, y_pred_bw_batch,
                           y_pred_mask_batch, blend_batch, y_prob_batch))
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
        # Save the metrics for the last subset
        save_videos(video_ref, video_rec, video_segm,
                    video_obj, video_of, prev_subset)

        # Save per-subset segmentation for evaluation
        if cfg.eval_metrics or (epoch_id + 1) % cfg.metrics_freq == 0:
            if cfg.mask_refinement != '':
                per_subset_segmentations[prev_subset] = video_ref
            else:
                per_subset_segmentations[prev_subset] = video_segm
            # START EVALUATION #
            evaluation = db_eval(subsets_list,
                                 per_subset_segmentations,
                                 per_subset_annotations,
                                 measures=cfg.measures,
                                 statistics=cfg.statistics,
                                 n_jobs=cfg.eval_n_jobs,
                                 verbose=True)

            exp_hash = cfg.checkpoints_dir.split('/')[-1]
            results_dir = os.path.join(cfg.results_path,
                                       'results', which_set, exp_hash)
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            results_file = os.path.join(results_dir,
                                        'results'+str(epoch_id)+'.json')
            with open(results_file, 'w') as f:
                json.dump(evaluation, f)
            # Get mean IoU for early stopping
            mIoU = evaluation['dataset']['J']['mean']

        video_rec = []
        video_segm = []
        video_obj = []
        video_of = []
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
             raw_data_batch, of_pred_fw_batch, of_pred_bw_batch,
             y_pred_batch, y_pred_fw_batch, y_pred_bw_batch, y_pred_mask_batch,
             blend_batch, y_prob_batch) = img

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

            if not np.all(of_pred_fw_batch):
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
                           len(of_pred_fw_batch), len(raw_data_batch))
                assert all(el == lengths[0] for el in lengths), (
                    'x_batch: {}\ny_batch: {}\nf_batch: {}\ny_pred_batch: {}'
                    '\ny_prob_batch: {}\nof_pred_batch: {}'
                    '\nraw_data_batch: {}'.format(*lengths))

            zip_list = (x_batch, y_batch, f_batch, subset, of_pred_fw_batch,
                        of_pred_bw_batch, y_pred_batch, y_pred_fw_batch,
                        y_pred_bw_batch, y_pred_mask_batch,
                        blend_batch, y_prob_batch, raw_data_batch)

            # Save samples, iterating over each element of the batch
            for el in zip(*zip_list):
                (x, y, f, subset, of_pred_fw, of_pred_bw, y_pred,
                 y_pred_fw, y_pred_bw, y_pred_mask, blend,
                 y_prob, raw_data) = el

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

                # TODO Process OF to create vector field representation
                # Predicted forward OF
                if of_pred_fw is not None:
                    of_rgb_fw = flowToColor(of_pred_fw,
                                            raw_data[which_frame - 1],
                                            cfg.show_flow_vector_field)
                    of_rgb_fw = cv2.resize(of_rgb_fw, (y_pred.shape[1],
                                                       y_pred.shape[0]))
                else:
                    of_rgb_fw = None

                # Predicted backward OF
                if of_pred_bw is not None:
                    of_rgb_bw = flowToColor(of_pred_bw,
                                            raw_data[which_frame + 1],
                                            cfg.show_flow_vector_field)
                    of_rgb_bw = cv2.resize(of_rgb_bw, (y_pred.shape[1],
                                                       y_pred.shape[0]))
                else:
                    of_rgb_bw = None

                # Predicted forward OF as vector field
                if of_pred_fw is not None:
                    of_vect_field_fw = flowToColor(of_pred_fw,
                                                   raw_data[which_frame - 1],
                                                   True)
                    of_vect_field_fw = cv2.resize(of_vect_field_fw, (y_pred.shape[1],
                                                  y_pred.shape[0]))
                else:
                    of_vect_field_fw = None

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
                        sample_in_fw = raw_data[which_frame - 1]
                        sample_in_bw = raw_data[which_frame + 1]
                        if y.shape[0] > 1:
                            y_in = y[which_frame]
                            y_pre = y[which_frame - 1]
                        else:
                            y_in = y[0]
                            y_pre = y_in
                    else:
                        sample_in = raw_data
                        y_in = y
                    save_samples_and_animations(sample_in, sample_in_fw,
                                                sample_in_bw, of, of_rgb_fw,
                                                of_rgb_bw, of_vect_field_fw,
                                                y_pred, y_pred_fw,
                                                y_pred_bw, y_pred_mask, blend,
                                                y_in, y_pre, cmap, nclasses,
                                                labels, subset, save_basedir,
                                                f, bidx)
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


def save_samples_and_animations(raw_data_gt, raw_data_fw, raw_data_bw, of,
                                of_pred_fw, of_pred_bw, of_vect_field_fw,
                                y_pred, y_pred_fw, y_pred_bw, y_pred_mask,
                                blend, y, y_pre, cmap, nclasses, labels,
                                subset, save_basedir, f, bidx):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import AxesGrid
    from StringIO import StringIO
    import matplotlib.gridspec as gridspec

    cfg = gflags.cfg

    fig = plt.figure(dpi=600)
    # Remove whitespace from around the image
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0, top=1)

    # Set number of rows
    # if cfg.model_returns_of:
    #     n_rows = 5
    #     n_cols = 3
    # else:
    #     n_rows = 2
    #     n_cols = 2

    gs = gridspec.GridSpec(12, 12)

    if raw_data_gt.shape[-1] == 1:
        raw_data_cmap = 'gray'
    else:
        raw_data_cmap = None
    # GT-Mask
    # im = None
    ax = plt.subplot(gs[:3, :3])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(np.squeeze(y), cmap=cmap, vmin=0, vmax=nclasses)
    ax.set_title('GT Mask')
    # starting mask
    ax = plt.subplot(gs[:3, 3:6])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(np.squeeze(y_pre), cmap=cmap, vmin=0, vmax=nclasses)
    ax.set_title('Mask t-1')
    # mask prediction
    ax = plt.subplot(gs[:3, 6:9])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(np.squeeze(y_pred_mask), cmap=cmap, vmin=0, vmax=nclasses)
    ax.set_title('MaskPred')
    # Mask OF
    ax = plt.subplot(gs[0:3, 9:])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(of_vect_field_fw)
    ax.set_title('FW OF Vector Field')
    # GT-Frame
    ax = plt.subplot(gs[3:6, :3])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(np.squeeze(raw_data_gt), cmap=raw_data_cmap)
    ax.set_title('GT Frame')
    # blended prediction
    ax = plt.subplot(gs[3:6, 3:6])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(np.squeeze(y_pred), cmap=cmap, vmin=0, vmax=nclasses)
    ax.set_title('JointPred')
    # Difference between gt-frame and blended prediction
    ax = plt.subplot(gs[3:6, 6:9])
    ax.set_xticks([])
    ax.set_yticks([])
    abs_diff = np.squeeze(np.abs(raw_data_gt - y_pred))
    ax.imshow(abs_diff, vmin=0, vmax=1, interpolation='nearest')
    ax.set_title('GT-JointPred')
    # Blend heatmap
    ax = plt.subplot(gs[3:6, 9:])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(np.squeeze(blend), cmap='hot', vmin=0, vmax=1,
              interpolation='nearest')
    ax.set_title('Blend Heatmap')
    # Frame t-1
    ax = plt.subplot(gs[6:9, :3])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(np.squeeze(raw_data_fw), cmap=raw_data_cmap)
    ax.set_title('Frame t-1')
    # forward prediction
    ax = plt.subplot(gs[6:9, 3:6])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(np.squeeze(y_pred_fw), cmap=cmap, vmin=0, vmax=nclasses)
    ax.set_title('FWPred')
    # Difference between gt-frame and forward prediction
    ax = plt.subplot(gs[6:9, 6:9])
    ax.set_xticks([])
    ax.set_yticks([])
    abs_diff = np.squeeze(np.abs(raw_data_gt - y_pred_fw))
    ax.imshow(abs_diff, vmin=0, vmax=1, interpolation='nearest')
    ax.set_title('GT-FWPred')
    # Forward OF
    ax = plt.subplot(gs[6:9, 9:])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(of_pred_fw)
    ax.set_title('FW Predicted OF')
    # Frame t+1
    ax = plt.subplot(gs[9:12, :3])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(np.squeeze(raw_data_bw), cmap=raw_data_cmap)
    ax.set_title('Frame t+1')
    # backward prediction
    ax = plt.subplot(gs[9:12, 3:6])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(np.squeeze(y_pred_bw), cmap=cmap, vmin=0, vmax=nclasses)
    ax.set_title('BWPred')
    # Difference between gt-frame and backward prediction
    ax = plt.subplot(gs[9:12, 6:9])
    ax.set_xticks([])
    ax.set_yticks([])
    abs_diff = np.squeeze(np.abs(raw_data_gt - y_pred_bw))
    ax.imshow(abs_diff, vmin=0, vmax=1, interpolation='nearest')
    ax.set_title('GT-BWPred')
    # Backward OF
    ax = plt.subplot(gs[9:12, 9:])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(of_pred_bw)
    ax.set_title('BW Predicted OF')
    # grid = AxesGrid(fig, 111,
    #                 nrows_ncols=(n_rows, n_cols),
    #                 axes_pad=0.50,
    #                 share_all=True,
    #                 label_mode="L",
    #                 cbar_location="right",
    #                 cbar_mode="single")
    # sh = raw_data_gt.shape
    # for ax in grid:
    #     ax.set_xticks([sh[1]])
    #     ax.set_yticks([sh[0]])

    # # This element is not used
    # # if cfg.model_returns_of:
    # # grid[5].set_visible(False)
    # # image
    # if raw_data_gt.shape[-1] == 1:
    #     raw_data_cmap = 'gray'
    # else:
    #     raw_data_cmap = None
    # # GT-Frame
    # grid[0].axhspan(0, 1, 0, 2)
    # grid[0].imshow(np.squeeze(raw_data_gt), cmap=raw_data_cmap)
    # grid[0].set_title('GT-Frame')
    # # GT
    # im = None
    # if y is not None:
    #     im = grid[1].imshow(np.squeeze(y), cmap=cmap, vmin=0, vmax=nclasses)
    #     grid[1].set_title('Ground truth')
    # else:
    #     grid[1].set_visible(False)

    # grid[3].set_visible(False)
    # # blended prediction
    # grid[4].imshow(np.squeeze(y_pred), cmap=cmap, vmin=0, vmax=nclasses)
    # grid[4].set_title('JointPred')
    # # Difference between gt-frame and blended prediction
    # abs_diff = np.squeeze(np.abs(raw_data_gt - y_pred))
    # grid[5].imshow(abs_diff, vmin=0, vmax=1, interpolation='nearest')
    # grid[5].set_title('GT-JointPred')
    # # Frame t-1
    # grid[6].imshow(np.squeeze(raw_data_fw), cmap=raw_data_cmap)
    # grid[6].set_title('Starting Frame')
    # # forward prediction
    # grid[7].imshow(np.squeeze(y_pred_fw), cmap=cmap, vmin=0, vmax=nclasses)
    # grid[7].set_title('FWPred')
    # # Difference between gt-frame and forward prediction
    # abs_diff = np.squeeze(np.abs(raw_data_gt - y_pred_fw))
    # grid[8].imshow(abs_diff, vmin=0, vmax=1, interpolation='nearest')
    # grid[8].set_title('GT-FWPred')
    # # Frame t+1
    # grid[9].imshow(np.squeeze(raw_data_bw), cmap=raw_data_cmap)
    # grid[9].set_title('Starting Frame')
    # # backward prediction
    # grid[10].imshow(np.squeeze(y_pred_bw), cmap=cmap, vmin=0, vmax=nclasses)
    # grid[10].set_title('BWPred')
    # # Difference between gt-frame and backward prediction
    # abs_diff = np.squeeze(np.abs(raw_data_gt - y_pred_bw))
    # grid[11].imshow(abs_diff, vmin=0, vmax=1, interpolation='nearest')
    # grid[11].set_title('GT-BWPred')
    # # Forward OF
    # grid[12].imshow(of_pred_fw)
    # grid[12].set_title('FW Predicted OF')
    # # Backward OF
    # grid[13].imshow(of_pred_bw)
    # grid[13].set_title('BW Predicted OF')
    # # Blend heatmap
    # grid[14].imshow(np.squeeze(blend), cmap='hot', vmin=0, vmax=1,
    #                 interpolation='nearest')
    # grid[14].set_title('Blend Heatmap')
    # # set the colorbar to match GT or prediction
    # grid.cbar_axes[0].colorbar(im)
    # # grid[11].set_visible(False)
    # for cax in grid.cbar_axes:
    #     cax.toggle_label(True)  # show labels
    #     cax.set_yticks(np.arange(len(labels)) + 0.5)
    #     cax.set_yticklabels(labels)

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


def write_video(frames, fname, fps, codec='X264', mask=False, flowRGB=False):
    """
    Utility function to serialize a 4D numpy tensor to video.

    Inputs:
        frames: 4D numpy array
        fname: output filename
        fps: frame rate of the ouput video
        codec: 4 digit string for the codec, default='H264'
    Returns:
        no return value
    """
    cfg = gflags.cfg

    import cv2
    from utils import flowToColor
    # http://www.pyimagesearch.com/2016/02/22/writing-to-video-with-opencv/
    fourcc = cv2.VideoWriter_fourcc(*codec)

    h, w = frames.shape[1:3]
    if not cfg.generate_images:
        writer = cv2.VideoWriter(fname, fourcc, fps, (w, h), True)
    else:
        image_frames = []
    for f in frames:
        if flowRGB:
            f = flowToColor(f, None, show_flow_vector_field=False)
        if mask:
            f = cv2.cvtColor(np.float32(f), cv2.COLOR_GRAY2BGR)
        else:
            f = cv2.cvtColor(np.float32(f), cv2.COLOR_RGB2BGR)
        f = (f * 255.0)
        f = f.astype('uint8')
        if not cfg.generate_images:
            writer.write(f)
        else:
            f = cv2.copyMakeBorder(f, 2, 2, 2, 2, cv2.BORDER_CONSTANT,
                                   value=[255, 255, 255])
            image_frames.append(f)
    if not cfg.generate_images:
        writer.release()
    else:
        cv2.imwrite(fname[:-4] + '.png',
                    np.concatenate(image_frames[20:40], axis=0))

def save_animation_frame(frame, video_name, save_basedir):
    import imageio
    f = os.path.join(save_basedir, 'animations', video_name + '.gif')
    if not os.path.exists(os.path.dirname(f)):
        os.makedirs(os.path.dirname(f))
    imageio.imwrite(f, frame, duration=0.7)
