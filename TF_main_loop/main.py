from copy import deepcopy
import hashlib
import os
import sys
from time import time

import dataset_loaders
import numpy as np
import tensorflow as tf
from tensorflow import nn
from tensorflow.contrib import slim
from tensorflow.python.training import training
from tqdm import tqdm

import gflags

from main_utils import (apply_loss, average_gradients, compute_chunk_size,
                        save_repos_hash)
from config import dataset, flow, optimization, misc  # noqa


FLAGS = gflags.FLAGS
gflags.DEFINE_bool('help', False, 'If True, shows this message')
gflags.DEFINE_bool('debug', False, 'If True, enable tensorflow debug')
gflags.DEFINE_bool('return_extended_sequences', False, 'If True, repeats '
                   'the first and last frame of each video to allow for '
                   'middle frame prediction')
gflags.DEFINE_bool('return_middle_frame_only', False, 'If True, return '
                   'the middle frame segmentation mask only for each sequence')
gflags.DEFINE_string('model_name', 'my_model', 'The name of the model, '
                     'for the checkpoint file')


def run(argv, build_model):
    __parse_config(argv)
    # Run main with the remaining arguments
    __run(build_model)


def __parse_config(argv=None):
    gflags.mark_flags_as_required(['dataset'])

    # ============ Manage gflags
    # Parse FLAGS
    try:
        FLAGS(argv)  # parse flags
    except gflags.FlagsError as e:
        print('Usage: %s ARGS\n%s\n\nError: %s' % (argv[0], FLAGS, e))
        sys.exit(0)

    # Show help message
    if FLAGS.help:
        print('%s' % FLAGS)
        sys.exit(0)

    # Convert FLAGS to namespace, so we can modify it
    from argparse import Namespace
    cfg = Namespace()
    fl = FLAGS.FlagDict()
    cfg.__dict__ = {k: el.value for (k, el) in fl.iteritems()}
    gflags.cfg = cfg

    # ============ gsheet
    # Save params for log, excluding non JSONable and not interesting objects
    exclude_list = ['checkpoints_dir', 'checkpoints_to_keep', 'dataset',
                    'debug', 'devices', 'do_validation_only', 'help',
                    'min_epochs', 'max_epochs', 'nthreads', 'num_gpus',
                    'patience', 'restore_model', 'use_threads', 'val_on_sets',
                    'val_skip_first', 'val_every_epochs' 'vgg_weights_file']
    param_dict = {k: deepcopy(v) for (k, v) in cfg.__dict__.iteritems()
                  if k not in exclude_list}
    h = hashlib.md5()
    h.update(str(param_dict))
    h = h.hexdigest()
    cfg.hash = h
    save_repos_hash(param_dict, cfg.model_name, ['tensorflow',
                                                 'dataset_loaders',
                                                 'TF_main_loop'])
    cfg.checkpoints_dir = os.path.join(cfg.checkpoints_dir, cfg.model_name,
                                       cfg.hash)
    cfg.train_checkpoints_dir = os.path.join(cfg.checkpoints_dir, 'train')
    cfg.val_checkpoints_dir = os.path.join(cfg.checkpoints_dir, 'valid')
    cfg.checkpoints_file = 'best.ckpt'

    # ============ A bunch of derived params
    cfg._FLOATX = 'float32'
    cfg.num_gpus = len([el for el in cfg.devices if 'gpu' in el])

    # Dataset
    try:
        Dataset = getattr(dataset_loaders, cfg.dataset)
    except AttributeError:
        Dataset = getattr(dataset_loaders, cfg.dataset.capitalize() +
                          'Dataset')
    cfg.Dataset = Dataset
    dataset_params = {}
    dataset_params['batch_size'] = cfg.batch_size
    dataset_params['data_augm_kwargs'] = {}
    dataset_params['data_augm_kwargs']['crop_size'] = cfg.crop_size
    dataset_params['data_augm_kwargs']['return_optical_flow'] = cfg.of
    dataset_params['return_one_hot'] = False
    dataset_params['return_01c'] = True
    if cfg.seq_per_subset:
        dataset_params['seq_per_subset'] = cfg.seq_per_subset
    if cfg.overlap is not None:
        dataset_params['overlap'] = cfg.overlap
    if cfg.seq_length:
        dataset_params['seq_length'] = cfg.seq_length
        cfg.input_shape = [None, cfg.seq_length, None, None, 3]
        if Dataset.data_shape:
            cfg.val_input_shape = (None, cfg.seq_length) + Dataset.data_shape
        else:
            cfg.val_input_shape = [None, cfg.seq_length, None, None, 3]
        if cfg.crop_size:
            cfg.input_shape[2:4] = cfg.crop_size
        ret_ext_seq = cfg.return_extended_sequences
        ret_middle_frame = cfg.return_middle_frame_only
        dataset_params['return_extended_sequences'] = ret_ext_seq
        dataset_params['return_middle_frame_only'] = ret_middle_frame
    else:
        cfg.input_shape = [None, None, None, 3]
        if Dataset.data_shape:
            cfg.val_input_shape = (None,) + Dataset.data_shape
        else:
            cfg.val_input_shape = [None, None, None, 3]
        if cfg.crop_size:
            cfg.input_shape[1:3] = cfg.crop_size
    dataset_params['use_threads'] = cfg.use_threads
    dataset_params['nthreads'] = cfg.nthreads
    dataset_params['remove_per_img_mean'] = cfg.remove_per_img_mean
    dataset_params['divide_by_per_img_std'] = cfg.divide_by_per_img_std
    dataset_params['remove_mean'] = cfg.remove_mean
    dataset_params['divide_by_std'] = cfg.divide_by_std
    cfg.dataset_params = dataset_params
    cfg.valid_params = deepcopy(cfg.dataset_params)
    cfg.valid_params.update({
        'batch_size': cfg.val_batch_size,
        'seq_per_subset': 0,
        'overlap': cfg.val_overlap,
        'shuffle_at_each_epoch': (cfg.val_overlap is not None and
                                  cfg.val_overlap != 0),
        'return_middle_frame_only': False,
        'use_threads': False,  # prevent shuffling
        # prevent crop
        'data_augm_kwargs': {'return_optical_flow': cfg.of}})
    cfg.void_labels = getattr(Dataset, 'void_labels', [])
    cfg.nclasses = Dataset.non_void_nclasses
    cfg.nclasses_w_void = Dataset.nclasses
    print('{} classes ({} non-void):'.format(cfg.nclasses_w_void,
                                             cfg.nclasses))

    # Optimization
    try:
        Optimizer = getattr(training, cfg.optimizer)
    except AttributeError:
        Optimizer = getattr(training, cfg.optimizer.capitalize() + 'Optimizer')
    cfg.optimizer = Optimizer(**cfg.optimizer_params)
    try:
        loss_fn = getattr(nn, cfg.loss_fn)
    except AttributeError:
        loss_fn = getattr(nn, cfg.loss_fn.capitalize())
    cfg.loss_fn = loss_fn

    # TODO Add val_every_iter?
    cfg.val_skip = (cfg.val_skip_first if cfg.val_skip_first else
                    max(1, cfg.val_every_epochs) - 1)


def __run(build_model):
    cfg = gflags.cfg

    # ============ Class balance
    # assert class_balance in [None, 'median_freq_cost', 'rare_freq_cost'], (
    #     'The balance class method is not implemented')

    # if class_balance in ['median_freq_cost', 'rare_freq_cost']:
    #     if not hasattr(Dataset, 'class_freqs'):
    #         raise RuntimeError('class_freqs is missing for dataset '
    #                            '{}'.format(Dataset.name))
    #     freqs = Dataset.class_freqs

    #     if class_balance == 'median_freq_cost':
    #         w_freq = np.median(freqs) / freqs
    #     elif class_balance == 'rare_freq_cost':
    #         w_freq = 1 / (cfg.nclasses * freqs)

    #     print("Class balance weights", w_freq)
    #     cfg.class_balance = w_freq

    # ============ Train/validation
    # Load data
    # init_epoch = 0
    # prev_history = None
    # best_loss = np.Inf
    # best_val = np.Inf if early_stop_strategy == 'min' else -np.Inf
    # val_metrics_ext = ['val_' + m for m in val_metrics]
    # history_path = tmp_path + save_name + '.npy'
    # if cfg.reload_weights:
    #     # Reload weights
    #     pass

    # BUILD GRAPH
    # TODO consider CPU case as well
    config = tf.ConfigProto(allow_soft_placement=True,
                            device_count={'GPU': cfg.num_gpus})
    sess = tf.Session(config=config)

    if cfg.debug:
        from tensorflow.python import debug as tf_debug
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    print("Building the model ...")
    cfg.global_step = tf.Variable(0, trainable=False, name='global_step',
                                  dtype=cfg._FLOATX)
    sym_inputs = tf.placeholder(shape=cfg.input_shape,
                                dtype=cfg._FLOATX, name='inputs')
    sym_val_inputs = tf.placeholder(shape=cfg.val_input_shape,
                                    dtype=cfg._FLOATX, name='val_inputs')
    sym_labels = tf.placeholder(shape=[None], dtype='int32', name='labels')

    # TODO is there another way to split the input in chunks when
    # batchsize is not a multiple of num_gpus?
    # Split in chunks, the size of each is provided in sym_input_split_dim
    sym_inputs_split_dim = tf.placeholder(shape=[cfg.num_gpus],
                                          dtype='int32',
                                          name='inputs_split_dim')
    sym_labels_split_dim = tf.placeholder(shape=[cfg.num_gpus],
                                          dtype='int32',
                                          name='label_split_dim')
    placeholders = [sym_inputs, sym_labels, sym_inputs_split_dim,
                    sym_labels_split_dim]
    val_placeholders = [sym_val_inputs, sym_labels, sym_inputs_split_dim,
                        sym_labels_split_dim]

    with sess, tf.device(cfg.devices[0]):
        # Model compilation
        # -----------------
        train_outs, train_summary_op = build_graph(placeholders,
                                                   cfg.input_shape,
                                                   cfg.optimizer,
                                                   cfg.weight_decay,
                                                   cfg.loss_fn, build_model,
                                                   True)

        val_outs, val_summary_ops = build_graph(val_placeholders,
                                                cfg.val_input_shape,
                                                cfg.optimizer,
                                                cfg.weight_decay,
                                                cfg.loss_fn, build_model,
                                                False)

        # Add the variables initializer Op.
        init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())

        # Initialize the variables (we might restore a subset of them..)
        sess.run(init)

        # for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        #     print(v)

        if cfg.restore_model:
            print("Restoring model from checkpoint ...")
            if cfg.checkpoints_file is None:  # default: last saved checkpoint
                checkpoint = tf.train.latest_checkpoint(cfg.checkpoints_dir)
                print(checkpoint)
            else:
                checkpoint = os.path.join(cfg.checkpoints_dir,
                                          cfg.checkpoints_file)
                print(checkpoint)
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint)
            print("Model restored.")
        # elif cfg.pretrained_vgg:
        #     print("Loading VGG16 weights ...")
        #     load_vgg_weights(file=cfg.vgg_weights_file,
        #                      vgg_var_to_load=cfg.vgg_var_to_load,
        #                      sess=sess)
        #     print("VGG16 pretrained weights loaded.")

        if not cfg.do_validation_only:
            # Start training loop
            main_loop_kwags = {'placeholders': placeholders,
                               'val_placeholders': val_placeholders,
                               'train_outs': train_outs,
                               'train_summary_op': train_summary_op,
                               'val_outs': val_outs,
                               'val_summary_ops': val_summary_ops,
                               'loss_fn': cfg.loss_fn,
                               'Dataset': cfg.Dataset,
                               'dataset_params': cfg.dataset_params,
                               'valid_params': cfg.valid_params,
                               'sess': sess}
            return main_loop(**main_loop_kwags)
        else:
            # Perform validation only
            mean_iou = []
            for s in cfg.val_on_sets:
                print('Starting validation on %s set' % s)
                from validate import validate
                mean_iou[s] = validate(
                    val_placeholders,
                    val_outs,
                    val_summary_ops[s],
                    sess,
                    0,
                    which_set=s)


def build_graph(placeholders, input_shape, optimizer, weight_decay, loss_fn,
                build_model, is_training):
    cfg = gflags.cfg
    num_gpus = cfg.num_gpus
    devices = cfg.devices
    nclasses = cfg.nclasses
    global_step = cfg.global_step
    [sym_inputs, sym_labels, sym_input_split_dim,
     sym_labels_split_dim] = placeholders

    # Split the input among the GPUs (batchwise)
    sym_inputs_per_gpu = tf.split(sym_inputs, sym_input_split_dim, 0)
    sym_labels_per_gpu = tf.split(sym_labels, sym_labels_split_dim, 0)
    for dev_idx in range(num_gpus):
        sym_inputs_per_gpu[dev_idx].set_shape(input_shape)

    # Init variables
    tower_grads = []
    tower_preds = []
    tower_soft_preds = []
    tower_losses = []
    summaries = {}
    if is_training:
        summaries['training'] = tf.get_collection_ref(key='train_summaries')
    else:
        for k in cfg.val_on_sets:
            summaries[k] = tf.get_collection_ref(key='val_' + k + '_summaries')

    for device_idx, (inputs, labels) in enumerate(zip(sym_inputs_per_gpu,
                                                      sym_labels_per_gpu)):
        with tf.device(devices[device_idx]):
            reuse_variables = not is_training or device_idx > 0
            tower_suffix = 'train' if is_training else 'val'
            with tf.name_scope('towr{}_{}'.format(device_idx,
                                                  tower_suffix)) as scope:
                with tf.variable_scope(cfg.model_name, reuse=reuse_variables):

                    net_out = build_model(inputs, is_training)
                    softmax_pred = slim.softmax(net_out)
                    tower_soft_preds.append(softmax_pred)

                    # Prediction
                    sym_pred = tf.argmax(softmax_pred, axis=-1)
                    tower_preds.append(sym_pred)

                    # Loss
                    # Use softmax, unless using the
                    # tf.nn.sparse_softmax_cross_entropy function that
                    # internally applies it already
                    if (loss_fn is not
                            tf.nn.sparse_softmax_cross_entropy_with_logits):
                        net_out = softmax_pred
                    loss = apply_loss(labels, net_out, loss_fn,
                                      weight_decay, is_training,
                                      return_mean_loss=True)
                    tower_losses.append(loss)
                    # Save this GPU's summary (thanks to scope)
                    with tf.name_scope('tower_loss_summaries'):
                        for k, s in summaries.iteritems():
                            s.append(tf.summary.scalar(
                                'Loss_tower_{}_{}'.format(scope, k), loss))

                    # Gradients
                    if is_training:
                        grads = optimizer.compute_gradients(loss)
                        tower_grads.append(grads)

                    # Print regularization
                    for v in tf.get_collection(
                            tf.GraphKeys.REGULARIZATION_LOSSES):
                        print('Regularization losses:\n{}'.format(v))

    # Convert from list of tensors to tensor, and average
    sym_preds = tf.concat(tower_preds, axis=0)
    softmax_preds = tf.concat(tower_soft_preds, axis=0)

    # Compute the mean IoU
    # TODO would it be better to use less precision here?
    sym_mask = tf.cast(tf.less_equal(sym_labels, nclasses), tf.int32)
    sym_preds_flat = tf.reshape(sym_preds, [-1])
    sym_m_iou, sym_cm_update_op = tf.metrics.mean_iou(sym_labels,
                                                      sym_preds_flat,
                                                      nclasses,
                                                      sym_mask)
    # Compute the average *per variable* across the towers
    sym_avg_tower_loss = tf.reduce_mean(tower_losses)

    if is_training:
        # Impose graph dependency so that update operations are computed
        # even if they're are not explicit in the outputs os session.run
        grads_and_vars = average_gradients(tower_grads)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars,
                                                 global_step=global_step)
        # Add the histograms of the gradients
        with tf.name_scope('grad_summaries'):
            for grad, var in grads_and_vars:
                if grad is not None:
                    summaries['training'].append(
                        tf.summary.histogram(var.op.name + '/gradients', grad))

    #############
    # SUMMARIES #
    #############
    with tf.name_scope('summaries'):
        for k, s in summaries.iteritems():
            s.append(tf.summary.scalar('Mean_tower_loss_' + k,
                                       sym_avg_tower_loss))
            s.append(tf.summary.scalar('Mean_IoU_' + k, sym_m_iou))

        if is_training:
            # Add the histograms for trainable variables
            for var in tf.trainable_variables():
                summaries['training'].append(tf.summary.histogram(var.op.name,
                                                                  var))
            train_summary_op = tf.summary.merge(summaries['training'])
        else:
            val_summary_ops = {}
            for k, s in summaries.iteritems():
                val_summary_ops[k] = tf.summary.merge(s)

    if is_training:
        return [sym_avg_tower_loss, train_op], train_summary_op
    else:
        return ([sym_preds, softmax_preds, sym_m_iou, sym_avg_tower_loss,
                 sym_cm_update_op], val_summary_ops)


def main_loop(placeholders, val_placeholders, train_outs, train_summary_op,
              val_outs, val_summary_ops, loss_fn, Dataset, dataset_params,
              valid_params, sess):

    cfg = gflags.cfg
    max_epochs = cfg.max_epochs

    # Prepare the summary objects
    summary_writer = tf.summary.FileWriter(logdir=cfg.train_checkpoints_dir,
                                           graph=sess.graph)
    saver = tf.train.Saver(max_to_keep=cfg.checkpoints_to_keep)

    # TRAIN
    dataset_params['batch_size'] *= cfg.num_gpus
    print('\nTrain dataset params:\n{}\n'.format(dataset_params))
    print('Validation dataset params:\n{}\n\n'.format(valid_params))
    train = Dataset(
        which_set='train',
        return_list=False,
        **dataset_params)

    # Setup loop parameters
    init_step = 0  # TODO do we need this? Can we get it out of the checkpoints
    val_skip = cfg.val_skip
    patience_counter = 0
    cum_iter = 0
    estop = False
    end_of_epoch = False
    last_epoch = False
    history_acc = np.array([]).tolist()

    # Start the training loop.
    start = time()
    print("Beginning main loop...")
    for epoch_id in range(init_step, max_epochs):
        pbar = tqdm(total=train.nbatches)
        epoch_start = time()

        for batch_id in range(train.nbatches):
            cum_iter += 1
            iter_start = time()

            # inputs and labels
            minibatch = train.next()
            t_data_load = time() - iter_start
            x_batch, y_batch = minibatch['data'], minibatch['labels']
            # sh = inputs.shape  # do NOT provide a list of shapes
            x_in = x_batch
            y_in = y_batch.flatten()
            # if cfg.use_second_path:
            #    x_in = [x_batch[..., :3], x_in[..., 3:]]
            # reset_states(model, sh)

            # TODO evaluate if it's possible to pass num_gpus inputs in
            # a list, rather than the input as a whole and the shape of
            # the splits as a tensor.

            split_dim, labels_split_dim = compute_chunk_size(
                x_batch.shape[0], np.prod(train.data_shape[:2]))

            # Create dictionary to feed the input placeholders
            # placeholders = [sym_inputs, sym_labels, sym_which_set,
            #                 sym_input_split_dim, sym_labels_split_dim]
            in_values = [x_in, y_in, split_dim, labels_split_dim]
            feed_dict = {p: v for (p, v) in zip(placeholders, in_values)}

            # train_op does not return anything, but must be in the
            # outputs to update the gradient
            loss_value, _ = sess.run(train_outs, feed_dict=feed_dict)
            t_iter = time() - iter_start

            # Upgrade the summaries
            summary_str = sess.run(train_summary_op, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, epoch_id)
            summary_writer.flush()
            # Save the checkpoint
            checkpoint_path = os.path.join(cfg.checkpoints_dir,
                                           cfg.checkpoints_file)
            saver.save(sess, checkpoint_path, global_step=cfg.global_step)

            pbar.set_description('Batch {}/{}({}) {}s (D {}s), Loss {}'.format(
                batch_id + 1, train.nbatches, cum_iter, t_iter, t_data_load,
                loss_value))
            pbar.update(1)

            # Verify if it's the end of the epoch
            if batch_id == train.nbatches - 1:
                end_of_epoch = True
                # valid_wait = 0 if valid_wait == 1 else valid_wait - 1
                t_epoch = time() - epoch_start

                # Is it also the last epoch?
                if epoch_id == max_epochs - 1:
                    last_epoch = True

                # Early stop if patience is over
                patience_counter += 1
                if (epoch_id >= cfg.min_epochs and
                        patience_counter >= cfg.patience):
                    estop = True

                pbar.clear()
                # TODO replace with logger
                print('Epoch time: {}s, Epoch {}/{}, Loss: {}'.format(
                    t_epoch, epoch_id + 1, max_epochs, loss_value))

            # TODO use tf.contrib.learn.monitors.ValidationMonitor?
            # Validate if last iteration, early stop or we reached valid_every
            if last_epoch or estop or (end_of_epoch and not val_skip):
                end_of_epoch = False
                # Validate
                mean_iou = {}
                from validate import validate
                for s in cfg.val_on_sets:
                    print('\nStarting validation on %s set' % s)
                    mean_iou[s] = validate(
                        val_placeholders,
                        val_outs,
                        val_summary_ops[s],
                        sess,
                        epoch_id,
                        which_set=s,
                        model_name='')

                # TODO gsheet
                history_acc.append([mean_iou.get('valid')])

                # Did we improve *validation* mean IOU accuracy?
                best_hist = np.array(history_acc).max()
                if (len(history_acc) == 0 or
                   mean_iou.get('valid') >= best_hist):
                    print('## Best model found! ##')
                    print('Saving the checkpoint ...')
                    checkpoint_path = os.path.join(cfg.checkpoints_dir,
                                                   cfg.checkpoints_file)
                    saver.save(sess, checkpoint_path,
                               global_step=cfg.global_step)

                    patience_counter = 0
                    estop = False
                # Start skipping again
                val_skip = max(1, cfg.val_every_epochs)

                # exit minibatches loop
                if estop:
                    print('Early Stop!')
                    break
                if last_epoch:
                    print('Last epoch!')
                    break

            elif end_of_epoch:
                end_of_epoch = False
                # We skipped validation, decrease the counter
                val_skip -= 1

        # exit epochs loop
        if estop or last_epoch:
            break

    max_valid_idx = np.argmax(np.array(history_acc))
    best = history_acc[max_valid_idx]
    (valid_mean_iou) = best

    print("")
    print('Best: Mean Class iou - Valid {:.5f}'.format(valid_mean_iou))
    print("")

    end = time()
    m, s = divmod(end - start, 60)
    h, m = divmod(m, 60)
    print("Total time elapsed: %d:%02d:%02d" % (h, m, s))

    # # Move complete models and stuff to shared fs
    # print('\n\nEND OF TRAINING!!\n\n')

    # def move_if_exist(filename, dest):
    #     if not os.path.exists(os.path.dirname(dest)):
    #         os.makedirs(os.path.dirname(dest))
    #     try:
    #         shutil.move(filename, dest)
    #     except IOError:
    #         print('Move error: {} does not exist.'.format(filename))

    # move_if_exist(tmp_path + save_name + "_best.w",
    #               'models/' + save_name + '_best.w')
    # move_if_exist(tmp_path + save_name + "_best_loss.w",
    #               'models/' + save_name + '_best_loss.w')
    # move_if_exist(tmp_path + save_name + "_latest.w",
    #               'models/' + save_name + '_latest.w')
    # move_if_exist(tmp_path + save_name + ".npy",
    #               'models/' + save_name + '.npy')
    # move_if_exist(tmp_path + save_name + ".svg",
    #               'models/' + save_name + '.svg')
    # validate = True  # print the best model's test error
    return best


# PUT THIS INTO YOUR MAIN
if __name__ == '__main__':
    from model import build_model
    run(sys.argv, build_model)
