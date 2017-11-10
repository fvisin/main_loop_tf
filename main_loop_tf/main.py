from copy import deepcopy
import abc
import hashlib
try:
    from itertools import izip_longest as zip_longest
except:
    from itertools import zip_longest
import logging
import os
import sys
from time import time

import dataset_loaders
import numpy as np
import tensorflow as tf
from tensorflow.python.training.supervisor import Supervisor
from tqdm import tqdm

import gflags
from optimization import get_optimizer
from utils import save_repos_hash, split_in_chunks, squash_maybe, TqdmHandler

# config module load all flags from source files
import config  # noqa

import cv2
try:
    import pygtk  # noqa
    import gtk
    gtk.gdk.threads_init()
except:
    import warnings
    warnings.warn('pygtk is not installed, it will not be possible to '
                  'debug the optical flow')
    pygtk = None
    gtk = None

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
gflags.DEFINE_string('supervisor_master', '', 'The "master" string for the '
                     'Supervisor')


class Experiment(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def build_loss(self, dev_labels, model_outs, inputs):
        pass

    @abc.abstractmethod
    def build_model(self, dev_inputs, is_training):
        pass

    # def validate_fn(self, input_placeholders, graph_outs, which_set):
    #     return dict

    # def metrics_graph_fn(self)
    #     return metrics_outs, metrics_ops

    def __init__(self, flags_argv, Optimizer=None):
        """Create an Experiment object

        Parameters
        ----------
        flags_argv: list
            A list of flags argument for gflags
        Optimizer: :class:`DistributedOptimizer`
            Optional. An optimizer object to be used in the optimization
            phase.
        """
        gflags.mark_flags_as_required(['dataset'])
        self.UserOptimizer = Optimizer

        # ============ Parse gflags
        try:
            FLAGS(flags_argv)  # parse flags
        except gflags.FlagsError as e:
            print('Usage: %s ARGS\n%s\n\nError: %s' % (flags_argv[0], FLAGS,
                                                       e))
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

        # ============ Hash, (gsheet) and checkpoints
        # Exclude non JSONable and not interesting objects
        exclude_list = ['checkpoints_dir', 'checkpoints_to_keep', 'dataset',
                        'debug', 'debug_of', 'devices', 'do_validation_only',
                        'group_summaries', 'help', 'hyperparams_summaries',
                        'max_epochs', 'min_epochs', 'model_name', 'nthreads',
                        'patience', 'restore_model',
                        'save_gif_frames_on_disk', 'save_gif_on_disk',
                        'save_raw_predictions_on_disk',
                        'show_heatmaps_summaries', 'show_samples_summaries',
                        'supervisor_master', 'thresh_loss',
                        'train_summary_freq', 'use_threads',
                        'val_every_epochs', 'val_on_sets', 'val_skip_first',
                        'val_summary_freq', 'summary_per_subset']
        param_dict = {k: deepcopy(v) for (k, v) in cfg.__dict__.iteritems()
                      if k not in exclude_list}
        h = hashlib.md5()
        h.update(str(param_dict))
        cfg.hash = h.hexdigest()
        save_repos_hash(param_dict, cfg.model_name, ['tensorflow',
                                                     'dataset_loaders',
                                                     'main_loop_tf'])

        if cfg.restore_model is None or cfg.restore_model == 'False':
            # If the model should not be restored from a checkpoint,
            # change the checkpoints directory by adding an incremental
            # suffix
            cfg.checkpoints_dir = os.path.join(cfg.checkpoints_dir,
                                               cfg.model_name, cfg.hash)
            incr_num = 0
            logdir = cfg.checkpoints_dir
            while(os.path.exists(logdir)):
                incr_num += 1
                if incr_num == 1:
                    logdir += '_' + str(incr_num)
                else:
                    logdir = cfg.checkpoints_dir + '_' + str(incr_num)
            cfg.checkpoints_dir = logdir
            del(logdir)
        else:
            restore_checkpoints_dir = os.path.join(cfg.checkpoints_dir,
                                                   cfg.model_name,
                                                   cfg.restore_model)
            # If you want to reload a specific  hash
            if os.path.exists(restore_checkpoints_dir):
                cfg.checkpoints_dir = restore_checkpoints_dir
            else:  # If you just want to reload the default hash
                cfg.checkpoints_dir = os.path.join(
                    cfg.checkpoints_dir, cfg.model_name, cfg.hash)

        cfg.train_checkpoints_dir = os.path.join(cfg.checkpoints_dir, 'train')
        cfg.val_checkpoints_dir = os.path.join(cfg.checkpoints_dir, 'valid')

        # ============ A bunch of derived params
        cfg._FLOATX = 'float32'
        cfg.num_gpus = len([el for el in cfg.devices if 'gpu' in el])
        cfg.num_cpus = len([el for el in cfg.devices if 'cpu' in el])
        cfg.num_splits = cfg.num_gpus + cfg.num_cpus

        # ============ Dataset init
        try:
            Dataset = getattr(dataset_loaders, cfg.dataset + 'Dataset')
        except AttributeError:
            Dataset = getattr(dataset_loaders, cfg.dataset.capitalize() +
                              'Dataset')

        self.Dataset = Dataset
        # Add dataset extra parameters specific for the dataset
        dataset_params = cfg.train_extra_params
        dataset_params['batch_size'] = cfg.batch_size * cfg.num_splits
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

            ret_ext_seq = cfg.return_extended_sequences
            ret_middle_frame = cfg.return_middle_frame_only
            dataset_params['return_extended_sequences'] = ret_ext_seq
            dataset_params['return_middle_frame_only'] = ret_middle_frame

        dataset_params['use_threads'] = cfg.use_threads
        dataset_params['nthreads'] = cfg.nthreads
        dataset_params['remove_per_img_mean'] = cfg.remove_per_img_mean
        dataset_params['divide_by_per_img_std'] = cfg.divide_by_per_img_std
        dataset_params['remove_mean'] = cfg.remove_mean
        dataset_params['divide_by_std'] = cfg.divide_by_std
        cfg.dataset_params = dataset_params
        cfg.valid_params = deepcopy(cfg.dataset_params)
        cfg.valid_params.update({
            'batch_size': cfg.val_batch_size * cfg.num_splits,
            'seq_per_subset': 0,
            'overlap': cfg.val_overlap,
            'shuffle_at_each_epoch': (cfg.val_overlap is not None and
                                      cfg.val_overlap != 0),
            'return_middle_frame_only': False,
            'one_subset_per_batch': True,  # prevent multiple subsets
            'use_threads': False,  # prevent shuffling
            # prevent crop
            'data_augm_kwargs': {'return_optical_flow': cfg.of}})
        # Add dataset extra parameters specific for each dataset
        cfg.valid_params.update(cfg.val_extra_params)

        # Create temporary dataset object (training/validation) to get
        # dynamic class elements (e.g. data_shape)
        train_temp = Dataset(
            which_set='train',
            return_list=False,
            **cfg.dataset_params)
        valid_temp = Dataset(
            which_set='valid',
            **cfg.valid_params)

        # TODO: check fvisin comment, this is not the correct behavior, but
        # it's done in order to work with movingMNST iirc
        if cfg.seq_length:
            cfg.input_shape = [None, cfg.seq_length] + list(
                train_temp.next()['data'].shape[2:])
            cfg.val_input_shape = [None, cfg.seq_length] + list(
                valid_temp.next()['data'].shape[2:])

            if cfg.of:
                cfg.input_shape[-1] = 6
                cfg.val_input_shape[-1] = 6

            if cfg.crop_size:
                cfg.input_shape[2:4] = cfg.crop_size
        else:
            cfg.input_shape = [None] + list(
                train_temp.next()['data'].shape[1:])
            cfg.val_input_shape = [None] + list(
                valid_temp.next()['data'].shape[1:])
            if cfg.crop_size:
                cfg.input_shape[1:3] = cfg.crop_size

        cfg.void_labels = getattr(Dataset, 'void_labels', [])
        cfg.nclasses = Dataset.non_void_nclasses
        cfg.nclasses_w_void = Dataset.nclasses
        tf.logging.info('{} classes ({} non-void):'.format(cfg.nclasses_w_void,
                                                           cfg.nclasses))
        # Destroy temporary dataset objects
        train_temp.finish()
        valid_temp.finish()

        self.cfg = cfg

        self.val_skip = (cfg.val_skip_first if cfg.val_skip_first else
                         max(1, cfg.val_every_epochs) - 1)

        # Init variables
        self.val_graph_outs = {}

        self.avg_loss = {}

        # Build the graph
        self.__build_graph__()

    def __build_graph__(self):
        cfg = self.cfg

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

        tf.logging.info("Building the model ...")
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Create a list of input placeholders for each device.
            # When the batchsize is not big enough to fill all of them we
            # would want to use a subset of the placeholders, but TF raises
            # a 'negative shape error' if a placeholder is not fed. Instead,
            # we provide all of them with values but we use n_spits to
            # select which of the inputs to process (and perform gradient
            # descent on) and which to ignore.
            # At runtime, we replicate the input data to feed all the
            # placeholders (even if it's internally ignored). We could use
            # placeholder_with_default to assign a value to it's input but
            # the batch_size might change dynamically, so we rather
            # replicate the input at runtime.
            train_inputs_per_gpu = []
            val_inputs_per_gpu = []
            labels_per_gpu = []
            # TODO add function to get the placeholders
            self.num_splits = tf.placeholder(np.int32, shape=None,
                                             name='num_splits')
            self.num_batches = tf.placeholder(np.int32, shape=None,
                                              name='num_batches')
            self.prev_err = tf.placeholder(shape=(), dtype=cfg._FLOATX,
                                           name='prev_err')
            for i, _ in enumerate(range(cfg.num_splits)):
                train_inputs_per_gpu.append(tf.placeholder(
                    dtype=cfg._FLOATX,
                    shape=cfg.input_shape,
                    name='inputs_per_gpu_%i' % i))
                val_inputs_per_gpu.append(tf.placeholder(
                    dtype=cfg._FLOATX,
                    shape=cfg.val_input_shape,
                    name='val_inputs_per_gpu_%i' % i))
                labels_per_gpu.append(tf.placeholder(
                    dtype=np.int32,
                    shape=[None],  # flattened
                    name='labels_per_gpu_%i' % i))
            self.train_inputs_per_gpu = train_inputs_per_gpu
            self.val_inputs_per_gpu = val_inputs_per_gpu
            self.labels_per_gpu = labels_per_gpu

            # Model compilation
            # -----------------
            # Model parameters on the FIRST device specified in cfg.devices
            # Gradient Average and the rest of the operations are on CPU
            with tf.device('/cpu:0'):
                # Build the training graph
                self.train_graph_outs = self.__build_device_graph(
                    which_set='train', is_training=True)

                # Build the validation graphs (reusing variables)
                for s in cfg.val_on_sets:
                    self.val_graph_outs[s] = self.__build_device_graph(
                        which_set=s, is_training=False)

                # Create the hyperparameters summaries operations
                if cfg.hyperparams_summaries is not None:
                    summary_text = []
                    for (key_header,
                         list_value) in cfg.hyperparams_summaries.iteritems():

                        header_list = []
                        text_list = []
                        for v in list_value:
                            header_list.append('**'+v+'**')
                            text_list.append(str(getattr(cfg, v)))
                        header_tensor = tf.constant(header_list)
                        text_tensor = tf.constant(text_list)

                        summary_text.append(tf.summary.text(
                            key_header,
                            tf.reshape(tf.concat([header_tensor, text_tensor],
                                                 axis=0), [2, -1])))
                    self.summary_text_op = tf.summary.merge(summary_text)

    def __build_device_graph(self, which_set, is_training):
        ''' Build the multiGPU graph of computation

        This function creates a copy of the computation graph on each GPU. The
        result of the computation of each GPU is stored in a "tower"
        Note that thanks to the use of name_scopes and variable_scopes,
        calling this function multiple times does not create multiple
        copies of the *Ops* and of the *Variables* (respectively), but
        rather only adds the Ops that change from one call to the other
        and reuses the same Variables.

        Furthermore, we accommodate for the case where some minibatches
        are smaller than the usual size and are not enough to feed all
        the devices. Since we cannot change the graph at runtime, we
        accomplish this by feeding the unused devices and discarding
        their output. To prevent the statistics of these unnecessary
        computation to be retrieved and visualized, we create several
        summary ops, to collect the summaries of the first device, of
        the first two devices, of the first three, .., and so on. This
        allows to choose at runtime which summary operations to call,
        depending on the batch size.  batch size.
        '''
        cfg = self.cfg
        reuse_variables = not is_training

        labels_per_gpu = self.labels_per_gpu
        grad_ops = []
        if is_training:
            inputs_per_gpu = self.train_inputs_per_gpu
            summaries_str = 'train_summaries_%s'
        else:
            inputs_per_gpu = self.val_inputs_per_gpu
            summaries_str = 'val_%s_summaries' % which_set + '_%s'

        self.global_step = tf.Variable(0, trainable=False, name='global_step',
                                       dtype='int32')
        # Set Optimizer
        if self.UserOptimizer is None:
            optimizer = get_optimizer(cfg.optimizer)(
                cfg=cfg, global_step=self.global_step)
        else:
            optimizer = self.UserOptimizer(
                cfg=cfg, global_step=self.global_step)

        # Create "towers" with the model outputs/loss keys and a value
        # for each device
        devs_model_outs = {}
        summaries = []
        for device in cfg.devices:
            device_str = device.replace('/', '').replace(':', '').lower()
            dev_set_str = '{}_{}'.format(device_str, which_set)
            summaries.append(summaries_str % device_str)
        these_s = summaries

        # Build a graph for each device, each with its input and output
        # placeholders. Collect the outputs in "towers"
        # -------------------------------------------------------------
        for device, dev_inputs, dev_labels in zip(cfg.devices,
                                                  inputs_per_gpu,
                                                  labels_per_gpu):
            with tf.name_scope('{}_{}'.format(device_str, which_set)), \
                    tf.variable_scope(cfg.model_name, reuse=reuse_variables), \
                    tf.device(device):
                reuse_variables = True

                # Model preactivation, activation (softmax) and prediction
                model_out = self.build_model(dev_inputs, dev_labels,
                                             is_training)
                assert isinstance(model_out, dict), """
                    Your model should return a dictionary"""
                assert 'out_preact' in model_out, """Your model
                    function should return a dictionary with attribute
                    'out_preact'!"""
                assert 'out_act' in model_out, """Your model function
                    should return a dictionary with attribute 'out_act'!"""
                assert 'pred' in model_out, """Your model function should
                    return a dictionary with at least attribute 'pred'!"""

                # Accumulate the loss outputs from each device into a
                # dictionary with the same keys and a list of values,
                # one for each device
                for k, v in model_out.iteritems():
                    devs_model_outs.setdefault(k, []).append(v)

                # Loss
                # TODO: create **loss_params to be specified externally
                # when specializing Experiment
                loss_outs = self.build_loss(dev_labels, model_out,
                                            is_training=is_training,
                                            # l2_reg=weight_decay,
                                            # gdl=cfg.gdl,
                                            # tv=cfg.tv,
                                            inputs=dev_inputs)
                assert loss_outs is not None and isinstance(loss_outs, dict), (
                    """Your loss should return a dictionary""")
                assert 'loss' in loss_outs, """Your loss function should
                    return a dictionary with attribute 'loss'!"""
                assert 'components' in loss_outs, """Your loss function should
                    return a dictionary with attribute 'components'
                    containing the list of terms that composes the total
                    loss!"""

                # Remove the name_scopes (the one from the variable_scope and
                # the one from the name_scope) and assign dev_set_str
                # TODO:
                # Save a summary with the loss per device
                scope_str = dev_set_str + '_stats'
                with tf.name_scope(None):
                    with tf.name_scope(scope_str) as dev_set_scope:
                        tf.summary.scalar('Loss', loss_outs['loss'], these_s)

                # Compute loss, gradients, add noise to the gradient and
                # create the op to apply it if needed.
                # Note that has to be called in validation as well to
                # compute the loss.
                # Create a *list* of gradient update ops. The t-th element
                # of the list updates the gradients of the devices *up to*
                # the t-th device
                grad_op = optimizer.distributed_minimize(
                    loss_outs=loss_outs,
                    is_training=is_training,
                    device=device,
                    dev_set_scope=dev_set_scope,
                    summaries=these_s,
                    colocate_gradients_with_ops=True)
                if is_training:  # dev_train_op will be None otherwise
                    grad_ops.append(grad_op)

            # Print regularization
            for v in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES):
                tf.logging.debug('Regularization losses:\n{}'.format(v))

            # Update the summaries that will be affected by the next graph.
            # We want summaries[0] to contain summaries relative to the
            # first device only, summaries[1] to the first and
            # second device and so on. This will be used in the main
            # loop to suppress some of the summaries when some of the
            # devices are not being used.
            #
            # Summary device0 device1 device2 ...
            #   0        X
            #   1        X       X
            #   1        X       X       X
            #  ...
            these_s = these_s[1:]

        # Merge the towers on CPU
        # -----------------------
        merged_model_outs = {}
        for k, v in devs_model_outs.iteritems():
            # Convert from list of tensors to concatenated tensor
            merged_model_outs[k] = tf.concat(v, axis=0, name='concat_%s' % k)
            merged_model_outs[k] = merged_model_outs[k][:self.num_batches]

        tf.summary.scalar('control_flow/batch_size_' + which_set,
                          tf.shape(merged_model_outs['pred'])[0], summaries)

        # Concatenate the per-gpu placeholders to get a placeholder for the
        # full list of gpus and one for the subset to be used for
        # the minibatch with less batches
        labels = tf.concat(self.labels_per_gpu, axis=0, name='concat_labels')
        # Remove the unused batches from the flattened labels
        # (equivalent to labels[:np.prod(merged_model_outs.shape)])
        labels = labels[:tf.shape(
            tf.reshape(merged_model_outs['pred'], [-1]))[0]]

        #############
        # SUMMARIES #
        #############
        # Visualize the avg loss
        # The number of devices will be dynamically selected by the
        # numerical value assigned to num_splits at run-time) and used
        # to update the loss summaries correctly
        self.avg_loss[is_training] = optimizer.get_avg_loss(self.num_splits)
        tf.summary.scalar(dev_set_scope + '_Mean_tower_loss/Total_Loss',
                          self.avg_loss[is_training], summaries)
        if is_training:
            # Add the per-component loss summaries
            avg_comp_loss = optimizer.get_avg_comp_loss(self.num_splits)
            for k, v in avg_comp_loss.iteritems():
                tf.summary.scalar(dev_set_scope + '_Mean_tower_loss/%s' % k, v,
                                  summaries)

            # Add the histograms for trainable variables
            for var in tf.trainable_variables():
                var_name = var.op.name
                scope_str, var_name = squash_maybe(dev_set_str, var_name)
                scope_str += '_%s_%s'  # metric, var
                scope_str = dev_set_str + '_%s_'  # metric
                scope_str, var_name = squash_maybe(scope_str, var_name)
                scope_str += '_%s'  # var name
                tf.summary.histogram(scope_str % ('Trainable_vars_activations',
                                                  var_name),
                                     var, summaries)

        # Create a list of summary ops that update the summary collections that
        # we used at graph creation time. Thanks to the way we decremented the
        # elements in the collections each time the graph for one device
        # was created, the n-th op in this list will update all the summaries
        # *up to* the n-th device. This will be used at run-time to ignore the
        # devices that are not in use when there are not enough batches to feed
        # all of them
        summary_ops = []
        for s in summaries:
            summary_ops.append(tf.summary.merge(tf.get_collection_ref(key=s)))

        graph_out = {
            'model_outs': merged_model_outs,
            'summary_ops': summary_ops,
            }
        if is_training:
            graph_out['grad_ops'] = grad_ops

        # User defined function to compute some metrics in the graph
        if hasattr(self, 'metrics_graph_fn'):
            metrics_outs, metrics_ops = self.metrics_graph_fn()
            graph_out['metrics_outs'] = metrics_outs
            graph_out['metrics_ops'] = metrics_ops

        return graph_out

    def run(self):
        with self.__init_sess__() as self.sess:
            if self.cfg.hyperparams_summaries is not None:
                # write Hyper parameters text summaries
                summary_str = self.sess.run(self.summary_text_op)
                self.sv.summary_computed(self.sess, summary_str)

            # Start training loop
            return self.__main_loop()

    def evaluate(self):
        with self.__init_sess__() as self.sess:
            validate_fn = getattr(self, "validate_fn", None)
            if validate_fn is not None:
                metrics_val = {}
                for s in self.cfg.val_on_sets:
                    metrics_val[s] = validate_fn(
                        self.val_inputs_per_gpu,
                        self.val_graph_outs[s],
                        which_set=s)
                return metrics_val
            else:
                raise ValueError('No validation function defined! You '
                                 'should implement __validate_fn')

    def __init_sess__(self):
        cfg = self.cfg
        with self.graph.as_default():
            # Group global and local init into one op. Could be split into
            # two different ops and passed to `init_op` and `local_init_op`
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())

            self.saver = tf.train.Saver(
                max_to_keep=cfg.checkpoints_to_keep)

            sv = Supervisor(
                graph=self.graph,
                init_op=init_op,
                summary_op=None,
                global_step=self.global_step,
                # TODO add option to restore best rather than last?
                logdir=cfg.checkpoints_dir,
                checkpoint_basename=cfg.model_name,
                saver=self.saver,
                # session_manager
                # summary_writer
                save_model_secs=300)
            self.sv = sv

            tf_config = tf.ConfigProto(allow_soft_placement=True)
            sess_gen = sv.managed_session(cfg.supervisor_master, tf_config)
            if self.cfg.debug:
                from tensorflow.python import debug as tf_debug
                sess_gen = tf_debug.LocalCLIDebugWrapperSession(sess_gen)
                sess_gen.add_tensor_filter("has_inf_or_nan",
                                           tf_debug.has_inf_or_nan)
            return sess_gen

    def __main_loop(self):
        cfg = gflags.cfg

        self.experiment_begin()

        while not self.sv.should_stop():
            self.epoch_id = (self.cum_iter+1) // self.train.nbatches

            # Callback
            self.epoch_begin()

            for batch_id in range(self.train.nbatches):
                self.cum_iter = self.sv.global_step.eval(self.sess)
                iter_start = time()

                # inputs and labels
                minibatch = self.train.next()
                self.t_data_load = time() - iter_start
                x_batch, y_batch = minibatch['data'], minibatch['labels']
                # sh = inputs.shape  # do NOT provide a list of shapes

                # Callback
                self.batch_begin(minibatch)

                # Is this batch shorter than batch_size?
                # Check if this batch will not be processed by all the devices.
                # When the sequence is shorter than seq_length or the number of
                # batches is smaller than batch_size, the batch will be smaller
                # than usual. When this happens we might not be able to feed
                # all the CPUs/GPUs altogether. In that case here we compute
                # the number of GPUs that we can use for the current batch
                # Spread the batch over the lowest number of GPUs
                this_n_splits = len(x_batch) // cfg.batch_size
                if len(x_batch) % cfg.batch_size != 0:
                    this_n_splits += 1
                num_devs = this_n_splits - 1

                # Get the per-device inputs
                # TODO inputs should be a list (as well as placeholders)
                x_batch_chunks, y_batch_chunks = split_in_chunks(x_batch,
                                                                 y_batch,
                                                                 this_n_splits)

                # Fill the placeholders with data up to this_n_splits, and
                # then repeat one of the chunks. Note that this will be
                # ignored later on (see comment where placeholders are created)
                in_vals = list(zip_longest(self.train_inputs_per_gpu,
                                           x_batch_chunks,
                                           fillvalue=x_batch_chunks[0]))
                in_vals.extend(list(zip_longest(self.labels_per_gpu,
                                                y_batch_chunks,
                                                fillvalue=y_batch_chunks[0])))
                in_vals.extend([(self.num_splits, this_n_splits)])
                in_vals.extend([(self.num_batches, len(x_batch))])
                in_vals.extend([(self.prev_err, self.loss_value)])
                feed_dict = {p: v for(p, v) in in_vals}

                # Callback
                fetch_dict = self.batch_do(num_devs, feed_dict)

                # Callback
                self.batch_end(minibatch, fetch_dict)

            self.epoch_end(minibatch, fetch_dict)

            # Verify epochs' loop exit conditions
            if self.estop:
                tf.logging.info('Early Stop!')
                self.sv.request_stop()
                break
            if self.last_epoch:
                tf.logging.info('Last epoch!')
                self.sv.request_stop()
                break

        self.experiment_end(fetch_dict)
        return self.return_val

    # ###########
    # Callbacks #
    # ###########
    def experiment_begin(self):
        # Add TqdmHandler
        handler = TqdmHandler()
        handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT, None))
        logger = logging.getLogger('tensorflow')
        del(logger.handlers[0])  # Remove the default handler
        logger.addHandler(handler)

        tf.logging.info('\nTrain dataset params:\n{}\n'.format(
            self.cfg.dataset_params))
        tf.logging.info('Validation dataset params:\n{}\n\n'.format(
            self.cfg.valid_params))
        if pygtk and self.cfg.debug_of:
            cv2.namedWindow("rgb-optflow")

        self.train = self.Dataset(
            which_set='train',
            return_list=False,
            **self.cfg.dataset_params)

        # Setup loop parameters
        self.cum_iter = self.sv.global_step.eval(self.sess)
        self.patience_counter = 0
        self.estop = False
        self.last_epoch = False
        self.history_acc = np.array([]).tolist()

        # Start the training loop
        self.start = time()
        tf.logging.info("Beginning main loop...")
        self.loss_value = 0

    def epoch_begin(self):
        summary_val = tf.Summary.Value(tag='control_flow/Epoch',
                                       simple_value=self.epoch_id + 1)
        summary = tf.Summary(value=[summary_val])
        self.sv.summary_computed(self.sess, summary,
                                 global_step=self.epoch_id)
        self.pbar = tqdm(total=self.train.nbatches,
                         bar_format='{n_fmt}/{total_fmt}{desc}'
                                    '{percentage:3.0f}%|{bar}| '
                                    '[{elapsed}<{remaining},'
                                    '{rate_fmt}{postfix}]')

    def batch_begin(self, minibatch):
        # TODO move in reconvnets
        x_batch = minibatch['data']
        # Show optical flow for debug
        if pygtk and self.cfg.debug_of:
            for x_b in x_batch:
                for x_frame in x_b:
                    rgb_of_frame = np.concatenate(
                        [x_frame[..., :3], x_frame[..., 3:]],
                        axis=1).astype(np.float32)
                    rgb_of_frame = cv2.cvtColor(rgb_of_frame,
                                                cv2.COLOR_RGB2BGR)
                    cv2.imshow("rgb-optflow", rgb_of_frame)
                    cv2.waitKey(100)

        # reset_states(model, sh)

    def batch_do(self, num_devs, feed_dict):
        # TODO move in reconvnets
        # Do not add noise if loss is less than threshold
        # TODO: It should be IoU or any other metric, but in this
        # case our loss is Dice Coefficient so it's fine
        self.loss_value = (-1.0 if self.loss_value < -self.cfg.thresh_loss
                           else self.loss_value)

        train_dict = {
            'avg_loss': self.avg_loss[True],
            'train_op': self.train_graph_outs['grad_ops'][num_devs]}
        train_summary_dict = {
            'avg_loss': self.avg_loss[True],
            'train_op': self.train_graph_outs['grad_ops'][num_devs],
            'summary_op': self.train_graph_outs['summary_ops'][num_devs]}

        # Compute (summaries and) loss
        if self.cum_iter % self.cfg.train_summary_freq == 0:
            fetch_dict = self.sess.run(train_summary_dict,
                                       feed_dict=feed_dict)
            self.sv.summary_computed(self.sess,
                                     fetch_dict['summary_op'])
        else:
            fetch_dict = self.sess.run(train_dict, feed_dict=feed_dict)
        return fetch_dict

    def batch_end(self, minibatch, fetch_dict):
        self.cum_iter += 1
        self.pbar.set_description('({:3d}) Ep {:d}'.format(
            self.cum_iter + 1, self.epoch_id + 1))
        self.pbar.set_postfix(
            {'D': '{:.2f}s'.format(self.t_data_load),
             'loss': '{:.3f}'.format(fetch_dict['avg_loss'])})
        self.pbar.update(1)

    def epoch_end(self, minibatch, fetch_dict):
        self.pbar.close()
        # TODO Add val_every_iter?
        # valid_wait = 0 if valid_wait == 1 else valid_wait - 1

        # Is it also the last epoch?
        if self.sv.should_stop() or self.epoch_id == self.cfg.max_epochs - 1:
            self.last_epoch = True

        # Early stop if patience is over
        self.patience_counter += 1
        if (self.epoch_id >= self.cfg.min_epochs and
                self.patience_counter >= self.cfg.patience):
            self.estop = True

        # Validate if last epoch, early stop or we reached valid_every
        metrics_val = {}
        validate_fn = getattr(self, "validate_fn", None)
        if callable(validate_fn) and (
             self.last_epoch or self.estop or not self.val_skip):

            for s in self.cfg.val_on_sets:
                metrics_val[s] = validate_fn(
                    self.val_inputs_per_gpu,
                    self.val_graph_outs[s],
                    which_set=s)

            # TODO gsheet
            self.history_acc.append([metrics_val.get('valid')])

            # Did we improve *validation* metric?
            best_hist = np.array(self.history_acc).max()
            if (len(self.history_acc) == 0 or
                    metrics_val.get('valid') >= best_hist):
                tf.logging.info('## Best model found! ##')
                t_save = time()
                checkpoint_path = os.path.join(self.cfg.checkpoints_dir,
                                               '{}_best.ckpt'.format(
                                                   self.cfg.model_name))

                self.saver.save(self.sess, checkpoint_path,
                                global_step=self.cfg.global_step)
                t_save = time() - t_save
                tf.logging.info('Checkpoint saved in {}s'.format(t_save))

                self.patience_counter = 0
                self.estop = False
            # Start skipping again
            self.val_skip = max(1, self.cfg.val_every_epochs) - 1
        else:
            # We skipped validation, decrease the counter
            self.val_skip -= 1
        self.metrics_val = metrics_val

    def experiment_end(self, fetch_dict):
        max_valid_idx = np.argmax(np.array(self.history_acc))
        best = self.history_acc[max_valid_idx]
        tf.logging.info('\nBest: Mean Class iou - Valid {:.5f}\n'.format(best))

        end = time()
        m, s = divmod(end - self.start, 60)
        h, m = divmod(m, 60)
        tf.logging.info("Total time elapsed: %d:%02d:%02d" % (h, m, s))

        self.return_val = best
        # # Move complete models and stuff to shared fs
        # tf.logging.info('\n\nEND OF TRAINING!!\n\n')

        # def move_if_exist(filename, dest):
        #     if not os.path.exists(os.path.dirname(dest)):
        #         os.makedirs(os.path.dirname(dest))
        #     try:
        #         shutil.move(filename, dest)
        #     except IOError:
        #         tf.logging.error('Move error: {} does not exist.'.format(
        #             filename))

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
        # validate = True  # Print the best model's test error
