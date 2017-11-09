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

        with tf.Graph().as_default():
            self.global_step = tf.Variable(0,
                                           trainable=False,
                                           name='global_step',
                                           dtype='int32')
            # Set Optimizer
            if Optimizer is None:
                self.Optimizer = get_optimizer(cfg.optimizer)(
                    cfg=cfg, global_step=self.global_step)
            else:
                self.Optimizer = Optimizer(cfg=cfg,
                                           global_step=self.global_step)

        self.val_skip = (cfg.val_skip_first if cfg.val_skip_first else
                         max(1, cfg.val_every_epochs) - 1)

        # Init variables
        self.val_summary_ops = {}
        self.val_cm_update_ops = {}
        self.val_cm_reset_ops = {}
        self.val_outs = {}
        self.val_metrics = {}

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

        # BUILD GRAPH
        tf.logging.info("Building the model ...")
        with tf.Graph().as_default():
            # Create a list of input placeholders for each GPU.
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
                self.__build_device_graph__(which_set='train')

                # Build the validation graphs (reusing variables)
                for s in ['eval_' + v for v in cfg.val_on_sets]:
                    self.__build_device_graph__(which_set=s)

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

    def __build_device_graph__(self, which_set):
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
        is_training = which_set == 'train'
        reuse_variables = not is_training

        labels_per_gpu = self.labels_per_gpu
        if is_training:
            inputs_per_gpu = self.train_inputs_per_gpu
            summaries_str = 'train_summaries_%s'
        else:
            inputs_per_gpu = self.val_inputs_per_gpu
            summaries_str = 'val_%s_summaries' % which_set + '_%s'

        tower_out = {}
        summaries = []
        for device in cfg.devices:
            device_str = device.replace('/', '').replace(':', '').lower()
            dev_set_str = '{}_{}'.format(device_str, which_set)
            summaries.append(summaries_str % device_str)
        these_s = summaries
        self.train_ops = []

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
                model_outs = self.build_model(dev_inputs, is_training)

                assert isinstance(model_outs, dict), """
                    Your model should return a dictionary"""
                assert 'out_preact' in model_outs, """Your model
                    function should return a dictionary with attribute
                    'out_preact'!"""
                assert 'out_act' in model_outs, """Your model function
                    should return a dictionary with attribute 'out_act'!"""
                assert 'pred' in model_outs, """Your model function should
                    return a dictionary with at least attribute 'pred'!"""

                # Store this device's model outputs in the tower
                for k, v in model_out.iteritems():
                    tower_out.setdefault(k, []).append(v)

                # Loss
                # TODO: create **loss_params to be specified externally
                # when specializing Experiment
                loss_outs = self.build_loss(dev_labels, model_outs,
                                            inputs=dev_inputs)

                # Validate loss_outs
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

                # Gradients
                if is_training:
                    # Compute gradients and add noise
                    # Gradient descent operation
                    # Create a *list* of gradient update ops. The t-th element
                    # of the list updates the gradients of the devices *up to*
                    # the t-th device
                    dev_train_op = self.Optimizer.distributed_minimize(
                        loss_outs=loss_outs['loss'],
                        device=device,
                        colocate_gradients_with_ops=True,
                        dev_set_scope=dev_set_scope,
                        summaries=these_s)
                    self.train_ops.append(dev_train_op)

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
        out_dict = {}
        for k, v in tower_out.iteritems():
            # Convert from list of tensors to concatenated tensor
            out_dict[k] = tf.concat(tower_out[k], axis=0, name='concat_%s' % k)
            out_dict[k] = out_dict[k][:self.num_batches]

        tf.summary.scalar('control_flow/batch_size_' + which_set,
                          tf.shape(out_dict['pred'])[0], summaries)

        # Concatenate the per-gpu placeholders to get a placeholder for the
        # full list of gpus and one for the subset to be used for
        # the minibatch with less batches
        labels = tf.concat(self.labels_per_gpu, axis=0, name='concat_labels')
        # Remove the unused batches from the flattened labels
        # (equivalent to labels[:np.prod(out_dict.shape)])
        labels = labels[:tf.shape(tf.reshape(out_dict['pred'], [-1]))[0]]

        #############
        # SUMMARIES #
        #############
        # Add the mean loss summary
        self.avg_tower_loss = self.Optimizer.mean_losses[:self.num_splits]
        tf.summary.scalar(dev_set_scope + '_Mean_tower_loss/Total_Loss',
                          self.avg_tower_loss, summaries)
        if is_training:
            # Add the per-component loss summaries
            for k, v in self.Optimizer.mean_losses_comp:
                tf.summary.scalar(dev_set_scope + '_Mean_tower_loss/%s' % k,
                                  v[:self.num_splits], summaries)

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

        # Create the summary ops out of the summary collections that we used
        # at graph creation time
        summary_ops = []
        for s in summaries:
            summary_ops.append(tf.summary.merge(tf.get_collection_ref(key=s)))

        # Save in the Experiment object whatever is useful elsewhere
        if is_training:
            self.train_summary_ops = summary_ops
            self.train_out_dict = out_dict
        else:
            metrics_out, cm_update_ops, cm_reset_ops = self.metrics()

            self.val_summary_ops[which_set] = summary_ops
            self.val_cm_update_ops[which_set] = cm_update_ops
            self.val_cm_reset_ops[which_set] = cm_reset_ops
            self.val_out[which_set] = out_dict
            self.val_metrics[which_set] = metrics_out

    def run(self):
        self.__init_sess__()
        with self.sess:
            if self.cfg.hyperparams_summaries is not None:
                # write Hyper parameters text summaries
                summary_str = self.sess.run(self.summary_text_op)
                self.sv.summary_computed(self.sess, summary_str)

            # Start training loop
            return self.main_loop()

    def validate(self):
        self.__init_sess__()
        with self.sess:
            # Perform validation only
            metrics = {}
            validate = getattr(self, "validate", None)
            if callable(validate):
                for s in self.cfg.val_on_sets:
                    metrics[s] = validate(
                        self.val_placeholders,
                        self.val_outs['eval_' + s],
                        self.val_metrics['eval_' + s],
                        self.val_summary_ops['eval_' + s],
                        self.val_cm_updatet_ops['eval_' + s],
                        self.val_cm_reset_ops['eval_' + s],
                        which_set='eval_' + s)
                return metrics
            else:
                raise NotImplementedError('Validation method not implemented!')

    def __init_sess__(self):
        cfg = self.cfg
        with tf.Graph().as_default() as graph:
            # Group global and local init into one op. Could be split into
            # two different ops and passed to `init_op` and `local_init_op`
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())

            self.saver = tf.train.Saver(
                max_to_keep=cfg.checkpoints_to_keep)

            sv = Supervisor(
                graph=graph,
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
            sess = sv.managed_session(cfg.supervisor_master, tf_config)
            if self.cfg.debug:
                from tensorflow.python import debug as tf_debug
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)
                sess.add_tensor_filter("has_inf_or_nan",
                                       tf_debug.has_inf_or_nan)
            self.sess = sess

    def main_loop(self):
        cfg = gflags.cfg

        # Add TqdmHandler
        handler = TqdmHandler()
        handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT, None))
        logger = logging.getLogger('tensorflow')
        del(logger.handlers[0])  # Remove the default handler
        logger.addHandler(handler)

        tf.logging.info('\nTrain dataset params:\n{}\n'.format(
            cfg.dataset_params))
        tf.logging.info('Validation dataset params:\n{}\n\n'.format(
            cfg.valid_params))
        if pygtk and cfg.debug_of:
            cv2.namedWindow("rgb-optflow")

        self.train = self.Dataset(
            which_set='train',
            return_list=False,
            **cfg.dataset_params)

        # Setup loop parameters
        self.cum_iter = self.sv.global_step.eval(self.sess)
        self.patience_counter = 0
        self.estop = False
        self.last_epoch = False
        self.history_acc = np.array([]).tolist()

        # Start the training loop
        start = time()
        tf.logging.info("Beginning main loop...")
        self.loss_value = 0

        while not self.sv.should_stop():
            self.epoch_id = (self.cum_iter+1) // self.train.nbatches
            summary_val = tf.Summary.Value(tag='control_flow/Epoch',
                                           simple_value=self.epoch_id + 1)
            summary = tf.Summary(value=[summary_val])
            self.sv.summary_computed(self.sess, summary,
                                     global_step=self.epoch_id)
            pbar = tqdm(total=self.train.nbatches,
                        bar_format='{n_fmt}/{total_fmt}{desc}'
                                   '{percentage:3.0f}%|{bar}| '
                                   '[{elapsed}<{remaining},'
                                   '{rate_fmt}{postfix}]')

            for batch_id in range(self.train.nbatches):
                self.cum_iter = self.sv.global_step.eval(self.sess)
                iter_start = time()

                # inputs and labels
                minibatch = self.train.next()
                t_data_load = time() - iter_start
                x_batch, y_batch = minibatch['data'], minibatch['labels']
                # sh = inputs.shape  # do NOT provide a list of shapes

                # Show optical flow for debug
                if pygtk and cfg.debug_of:
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

                # Do not add noise if loss is less than threshold
                # TODO: It should be IoU or any other metric, but in this
                # case our loss is Dice Coefficient so it's fine
                self.loss_value = (-1.0 if self.loss_value < -cfg.thresh_loss
                                   else self.loss_value)

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

                end = this_n_splits - 1
                train_dict = {
                    'avg_tower_loss': self.avg_tower_loss,
                    'train_op': self.train_ops[end]}
                train_summary_dict = {
                    'avg_tower_loss': self.avg_tower_loss,
                    'train_op': self.train_ops[end],
                    'summary_op': self.train_summary_ops[end]}

                # Get the per-device inputs
                x_batch_chunks, y_batch_chunks = split_in_chunks(x_batch,
                                                                 y_batch,
                                                                 this_n_splits)

                # Fill the placeholders with data up to this_n_splits, and
                # then repeat one of the chunks. Note that this will be
                # ignored later on (see comment where placeholders are created)
                in_vals = list(zip_longest(self.inputs_per_gpu,
                                           x_batch_chunks,
                                           fillvalue=x_batch_chunks[0]))
                in_vals.extend(list(zip_longest(self.labels_per_gpu,
                                                y_batch_chunks,
                                                fillvalue=y_batch_chunks[0])))
                in_vals.extend([(self.num_splits, this_n_splits)])
                in_vals.extend([(self.num_batches, len(x_batch))])
                in_vals.extend([(self.prev_err, self.loss_value)])
                feed_dict = {p: v for(p, v) in in_vals}

                # Compute (summaries and) loss
                if self.cum_iter % cfg.train_summary_freq == 0:
                    fetch_dict = self.sess.run(
                        train_summary_dict,
                        feed_dict=feed_dict)
                    self.sv.summary_computed(self.sess,
                                             fetch_dict['summary_op'])
                else:
                    fetch_dict = self.sess.run(train_dict, feed_dict=feed_dict)

                pbar.set_description('({:3d}) Ep {:d}'.format(
                    self.cum_iter + 1, self.epoch_id + 1))
                pbar.set_postfix(
                    {'D': '{:.2f}s'.format(t_data_load),
                     'loss': '{:.3f}'.format(fetch_dict['avg_tower_loss'])})
                pbar.update(1)

            # It's the end of the epoch
            pbar.close()
            # TODO Add val_every_iter?
            # valid_wait = 0 if valid_wait == 1 else valid_wait - 1

            # Is it also the last epoch?
            if self.sv.should_stop() or self.epoch_id == cfg.max_epochs - 1:
                self.last_epoch = True

            # Early stop if patience is over
            self.patience_counter += 1
            if (self.epoch_id >= cfg.min_epochs and
                    self.patience_counter >= cfg.patience):
                self.estop = True

            # Validate if last epoch, early stop or we reached valid_every
            validate = getattr(self, "validate", None)
            if callable(validate) and (
                 self.last_epoch or self.estop or not self.val_skip):

                metric_val = {}
                for s in cfg.val_on_sets:
                    metric_val[s] = validate(
                        self.val_placeholders,
                        self.val_outs['eval_' + s],
                        self.val_metrics['eval_' + s],
                        self.val_summary_ops['eval_' + s],
                        self.val_cm_reset_ops['eval_' + s],
                        self.val_cm_update_ops['eval_' + s],
                        which_set='eval_' + s)

                # TODO gsheet
                self.history_acc.append([metric_val.get('valid')])

                # Did we improve *validation* metric?
                best_hist = np.array(self.history_acc).max()
                if (len(self.history_acc) == 0 or
                        metric_val.get('valid') >= best_hist):
                    tf.logging.info('## Best model found! ##')
                    t_save = time()
                    checkpoint_path = os.path.join(cfg.checkpoints_dir,
                                                   '{}_best.ckpt'.format(
                                                       cfg.model_name))

                    self.saver.save(self.sess, checkpoint_path,
                                    global_step=cfg.global_step)
                    t_save = time() - t_save
                    tf.logging.info('Checkpoint saved in {}s'.format(t_save))

                    self.patience_counter = 0
                    self.estop = False
                # Start skipping again
                self.val_skip = max(1, cfg.val_every_epochs) - 1
            else:
                # We skipped validation, decrease the counter
                self.val_skip -= 1

            # Verify epochs' loop exit conditions
            if self.estop:
                tf.logging.info('Early Stop!')
                self.sv.request_stop()
                break
            if self.last_epoch:
                tf.logging.info('Last epoch!')
                self.sv.request_stop()
                break

        max_valid_idx = np.argmax(np.array(self.history_acc))
        best = self.history_acc[max_valid_idx]
        tf.logging.info('\nBest: Mean Class iou - Valid {:.5f}\n'.format(best))

        end = time()
        m, s = divmod(end - start, 60)
        h, m = divmod(m, 60)
        tf.logging.info("Total time elapsed: %d:%02d:%02d" % (h, m, s))

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
        return best
