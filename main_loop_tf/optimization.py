import tensorflow as tf
import gflags

from tensorflow.contrib.layers.python.layers.optimizers import (
    _clip_gradients_by_norm,
    _add_scaled_noise_to_gradients,
    _multiply_gradients)
from tensorflow.python.ops.metrics_impl import _streaming_confusion_matrix
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import training
from tensorflow.python.training.learning_rate_decay import (exponential_decay,
                                                            piecewise_constant,
                                                            polynomial_decay,
                                                            natural_exp_decay,
                                                            inverse_time_decay)
from utils import squash_maybe

smooth = 1.


def get_optimizer(optimizer):
    try:
        return getattr(training, optimizer + 'Optimizer')
    except AttributeError:
        return getattr(training, optimizer.capitalize() + 'Optimizer')


def apply_lr_decay(cfg, global_step):
    # Learning rate schedule
    if cfg.lr_decay is None:
        lr = cfg.lr
    elif cfg.lr_decay == 'exp':
        lr = exponential_decay(cfg.lr,
                               global_step,
                               cfg.decay_steps,
                               cfg.decay_rate,
                               staircase=cfg.staircase)
    elif cfg.lr_decay == 'piecewise':
        lr = piecewise_constant(global_step,
                                cfg.lr_boundaries,
                                cfg.lr_values)
    elif cfg.lr_decay == 'polynomial':
        lr = polynomial_decay(cfg.lr,
                              global_step,
                              cfg.decay_steps,
                              end_learning_rate=cfg.end_lr,
                              power=cfg.power,
                              cycle=cfg.staircase)

    elif cfg.lr_decay == 'natural_exp':
        lr = natural_exp_decay(cfg.lr,
                               global_step,
                               cfg.decay_steps,
                               cfg.decay_rate,
                               staircase=cfg.staircase)
    elif cfg.lr_decay == 'inverse_time':
        lr = inverse_time_decay(cfg.lr,
                                global_step,
                                cfg.decay_steps,
                                cfg.decay_rate,
                                staircase=cfg.staircase)

    elif cfg.lr_decay == 'STN':
        epoch = tf.cast(global_step / cfg.decay_steps, tf.int32)
        lr = cfg.lr * tf.pow(0.5, tf.cast(epoch / 50, cfg._FLOATX))
    else:
        raise NotImplementedError()
    return lr


def process_gradients(cfg, global_step, prev_err, grads_and_vars):
    """Add noise and multipliers to gradient

    Parameters
    ----------
    grads_and_vars: list
        The list of gradients to be modified.
    """
    grad_noise_scale = _get_grad_noise_scale(cfg, global_step, prev_err)

    if grad_noise_scale is not None:
        grads_and_vars = _add_scaled_noise_to_gradients(
            grads_and_vars, grad_noise_scale)

    # Optionally multiply some gradients
    if cfg.grad_multiplier is not None:
        grads_and_vars = _multiply_gradients(
            grads_and_vars, cfg.gradient_multiplier)
        if not grads_and_vars:
            raise ValueError(
                'Empty list of (gradient,var) pairs encountered. '
                'This is most likely caused by an improper value '
                'of cfg.gradient_multipliers.')

    # Optionally clip gradients by global norm
    if isinstance(cfg.max_grad_norm, float):
        grads_and_vars = _clip_gradients_by_norm(
            grads_and_vars, cfg.max_grad_norm)
    elif callable(cfg.max_grad_norm):
        grads_and_vars = cfg.max_grad_norm(grads_and_vars)
    elif cfg.max_grad_norm is not None:
        raise ValueError(
            "Unknown type %s for cfg.max_grad_norm" %
            type(cfg.max_grad_norm))
    return grads_and_vars, grad_noise_scale


def compute_and_process_grads(self, loss_out, var_list=None,
                              gate_gradients=None, aggregation_method=None,
                              colocate_gradients_with_ops=True, name=None,
                              grad_loss=None, phase_set_dev='',
                              summaries=None, loss=None):
    """Minimize over multiple devices with grad noise

    Extend Optimizer.minimize() in several ways:
        * Add noise and multipliers
        * Add various gradient summaries
        * Be stateful and keep trace of previously computed
          gradients
        * Add a control dependency on update ops before computing
          the gradient
        * Average gradient over the devices processed so far.
        * It also does not have global_step as an argument, as it's
          in the state of the optimizer already.
        * Sets colocate_gradients_with_ops to True by default.
    """
    if loss is not None:
        raise ValueError('This Optimizer expects a dictionary of '
                         'losses rather than a single loss. Do not '
                         'use it as a normal tf optimizer but rather '
                         'rely on loss_out')
    if gate_gradients is None:
        gate_gradients = self.optimizer.GATE_OP

    # This device's gradients
    grads_and_vars = self.optimizer.compute_gradients(
        loss_out['loss'], var_list=var_list,
        gate_gradients=gate_gradients,
        aggregation_method=aggregation_method,
        colocate_gradients_with_ops=colocate_gradients_with_ops,
        grad_loss=grad_loss)

    # Check if no gradient
    vars_with_grad = [v for g, v in grads_and_vars if g is not None]
    if not vars_with_grad:
        raise ValueError(
            "No gradients provided for any variable, check your graph "
            "for ops that do not support gradients, between variables "
            "%s and loss %s." % ([str(v) for _, v in grads_and_vars],
                                 loss))

    # Add suffix to name_scope (rather than nesting name scopes)
    with tf.name_scope(None):
        with tf.name_scope(phase_set_dev + 'grad_processing'):
            # Add noise and multipliers to gradient
            grads_and_vars, grad_noise_scale = process_gradients(
                self.cfg, self.global_step, self.sym_prev_err,
                grads_and_vars)

    # Create some summaries
    add_summaries(grads_and_vars, grad_noise_scale, phase_set_dev,
                  summaries)

    return grads_and_vars


def _get_grad_noise_scale(cfg, global_step, prev_err):
    if cfg.grad_noise_decay is None:
        grad_noise_scale = cfg.grad_noise_scale
    elif cfg.grad_noise_decay == 'annealing':
        """
        Adds annealed gaussian noise to the gradients at
        every time step, by decaying the variance at each
        time step
        g_t <- g_t + N(0, sigma_t^2)
        sigma_t^2 = eta / (1 + t)^gamma

        with eta selected from {0.01, 0.3, 1.0) and
        gamma = 0.55
        See: "Adding gradient noise improves learning
        for very deep networks",
        http://arxiv.org/pdf/1511.06807v1.pdf
        """
        eta = cfg.grad_noise_scale ** 0.5
        gamma = 0.55 / 2
        grad_noise_scale = eta * tf.pow(tf.cast(
            global_step + 1, cfg._FLOATX), -gamma)
    elif cfg.grad_noise_decay == 'neural_gpu':
        if prev_err is None:
            grad_noise_scale = cfg.grad_noise_scale
        else:
            eta = cfg.grad_noise_scale
            gamma = 0.55
            grad_noise_scale = eta * tf.sqrt(
                prev_err * tf.pow(tf.cast(
                    global_step + 1, cfg._FLOATX), -gamma))
    else:
        raise NotImplementedError('Unknown value of '
                                  'cfg.grad_noise_decay: %s' %
                                  cfg.grad_noise_decay)
    return grad_noise_scale


def add_summaries(grads_and_vars, grad_noise_scale, phase_set_dev,
                  summaries=[]):
    if summaries == []:
        return

    # Add summary for the noise on the gradient
    # -----------------------------------------
    if grad_noise_scale is not None:
        tf.summary.scalar(phase_set_dev + "grad_noise", grad_noise_scale,
                          summaries)

    # Add histograms for variables, grads and grad norms
    # --------------------------------------------------
    with tf.name_scope(None):
        with tf.name_scope(phase_set_dev + 'grad_norms') as norm_scope:
            with tf.name_scope(phase_set_dev + 'grad_hists') as hist_scope:
                for grad, var in grads_and_vars:
                    if isinstance(grad, tf.IndexedSlices):
                        grad = grad.values

                    if grad is not None:
                        # Remove the implicit name_scope of the variable
                        # scope
                        var_name = var.op.name.replace('model/', '')
                        _, var_name = squash_maybe('', var_name)
                        # Write the summary
                        with tf.name_scope(norm_scope):
                            tf.summary.scalar(var_name,
                                              tf.global_norm([grad]),
                                              summaries)
                        with tf.name_scope(hist_scope):
                            tf.summary.histogram(var_name,
                                                 grad,
                                                 summaries)


def average_gradients(grad_dict, phase_set_dev, up_to_dev=None):
    """Calculate the mean gradient for the devices processed so far

    Note
    ----
    This function provides a synchronization point across all towers.

    Parameters
    ----------
    grad_dict: Dict of lists of gradients (per device).
        A dictionary with variables as keys and a list of gradients per
        device as values.
    phase_set_dev: string
        A name scope
    up_to_dev: int
        Up to which device to compute the average on

    Return
    ------
    List of pairs of (gradient, variable) where the gradient has been
    averaged across all towers.
    """
    average_grads = []
    with tf.name_scope(None):
        with tf.name_scope(phase_set_dev + 'grad_avg'):
            for v, grads_list in grad_dict.iteritems():
                if isinstance(v, tf.Variable):
                    name = '' + v.name.replace('/', '_').replace(':', '_')
                    sname = 'pre_grad_avg_stack_%s' % name
                    rname = 'grad_avg_%s' % name
                else:
                    sname = 'pre_grad_avg_stack'
                    rname = 'grad_avg'
                if up_to_dev is not None:
                    grads_list = grads_list[:up_to_dev + 1]
                grad_list = tf.stack(axis=0, values=grads_list, name=sname)
                avg_grad = tf.reduce_mean(grad_list, 0, name=rname)
                average_grads.append((avg_grad, v))
    return average_grads


def average_list_gradients(tower_grads):
    """Calculate the mean gradient for each shared variable across all towers.

    Note
    ----
    This function provides a synchronization point across all towers.

    Parameters
    ----------
    tower_grads: List of lists of (gradient, variable) tuples.
        The outer list is over individual gradients. The inner list is
        over the gradient calculation for each tower.

    Return
    ------
    List of pairs of (gradient, variable) where the gradient has been
    averaged across all towers.
    """
    average_grads = []
    for grads_and_vars in zip(*tower_grads):
        # Note that each grads_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        # TODO no need for the loop here
        # grad.append(mean(grad_gpu[0..N]), var_gpu0)
        grads = []
        for g, _ in grads_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grads_and_vars[0][1]
        grads_and_vars = (grad, v)
        average_grads.append(grads_and_vars)

    return average_grads


def dice_coef(labels, logits, class_dice=1):
    cfg = gflags.cfg
    '''
    Dice loss -- works ONLY for binary classification.
        labels: GT index class (0 or 1)
        logits: softmax output in one-hot notation
    '''
    with tf.variable_scope('dice_coef'):
        labels_f = tf.cast(tf.reshape(labels, [-1]), cfg._FLOATX)
        logits_f = tf.reshape(logits[..., class_dice], [-1])
        intersection = tf.reduce_sum(labels_f * logits_f)
        dice = (2. * intersection + smooth) / (
            tf.reduce_sum(labels_f) + tf.reduce_sum(logits_f) + smooth)

    return dice


def mean_iou(labels,
             predictions,
             num_classes,
             weights=None,
             metrics_collections=None,
             updates_collections=None,
             name=None):

    """Calculate per-step mean Intersection-Over-Union (mIOU).

    Mean Intersection-Over-Union is a common evaluation metric for
    semantic image segmentation, which first computes the IOU for each
    semantic class and then computes the average over classes.
    IOU is defined as follows:
      IOU = true_positive / (true_positive + false_positive + false_negative).
    The predictions are accumulated in a confusion matrix,
    weighted by `weights`, and mIOU is then calculated from it.
     For estimation of the metric over a stream of data, the function
     creates 7
     an `update_op` operation that updates these variables and returns
     the `mean_iou`.  If `weights` is `None`, weights default to 1.
     Use weights of 0 to mask values.
     Args:
      labels: A `Tensor` of ground truth labels with shape [batch size] and of
        type `int32` or `int64`. The tensor will be flattened if its rank > 1.
      predictions: A `Tensor` of prediction results for semantic labels, whose
        shape is [batch size] and type `int32` or `int64`. The tensor will be
        flattened if its rank > 1.
      num_classes: The possible number of labels the prediction task can
        have. This value must be provided, since a confusion matrix of
        dimension = [num_classes, num_classes] will be allocated.
      weights: Optional `Tensor` whose rank is either 0, or the same rank as
        `labels`, and must be broadcastable to `labels` (i.e., all dimensions
        must be either `1`, or the same as the corresponding `labels`
        dimension).
      metrics_collections: An optional list of collections that `mean_iou`
        should be added to.
      updates_collections: An optional list of collections `update_op` should
      be added to.
      name: An optional variable_scope name.
     Returns:
      mean_iou: A `Tensor` representing the mean intersection-over-union.
      update_op: An operation that increments the confusion matrix.
     Raises:
      ValueError: If `predictions` and `labels` have mismatched shapes, or if
        `weights` is not `None` and its shape doesn't match `predictions`,
        or if either `metrics_collections` or `updates_collections` are not a
        list or tuple.
    """
    with tf.variable_scope(
          name, 'mean_iou', (predictions, labels, weights)):
        # Check if shape is compatible.
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())

        total_cm, update_op = _streaming_confusion_matrix(labels, predictions,
                                                          num_classes, weights)

        reset_cm_op = tf.assign(total_cm, tf.zeros_like(total_cm,
                                                        total_cm.dtype,
                                                        'reset_cm'))

        def compute_mean_iou(name):
            """Compute the mean intersection-over-union via the confusion
            matrix."""
            sum_over_row = math_ops.to_float(math_ops.reduce_sum(total_cm, 0))
            sum_over_col = math_ops.to_float(math_ops.reduce_sum(total_cm, 1))
            cm_diag = math_ops.to_float(array_ops.diag_part(total_cm))
            denominator = sum_over_row + sum_over_col - cm_diag

            # If the value of the denominator is 0, set it to 1 to avoid
            # zero division.
            denominator = array_ops.where(
                math_ops.greater(denominator, 0),
                denominator,
                array_ops.ones_like(denominator))
            iou = math_ops.div(cm_diag, denominator)
            return math_ops.reduce_mean(iou, name=name), iou

        mean_iou_v, iou = compute_mean_iou('mean_iou')

        if metrics_collections:
            ops.add_to_collections(metrics_collections, mean_iou_v)

        if updates_collections:
            ops.add_to_collections(updates_collections, update_op)

        return mean_iou_v, iou, update_op, reset_cm_op


def dice_coef_loss(labels, logits, class_dice=1):
    return -dice_coef(labels, logits, class_dice)
