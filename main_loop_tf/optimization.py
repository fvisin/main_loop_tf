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
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import training
from tensorflow.python.training.learning_rate_decay import (exponential_decay,
                                                            piecewise_constant,
                                                            polynomial_decay,
                                                            natural_exp_decay,
                                                            inverse_time_decay)
from tensorflow.python.training.training import Optimizer

from utils import squash_maybe

smooth = 1.


def get_optimizer(optimizer):
    try:
        BaseClass = getattr(training, optimizer + 'Optimizer')
    except AttributeError:
        BaseClass = getattr(training, optimizer.capitalize() + 'Optimizer')
    return type("DistributedOptimizer",
                (DistributedOptimizer, BaseClass), {})


class DistributedOptimizer(object):
    """This class will be specialized by get_optimizer"""
    def __init__(self, cfg, global_step):
        self.cfg = cfg
        self.initial_lr = cfg.lr
        self.lr_decay = cfg.lr_decay

        # Learning rate schedule
        if cfg.lr_decay is None:
            lr = self.initial_lr
        elif cfg.lr_decay == 'exp':
            lr = exponential_decay(cfg.lr,
                                   self.global_step,
                                   cfg.decay_steps,
                                   cfg.decay_rate,
                                   staircase=cfg.staircase)
        elif cfg.lr_decay == 'piecewise':
            lr = piecewise_constant(self.global_step,
                                    cfg.lr_boundaries,
                                    cfg.lr_values)
        elif cfg.lr_decay == 'polynomial':
            lr = polynomial_decay(cfg.lr,
                                  self.global_step,
                                  cfg.decay_steps,
                                  end_learning_rate=cfg.end_lr,
                                  power=cfg.power,
                                  cycle=cfg.staircase)

        elif cfg.lr_decay == 'natural_exp':
            lr = natural_exp_decay(cfg.lr,
                                   self.global_step,
                                   cfg.decay_steps,
                                   cfg.decay_rate,
                                   staircase=cfg.staircase)
        elif cfg.lr_decay == 'inverse_time':
            lr = inverse_time_decay(cfg.lr,
                                    self.global_step,
                                    cfg.decay_steps,
                                    cfg.decay_rate,
                                    staircase=cfg.staircase)

        elif cfg.lr_decay == 'STN':
            epoch = tf.cast(self.global_step / cfg.decay_steps, tf.int32)
            lr = cfg.lr * tf.pow(0.5, tf.cast(epoch / 50, cfg._FLOATX))
        else:
            raise NotImplementedError()
        self.lr = lr

        # Per-device gradients
        self._tower_grads = []
        self._tower_loss = []
        self._tower_comp = []
        self.mean_losses = []
        self.mean_losses_comp = {}
        return super(DistributedOptimizer, self).__init__(
            learning_rate=lr, **cfg.optimizer_params)

    def __process_gradients(self, gradients):
        """Add noise and multipliers to gradient

        Parameters
        ----------
        gradients: list
            The list of gradients to be modified.
        """
        grad_noise_scale = self.__get_grad_noise_scale(gradients)

        if grad_noise_scale is not None:
            gradients = _add_scaled_noise_to_gradients(
                gradients, grad_noise_scale)

        # Optionally multiply some gradients
        if self.cfg.grad_multiplier is not None:
            gradients = _multiply_gradients(
                gradients, self.cfg.gradient_multiplier)
            if not gradients:
                raise ValueError(
                    'Empty list of (gradient,var) pairs encountered. '
                    'This is most likely caused by an improper value '
                    'of cfg.gradient_multipliers.')

        # Optionally clip gradients by global norm
        if isinstance(self.cfg.max_grad_norm, float):
            gradients = _clip_gradients_by_norm(
                gradients, self.cfg.max_grad_norm)
        elif callable(self.cfg.max_grad_norm):
            gradients = self.cfg.max_grad_norm(gradients)
        elif self.cfg.max_grad_norm is not None:
            raise ValueError(
                "Unknown type %s for cfg.max_grad_norm" %
                type(self.cfg.max_grad_norm))

        return gradients, grad_noise_scale

    def __compute_mean_gradients(self, loss_out):
        # Extend the per-device lists of losses and component-losses
        self._tower_loss.append(loss_out['loss'])
        for el in loss_out['components']:
            # Create a dictionary of components with a list of their
            # values per each device
            for k, v in el.iteritems():
                self._tower_comp.setdefault(k, []).append(v)

        # Compute the mean losses for the devices we processed so far.
        # One of these will be dynamically selected at run-time (depending on
        # the number of devices in use, i.e., the numerical value of
        # num_splits_tensor) and used to update the loss summaries
        # Overall loss
        # -------------------------------------------------------------------
        dev_losses_stack = tf.stack(self._tower_loss, axis=0,
                                    name='concat_losses')
        self.mean_losses.append(tf.reduce_mean(dev_losses_stack))

        # Loss per component
        for (comp_k, comp_loss) in self._tower_comp.iteritems():
            loss_comp_stack = tf.stack(comp_loss, axis=0,
                                       name='concat_losses_comp_%s' % comp_k)
            self.mean_losses_comp.setdefault(comp_k, []).append(
                tf.reduce_mean(loss_comp_stack))

    def __get_grad_noise_scale(self, gradients):
        if self.cfg.grad_noise_decay is None:
            grad_noise_scale = self.cfg.grad_noise_scale
        elif self.cfg.grad_noise_decay == 'annealing':
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
            eta = self.cfg.grad_noise_scale ** 0.5
            gamma = 0.55 / 2
            grad_noise_scale = eta * tf.pow(tf.cast(
                self.global_step + 1, self.cfg._FLOATX), -gamma)
        elif self.cfg.grad_noise_decay == 'neural_gpu':
            eta = self.cfg.grad_noise_scale
            gamma = 0.55
            grad_noise_scale = eta * tf.sqrt(
                self.prev_err * tf.pow(tf.cast(
                    self.global_step + 1, self.cfg._FLOATX), -gamma))
        else:
            # Raise ValueError
            raise NotImplementedError('Unknown value of '
                                      'cfg.grad_noise_decay: %s' %
                                      self.cfg.grad_noise_decay)

        return grad_noise_scale

    def __add_summaries(self, gradients, grad_noise_scale,
                        dev_set_scope, summaries=[]):
        if summaries == []:
            return

        # Add summary for the noise on the gradient
        # -----------------------------------------
        with tf.name_scope(dev_set_scope):
            tf.summary.scalar("NoiseGrad", grad_noise_scale,
                              summaries)

        # Add histograms for variables, grads and grad norms
        # --------------------------------------------------
        for gradient, variable in gradients:
            if isinstance(gradient, tf.IndexedSlices):
                grad_vals = gradient.values
            else:
                grad_vals = gradient

            if grad_vals is not None:
                # Remove model_name/
                var_name = variable.op.name.replace(
                    self.cfg.model_name + '/', '')
                scope_str = dev_set_scope + '_%s'  # metric
                scope_str, var_name = squash_maybe(scope_str,
                                                   var_name)
                scope_str += '_%s'  # var name
                # Write the summary
                with tf.name_scope(None):
                    tf.summary.scalar(
                        scope_str % ('GradientNorm', var_name),
                        tf.global_norm([grad_vals]), summaries)
                    tf.summary.histogram(
                        scope_str % ('GradientHist', var_name),
                        grad_vals, summaries)

        # Remove the name_scopes (the one from the variable_scope
        # and the one from the name_scope)
        with tf.name_scope(dev_set_scope):
            name = ('clipped_grad_norm' if self.cfg.max_grad_norm else
                    'grad_norm')
            tf.summary.scalar('Global_norm/' + name,
                              tf.global_norm(list(zip(*gradients))[0]),
                              summaries)

    def distributed_minimize(self, loss_out, global_step=None, var_list=None,
                             gate_gradients=None, aggregation_method=None,
                             colocate_gradients_with_ops=False, name=None,
                             grad_loss=None, device=None, dev_set_scope='',
                             summaries=None, loss=None):
        """Distributed minimize with grad noise

        Extend minimize() in several ways:
            * Add noise and multipliers
            * Add various gradient summaries
            * Be stateful and keep trace of previously computed
            gradients
            * Add a control dependency on update ops before computing
              the gradient
            * Average gradient over the devices processed so far.
        """
        if loss is not None:
            raise ValueError('This Optimizer expects a dictionary of '
                             'losses rather than a single loss')
        if gate_gradients is None:
            gate_gradients = self.GATE_OP  # access parent class attrib

        grads_and_vars = self.compute_gradients(
            loss_out['loss'], var_list=var_list, gate_gradients=gate_gradients,
            aggregation_method=aggregation_method,
            colocate_gradients_with_ops=colocate_gradients_with_ops,
            grad_loss=grad_loss)

        # Add noise and multipliers to gradient
        grads_and_vars, grad_noise_scale = self.__process_gradients(
            grads_and_vars)

        # Compute the average gradients over the devices seen so far
        self.__compute_mean_gradients(loss_out)

        # Create some summaries
        self.__add_summaries(grads_and_vars, grad_noise_scale, dev_set_scope,
                             summaries)

        # Save gradients for each gpu to be averaged out
        self._tower_grads.append(grads_and_vars)

        # Gradient descent
        # ----------------
        # Note the collection contains the ops for the devices processed
        # so far
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # TODO Do we still need this?
        # Impose graph dependency so that update operations are computed
        # even if they're are not explicit in the outputs of session.run
        with tf.control_dependencies(update_ops):
            vars_with_grad = [v for g, v in grads_and_vars
                              if g is not None]
            if not vars_with_grad:
                raise ValueError(
                    "No gradients provided for any variable, check "
                    "your graph for ops that do not support gradients, "
                    "between variables %s and loss %s." %
                    ([str(v) for _, v in grads_and_vars], loss))

            # Average the gradients over the devices processed so far
            grads_and_vars = average_gradients(self._tower_grads)

            grad_op = self.apply_gradients(grads_and_vars,
                                           global_step=global_step,
                                           name=name)
        # TODO: Averaged gradients visualisation
        # Add the histograms of the gradients
        # with tf.name_scope('grad_summaries'):
        #     for grad, var in grads_and_vars:
        #         if grad is not None:
        #             summaries['training'].append(
        #                 tf.summary.histogram(
        #                   var.op.name + '/gradients', grad))

        return grad_op


def average_gradients(tower_grads):
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
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        # TODO no need for the loop here
        # grad.append(mean(grad_gpu[0..N]), var_gpu0)
        grads = []
        for g, _ in grad_and_vars:
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
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

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
    with variable_scope.variable_scope(
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
