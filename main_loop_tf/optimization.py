import tensorflow as tf
import gflags

from tensorflow.python.ops.metrics_impl import _streaming_confusion_matrix
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
smooth = 1.


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
