import tensorflow as tf
import gflags
smooth = 1.


def dice_coef(labels, logits, class_dice=1):
    '''
    Dice loss -- works ONLY for binary classification.
        labels: GT index class (0 or 1)
        logits: softmax output in one-hot notation
    '''
    cfg = gflags.cfg

    with tf.variable_scope('dice_coef'):
        labels_f = tf.cast(tf.reshape(labels, [-1]), cfg._FLOATX)
        logits_f = tf.reshape(logits[..., class_dice], [-1])
        intersection = tf.reduce_sum(labels_f * logits_f)
        loss = (2. * intersection + smooth) / (tf.reduce_sum(labels_f) +
                                               tf.reduce_sum(logits_f) +
                                               smooth)
        return loss


def dice_coef_loss(labels, logits, class_dice=1):
    return -dice_coef(labels, logits, class_dice)
