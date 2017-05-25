import logging
from subprocess import check_output
import sys
import tqdm

import gflags
import matplotlib
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers.optimizers import (
    _clip_gradients_by_norm,
    _add_scaled_noise_to_gradients,
    _multiply_gradients)

# Initialize numpy's random seed
# import settings  # noqa


# Force matplotlib not to use any Xwindows backend.
matplotlib.use('Agg')
sys.setrecursionlimit(99999)
tf.logging.set_verbosity(tf.logging.INFO)


def split_in_chunks(x_batch, y_batch, gpus_used):
    '''Return the splits per gpu

    Return
        * the batches per gpu
        * the labels elements per gpu
    '''

    x_batch_chunks = np.array_split(x_batch, gpus_used)
    y_batch_chunks = np.array_split(y_batch, gpus_used)
    for i in range(gpus_used):
        y_batch_chunks[i] = y_batch_chunks[i].flatten()

    return x_batch_chunks, y_batch_chunks


def apply_loss(labels, net_out, loss_fn, weight_decay, is_training,
               return_mean_loss=False, mask_voids=True):
    '''Applies the user-specified loss function and returns the loss

    Note:
        SoftmaxCrossEntropyWithLogits expects labels NOT to be one-hot
        and net_out to be one-hot.
    '''

    cfg = gflags.cfg

    if mask_voids and len(cfg.void_labels):
        # TODO Check this
        print('Masking the void labels')
        mask = tf.not_equal(labels, cfg.void_labels)
        labels *= tf.cast(mask, 'int32')  # void_class --> 0 (random class)
        # Train loss
        loss = loss_fn(labels=labels,
                       logits=tf.reshape(net_out, [-1, cfg.nclasses]))
        mask = tf.cast(mask, 'float32')
        loss *= mask
    else:
        # Train loss
        loss = loss_fn(labels=labels,
                       logits=tf.reshape(net_out, [-1, cfg.nclasses]))

    if is_training:
        loss = apply_l2_penalty(loss, weight_decay)

    # Return the mean loss (over pixels *and* batches)
    if return_mean_loss:
        if mask_voids and len(cfg.void_labels):
            return tf.reduce_sum(loss) / tf.reduce_sum(mask)
        else:
            return tf.reduce_mean(loss)
    else:
        return loss


def apply_l2_penalty(loss, weight_decay):
    with tf.variable_scope('L2_regularization'):
        trainable_variables = tf.trainable_variables()
        l2_penalty = tf.add_n([tf.nn.l2_loss(v) for v in trainable_variables
                               if 'bias' not in v.name])
        loss += l2_penalty * weight_decay

    return loss


def process_gradients(gradients,
                      gradient_noise_scale=None,
                      gradient_multipliers=None,
                      clip_gradients=None):

    """
    gradient_noise_scale: float or None, adds 0-mean normal noise scaled
        by this value.
    gradient_multipliers: dict of variables or variable names to floats.
        If present, gradients for specified variables will be multiplied
        by given constant.
    clip_gradients: float, callable or `None`. If float, is provided, a global
      clipping is applied to prevent the norm of the gradient to exceed this
      value. Alternatively, a callable can be provided e.g.: adaptive_clipping.
      This callable takes a `list` of `(gradients, variables)` `tuple`s and
      returns the same thing with the gradients modified.
    """

    if gradient_noise_scale is not None:
        gradients = _add_scaled_noise_to_gradients(
            gradients, gradient_noise_scale)

    # Multiply some gradients.
    if gradient_multipliers is not None:
        gradients = _multiply_gradients(
            gradients, gradient_multipliers)
        if not gradients:
            raise ValueError(
                "Empty list of (gradient,var) pairs"
                "encountered. This is most likely "
                "to be caused by an improper value "
                "of gradient_multipliers.")

    # Optionally clip gradients by global norm.
    if isinstance(clip_gradients, float):
        gradients = _clip_gradients_by_norm(
            gradients, clip_gradients)
    elif callable(clip_gradients):
        gradients = clip_gradients(gradients)
    elif clip_gradients is not None:
        raise ValueError(
            "Unknown type %s for clip_gradients" % type(clip_gradients))

    return gradients


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
    Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
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


def save_repos_hash(params_dict, this_repo_name, packages=['theano']):
    # Repository hash and diff
    params_dict[this_repo_name + '_hash'] = check_output('git rev-parse HEAD',
                                                         shell=True)[:-1]
    diff = check_output('git diff', shell=True)
    if diff != '':
        params_dict[this_repo_name + '_diff'] = diff
    # packages
    for p in packages:
        this_pkg = __import__(p)
        params_dict[p + '_hash'] = this_pkg.__version__


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


class TqdmHandler(logging.StreamHandler):
    # From https://github.com/tqdm/tqdm/issues/193#issuecomment-233212170
    def __init__(self):
        logging.StreamHandler.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        tqdm.tqdm.write(msg)
