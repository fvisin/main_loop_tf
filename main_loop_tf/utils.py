import tensorflow as tf
import gflags
import sys

import matplotlib
# Initialize numpy's random seed
# import settings  # noqa

from subprocess import check_output
from tensorflow.contrib.layers.python.layers.optimizers import(
    _clip_gradients_by_norm,
    _add_scaled_noise_to_gradients,
    _multiply_gradients)

# Force matplotlib not to use any Xwindows backend.
matplotlib.use('Agg')
sys.setrecursionlimit(99999)
tf.logging.set_verbosity(tf.logging.INFO)


def compute_chunk_size(batch_size, npixels):
    '''Return the splits per gpu

    Return
        * the number of batches per gpu
        * the number of labels elements per gpu
    '''
    cfg = gflags.cfg

    # Compute the shape of the input chunk for each GPU
    a_device_shape = int(batch_size / cfg.num_splits)
    split_dim = [a_device_shape] * cfg.num_splits
    if batch_size % cfg.num_splits != 0:
        # Fill the last device with what remains
        split_dim[-1] = batch_size % cfg.num_splits
    # Labels are flattened, so we need to take into account the
    # pixels as well
    labels_split_dim = [el * npixels for el in split_dim]
    return split_dim, labels_split_dim


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
    reg_losses = tf.losses.get_regularization_losses()
    if len(reg_losses):
        l2_penalty = tf.add_n(reg_losses)
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
