from subprocess import check_output
import sys

# Force matplotlib not to use any Xwindows backend.
# Initialize numpy's random seed
#import settings  # noqa

import gflags
import tensorflow as tf

# from validate import evaluate_generator, cat_masked_accuracy, dice_coef_loss


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
    a_gpu_shape = int(batch_size / cfg.num_gpus)
    split_dim = [a_gpu_shape] * cfg.num_gpus
    if batch_size % cfg.num_gpus != 0:
        # Fill the last GPU with what remains
        split_dim[-1] = batch_size % cfg.num_gpus
    # Labels are flattened, so we need to take into account the
    # pixels as well
    labels_split_dim = [el * npixels for el in split_dim]
    return split_dim, labels_split_dim


def apply_loss(labels, net_out, loss_fn, weight_decay, is_training,
               return_mean_loss=False, mask_voids=True, scope=None):
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
        labels *= tf.to_int32(mask)
        mask_out = tf.cast(tf.expand_dims(tf.reshape(
            mask, tf.shape(net_out)[:3]), -1), cfg._FLOATX)
        net_out *= mask_out

    # Train loss
    loss = loss_fn(labels=labels,
                   logits=tf.reshape(net_out, [-1, cfg.nclasses]))

    # Add L2 penalty only if we are training
    # loss = tf.cond(is_training,
    #                lambda: get_l2_penalty(loss, weight_decay),
    #                lambda: tf.identity(loss))
    if is_training:
        loss = get_l2_penalty(loss, weight_decay)

    # Return the mean loss (over pixels *and* batches)
    if return_mean_loss:
        if mask_voids and len(cfg.void_labels):
            return tf.reduce_sum(loss) / tf.reduce_sum(mask_out)
        else:
            return tf.reduce_mean(loss)
    else:
        return loss


def get_l2_penalty(loss, weight_decay):
    reg_losses = tf.losses.get_regularization_losses()
    if len(reg_losses):
        l2_penalty = tf.add_n(reg_losses)
        loss += l2_penalty * weight_decay

    return tf.identity(loss)


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
