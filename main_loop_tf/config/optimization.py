import gflags
from main_loop_tf import gflags_ext


# ============ Learning
# optimizer_params --> make it a string and parse it?
# class_balance=None
# gflags.DEFINE_integer('BN_mode', 2, 'The batch normalization mode')
gflags.DEFINE_float('weight_decay', 0, 'The weight decay')
gflags.DEFINE_string('optimizer', 'adam', 'The optimizer')
gflags_ext.DEFINE_multidict('optimizer_params', {}, 'The params for the '
                            'optimizer')
gflags.DEFINE_string('loss_fn', 'sparse_softmax_cross_entropy_with_logits',
                     'The loss function')
gflags.DEFINE_float('dropout', 0, 'The dropout probability')
gflags.DEFINE_bool('stateful_validation', True, 'If True the state of '
                   'the RNNs will be kept to process the next batch (if '
                   'consecutive)')
gflags.DEFINE_bool('use_extra_BN', False, 'Whether to add a BN layer on '
                   'the input')

gflags.DEFINE_float("max_grad_norm", None, "Clip gradients to this norm.")
gflags.DEFINE_float("grad_noise_scale", None, "Gradient noise scale.")
gflags.DEFINE_float("grad_multiplier", None, "Gradient Multipliers")
