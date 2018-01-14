import gflags
from main_loop_tf import gflags_ext


# ============ Learning
# class_balance=None

# ============ Optimizers
# Common params for optimizers
gflags.DEFINE_string('optimizer', 'adam', 'The optimizer')
gflags_ext.DEFINE_multidict('optimizer_params', {},
                            'The params for the optimizer')
# gflags.DEFINE_integer('BN_mode', 2, 'The batch normalization mode')
gflags.DEFINE_float('lr', 1e-4, 'Initial Learning Rate')
gflags.DEFINE_string('lr_decay', None, 'LR Decay schedule')
gflags.DEFINE_bool('staircase', False, 'Whether to apply decay in a '
                   'discrete staircase, as opposed to continuous, '
                   'fashion.')

# Specific params for Piecewise Constant
gflags_ext.DEFINE_intlist('lr_boundaries', None,
                          'A list of Tensors or ints or floats'
                          'with strictly increasing entries, and with all'
                          'elements having the same type as the index.')
gflags_ext.DEFINE_floatlist('lr_values', None, 'List of learning rate')

# Specific params for Exponential Decay
gflags.DEFINE_integer('decay_steps', None,
                      'How often to decay the LR [in steps]')
gflags.DEFINE_float('decay_rate', None,
                    'Decay rate at each decay step (0,1)')

# Specific params for Polynomial Decay
gflags.DEFINE_float('power', None,
                    'the power of the polynomial decay. '
                    'Defaults to linear, 1.0. Usually 0.5')
gflags.DEFINE_float('end_lr', None, 'The minimal end Learning Rate')

# Specific params for Neural GPU
gflags.DEFINE_string("thresh_loss", 0.7,
                     "Do not add noise if loss is less than threshold")

# ============ Regularization and gradients
# Regularization parameters
gflags.DEFINE_float('weight_decay', 0, 'The weight decay')
# We leave dropout to be defined in the model, since it is usually applied
# in different ways to different parts of the model
# gflags.DEFINE_float('dropout', 0, 'The dropout probability')

# Gradient processing
gflags.DEFINE_float("max_grad_norm", None, "Clip gradients to this norm.")
gflags.DEFINE_float("grad_noise_scale", None,
                    "Gradient noise scale {0.01, 0.3, 1.0} ")
gflags.DEFINE_string("grad_noise_decay", None,
                     "Gradient Noise Decay Schedule [neural_gpu]")
gflags.DEFINE_float("grad_multiplier", None, "Gradient Multipliers")
