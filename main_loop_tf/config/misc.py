import gflags


# Checkpoints
gflags.DEFINE_integer('checkpoints_to_keep', 2, 'The number of checkpoints '
                      'to keep', lower_bound=0)
gflags.DEFINE_string('checkpoints_dir', './checkpoints', 'The path where '
                     'the model checkpoints are stored')
gflags.DEFINE_list('devices', ['/cpu:0'], 'A list of devices to use')
gflags.DEFINE_bool('debug_of', False,
                   'Show rgb and optical flow of each batch in a window')

# gflags.DEFINE_bool('restore_model', False, 'Whether to reload the weights of '
#                    'the model')

# Other flags we might want to define (see also config/flow.py):
# See https://www.tensorflow.org/versions/r0.10/tutorials/monitors/
#                customizing_the_evaluation_metrics
# metrics=[],  # TODO add additional metrics
# val_metrics=['dice_loss', 'acc', 'jaccard'],
# TODO parametrize according to which metric to save the model (best val loss?)
