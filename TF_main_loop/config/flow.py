import gflags


# ============ Flow control
#         # early_stop_metric='subsets_avg_val_jaccard_fg',
#         # early_stop_strategy='max',
#         # val_every_batches=None,  # validate every n batches (else epoch)
gflags.DEFINE_integer('checkpoints_to_keep', 2, 'The number of checkpoints '
                      'to keep', lower_bound=0)
gflags.DEFINE_string('checkpoints_dir', './checkpoints', 'The path where '
                     'the model checkpoints are stored')
gflags.DEFINE_string('checkpoints_file', None, 'The filename of the '
                     'checkpoint')
gflags.DEFINE_string('tmp_path', './models/tmp_', 'Where to save temporary '
                     'stuff')
gflags.DEFINE_integer('val_every_epochs', 1, 'Validation frequency, in epochs',
                      lower_bound=1)
gflags.DEFINE_spaceseplist('val_on_sets', 'valid', 'On which sets to '
                           'perform validation')
gflags.DEFINE_integer('val_skip_first', 0, 'How many epochs to skip before '
                      'validating', lower_bound=0)
gflags.DEFINE_integer('min_epochs', 1, 'The minimum number of epochs '
                      'before early stopping is possible', lower_bound=1)
gflags.DEFINE_integer('max_epochs', 100, 'The maximum number of epochs',
                      lower_bound=1)
gflags.DEFINE_integer('patience', 100, 'The number of validation with no '
                      'improvement the model will wait before early stopping',
                      lower_bound=1)
gflags.DEFINE_bool('do_validation_only', False, 'If True does one round '
                   'of validation')
