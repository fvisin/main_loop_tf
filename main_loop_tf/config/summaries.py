import gflags
from main_loop_tf import gflags_ext


# Summaries and samples
gflags.DEFINE_bool('group_summaries', True, 'If True, groups the scalar '
                   'summaries by `layer_sublayer` rather than just by '
                   '`layer`. The total number of summaries remains unchanged')
gflags.DEFINE_integer('train_summary_freq', 10,
                      'How frequent save train summaries (in steps)')
gflags_ext.DEFINE_multidict('hyperparams_summaries',
                            {'1-Dataset': ['dataset',
                                           'batch_size',
                                           'crop_size',
                                           'seq_length',
                                           'of',
                                           'remove_mean',
                                           'divide_by_std',
                                           'remove_per_img_mean',
                                           'divide_by_per_img_std'],
                             '2-Optimization': ['optimizer',
                                                'lr',
                                                'lr_decay',
                                                'decay_steps',
                                                'decay_rate',
                                                'staircase',
                                                'lr_boundaries',
                                                'lr_values',
                                                'power',
                                                'end_lr'],
                             '3-GradientProcessing': ['max_grad_norm',
                                                      'grad_noise_scale',
                                                      'grad_noise_decay',
                                                      'thresh_loss',
                                                      'grad_multiplier'],
                             '4-Regularization': ['weight_decay']
                             },
                            'Hyperparams you want to show in the summaries')
