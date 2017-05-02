import gflags


#         vgg_regularize=False,
#         rnn_regularize=False,
#         up_regularize=False,
#         pretrained_vgg=False,
# See https://www.tensorflow.org/versions/r0.10/tutorials/monitors/
#                customizing_the_evaluation_metrics
#         # metrics=[],  # TODO add additional metrics
#         # val_metrics=['dice_loss', 'acc', 'jaccard'],
#         # TODO parametrize which ones to keep (best val loss?)
#         # save_name,  --> come salvo le immagini?
gflags.DEFINE_list('devices', ['/cpu:0'], 'A list of devices to use')
gflags.DEFINE_string('vgg_weights_file', './vgg16_weights.npz', 'The '
                     'path of the vgg weights file')
gflags.DEFINE_list('vgg_var_to_load',
                   ['Conv1', 'Conv2', 'Conv3', 'Conv4', 'Conv5'], 'Which vgg '
                   'layers to reload')
gflags.DEFINE_bool('restore_model', False, 'Whether to reload the weights of '
                   'the model')
gflags.DEFINE_bool('save_samples', True, 'Whether to save GT/Prediction '
                   'image summaries')
gflags.DEFINE_bool('save_raw_predictions', False, 'Whether to save raw '
                   'predictions on disk')
gflags.DEFINE_bool('save_heatmap', True, 'Whether to save heatmap image '
                   'image summaries')
gflags.DEFINE_bool('save_animations', False, 'Whether to save gif '
                   'animations')
gflags.DEFINE_bool('save_images_on_disk', False, 'If True the images '
                   'will be saved also on disk as well as in '
                   'Tensorboard')
gflags.DEFINE_integer('img_summaries_freq', 1, 'How often, in terms '
                      'of batches, to save image summaries during '
                      'validation')
