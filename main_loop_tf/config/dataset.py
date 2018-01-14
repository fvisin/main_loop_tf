import gflags
from main_loop_tf import gflags_ext


# ============ Dataset params
gflags.DEFINE_integer('batch_size', 1, 'The batch size', lower_bound=0)
gflags_ext.DEFINE_intlist('crop_size', None, 'The training crop-size')
gflags.DEFINE_string('crop_mode', 'random', '')
gflags.DEFINE_float('smart_crop_threshold', 0.5, '')
gflags.DEFINE_integer('smart_crop_search_step', 10, '')

gflags.DEFINE_string('dataset', None, 'The dataset')
gflags.DEFINE_string('of', None, 'Whether to have the opt flow as an input')
gflags.DEFINE_integer('seq_length', None, 'The length of the sequence, in '
                      'case the input is a video', lower_bound=0)
gflags.DEFINE_integer('seq_per_subset', None, 'Limit the number of sequences '
                      'per subset (i.e., video or prefix or directory).  If '
                      'zero all the possible sequences will be returned',
                      lower_bound=0)
gflags.DEFINE_bool('return_extended_sequences', False, 'If True, repeats '
                   'the first and last frame of each video to allow for '
                   'middle frame prediction')
gflags.DEFINE_bool('return_middle_frame_only', False, 'If True, return '
                   'the middle frame segmentation mask only for each sequence')
gflags.DEFINE_integer('overlap', None, 'The overlap (in number of frames) '
                      'of consecutive sequences of the same video for '
                      'training. If negative not all the frames will be '
                      'returned')
gflags.DEFINE_bool('use_threads', True, 'Whether to use parallel threads to '
                   'load the dataset')
gflags.DEFINE_integer('nthreads', 3, 'The number of threads ot use, if '
                      'use_threads is True', lower_bound=0)
gflags.DEFINE_integer('data_queues_size', 30, 'The size of the data '
                      'loading queue. Consider increasing this if you suspect '
                      'starvation.', lower_bound=1)
gflags.DEFINE_bool('remove_mean', False, 'If True each image or frame '
                   'will be divided by the dataset mean, if any.')
gflags.DEFINE_bool('divide_by_std', False, 'If True each image or '
                   'frame will be divided by the dataset std dev, if any.')
gflags.DEFINE_bool('remove_per_img_mean', False, 'If True each image or frame '
                   'will be normalized to have zero mean')
gflags.DEFINE_bool('divide_by_per_img_std', False, 'If True each image or '
                   'frame will be normalized to have unit variance')
gflags.DEFINE_integer('val_batch_size', 1, 'The validation batch size',
                      lower_bound=0)
gflags.DEFINE_integer('val_overlap', None, 'The overlap (in number of frames) '
                      'of consecutive sequences of the same video for '
                      'validation. If negative not all the frames will be '
                      'returned')
gflags_ext.DEFINE_multidict('train_extra_params', {},
                            'Dataset train extra params')
gflags_ext.DEFINE_multidict('val_extra_params', {},
                            'Dataset valid extra params')
