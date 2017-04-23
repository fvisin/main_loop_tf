def build_model(inputs, is_training):
    from tensorflow.contrib import slim
    import gflags

    cfg = gflags.cfg
    conv = slim.conv2d(inputs,
                       num_outputs=cfg.nclasses,
                       kernel_size=(1, 1),
                       stride=1)

    if is_training:
        return conv
    else:
        return conv * 2


if __name__ == '__main__':
    import sys
    from main_loop_tf.main import run

    # You can also add fixed values like this
    argv = sys.argv
    argv += ['--dataset', 'camvid']

    run(argv, build_model)
