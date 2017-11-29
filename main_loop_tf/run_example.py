"""A minimal working example of run file and model/loss

This code is meant to provide an example to implement your own run file
and model/loss and to show the most basic usage of this framework.

This code is NOT meant to be an example of how to properly tackle the
semantic segmentation problem of Camvid, nor of how to build a neural
network in general. It's only meant to show the basic API of this framework.
"""
from main_loop_tf import Experiment
import tensorflow as tf
from tensorflow.contrib import slim


class ExampleExperiment(Experiment):
    """An example implementation of the Experiment abstract class"""

    def build_model(self, inputs, is_training):
        """The definition of the architecture

        This method defines the sequence of transformations that compute
        the output of the network given its input.
        """
        import gflags

        cfg = gflags.cfg
        ret = {}
        conv = slim.conv2d(inputs['data'],
                           num_outputs=50,
                           kernel_size=(1, 1),
                           stride=1)
        conv = slim.conv2d(conv,
                           num_outputs=250,
                           kernel_size=(1, 1),
                           stride=1)
        conv = slim.conv2d(conv,
                           # We might want to be smarter than this and
                           # ignore void classes, but for this example
                           # let's ignore it.
                           num_outputs=cfg.nclasses_w_void,
                           kernel_size=(1, 1),
                           stride=1,
                           activation_fn=None,
                           normalizer_fn=None)

        ret['out_preact'] = conv
        act = tf.nn.softmax(conv)
        ret['out_act'] = act
        ret['pred'] = tf.argmax(act, axis=-1)
        return ret

    def build_loss(self, placeholders, model_out, is_training,
                   **loss_params):
        """The definition of the loss function(s)

        This method defines at least one loss function to be used to
        train the model. When the loss is a composition of multiple
        losses, the single losses can be specified in 'components' for
        visualization purposes.
        """
        labels = placeholders['labels']
        preact = model_out['out_preact']
        shape = preact.get_shape().as_list()[1:3]
        labels = tf.reshape(labels, [-1] + shape)

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=preact, labels=labels))

        # in this simple example the loss is not an aggregate of
        # multiple components, so components only contains the loss
        # itself
        return {'loss': loss, 'components': {'main_loss': loss}}

    def validate(self):
        """A validation function to evaluate the model should be defined."""
        return 0


if __name__ == '__main__':
    import sys

    # You can also add fixed values like this
    argv = sys.argv
    argv += ['--dataset', 'camvid']

    exp = ExampleExperiment(argv)
    if exp.cfg.do_validation_only:
        exp.validate()
    else:
        exp.run()
