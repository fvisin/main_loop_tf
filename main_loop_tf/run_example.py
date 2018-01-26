"""A minimal working example of run file and model/loss

This code is meant to provide an example to implement your own run file
and model/loss and to show the most basic usage of this framework.

This code is NOT meant to be an example of how to properly tackle the
semantic segmentation problem of Camvid, nor of how to build a neural
network in general. It's only meant to show the basic API of this
framework.
"""
try:
    from itertools import izip_longest as zip_longest
except:
    from itertools import zip_longest

from main_loop_tf import Experiment
from main_loop_tf.utils import split_in_chunks
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
                           num_outputs=64,
                           kernel_size=(5, 5),
                           padding='SAME',
                           stride=1)
        conv = slim.conv2d(conv,
                           num_outputs=128,
                           kernel_size=(5, 5),
                           padding='SAME',
                           stride=1)
        conv = slim.conv2d(conv,
                           num_outputs=256,
                           kernel_size=(5, 5),
                           padding='SAME',
                           stride=1)
        conv = slim.conv2d(conv,
                           num_outputs=256,
                           kernel_size=(5, 5),
                           padding='SAME',
                           stride=1)
        conv = slim.conv2d(conv,
                           num_outputs=256,
                           kernel_size=(5, 5),
                           padding='SAME',
                           stride=1)
        conv = slim.conv2d(conv,
                           num_outputs=256,
                           kernel_size=(1, 1),
                           padding='SAME',
                           stride=1)
        conv = slim.conv2d(conv,
                           # We might want to be smarter than this and
                           # ignore void classes, but for this example
                           # let's ignore it.
                           num_outputs=cfg.nclasses_w_void,
                           kernel_size=(1, 1),
                           padding='SAME',
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

    def validate_fn(self, graph_out, which_set):
        cfg = self.cfg

        if hasattr(self, '_minibatch'):
            minibatch = self._minibatch
            dataset = self.train
        else:
            print('getting a new minibatch - no random seed!!')
            dataset = self.Dataset(
                which_set='train',
                return_list=False,
                **self.cfg.dataset_params)
            minibatch = dataset.next()
        x_batch = minibatch['data']

        # Is this batch shorter than batch_size?
        # Check if this batch will not be processed by all the devices.
        # When the sequence is shorter than seq_length or the number of
        # batches is smaller than batch_size, the batch will be smaller
        # than usual. When this happens we might not be able to feed
        # all the CPUs/GPUs altogether. In that case here we compute
        # the number of GPUs that we can use for the current batch
        # Spread the batch over the lowest number of GPUs
        this_n_splits = len(x_batch) // cfg.batch_size
        if len(x_batch) % cfg.batch_size != 0:
            this_n_splits += 1

        # Get the per-device inputs
        minibatch_chunks = split_in_chunks(minibatch, this_n_splits,
                                           flatten_keys=['labels'])

        # Associate each placeholder (of each device) with its input data. Note
        # that the data is split in chunk, one per device. If this_n_splits is
        # smaller than the number of devices, the placeholders of the "extra"
        # devices are filled with the data of the first chunk. This is
        # necessary to feed the graph with the expected number of inputs, but
        # note that the extra outputs and loss will be ignored (see comment
        # where placeholders are created)
        feed_dict = {}
        for p_dict, batch_dict in zip_longest(self.per_dev_placeholders[False],
                                              minibatch_chunks,
                                              fillvalue=minibatch_chunks[0]):
            for p_name, p_obj in p_dict.iteritems():
                feed_dict[p_obj] = batch_dict[p_name]

        # Extend the user-defined placeholders with those needed by the
        # main loop
        feed_dict[self.sym_num_devs] = this_n_splits
        feed_dict[self.sym_num_batches] = len(x_batch)
        self._feed_dict = feed_dict

        # Use the op for the number of devices the current batch can feed
        sym_pred = graph_out['model_outs']['pred']
        val_dict = {'pred': sym_pred}

        # Save one sample on disk
        fetch_dict = self.unhookedsess.run(val_dict, feed_dict=feed_dict)
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        cmap = mpl.colors.ListedColormap(dataset.cmap)
        fname = '/home/francesco/exp/main_loop_tf/main_loop_tf/checkpoints/'
        fname += 'camvid'
        if hasattr(self, 'epoch_id'):
            fname += str(self.epoch_id)
        fname += '.png'
        plt.imsave(fname, fetch_dict['pred'][0], cmap=cmap, vmin=0, vmax=12)
        return 0

    def batch_begin(self):
        self._t_data_load = 0
        # Overfit on one image!
        if not hasattr(self, '_minibatch'):
            import matplotlib as mpl
            import matplotlib.pyplot as plt
            self._minibatch = self.train.next()
            self._t_data_load = 10
            cmap = mpl.colors.ListedColormap(self.train.cmap)
            fname = '/home/francesco/exp/main_loop_tf/main_loop_tf/'
            fname += 'checkpoints/camvid_GT.png'
            plt.imsave(fname, self._minibatch['labels'][0], cmap=cmap,
                       vmin=0, vmax=12)
            return 0


if __name__ == '__main__':
    import sys

    # You can also add fixed values like this
    argv = sys.argv
    argv += ['--dataset', 'camvid']
    argv += ['--max_epochs', '50']
    argv += ['--val_every_epochs', '1']
    argv += ['--nouse_threads']
    # argv += ['--devices', '/gpu:0,/gpu:1']
    # argv += ['--devices', '/cpu:0']

    exp = ExampleExperiment(argv)
    if exp.cfg.validate:
        exp.validate()
    else:
        exp.run()
