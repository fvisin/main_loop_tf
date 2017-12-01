from pprint import pprint
import tensorflow as tf

from optimization import average_gradients
import sys


def pprint_run(feed_dict, avg):
    tprint(feed_dict)
    avg_num = sess.run(avg, feed_dict)
    print(avg_num)


# Pretty print fetch_dict
def tprint(dic):
    val_lst = []
    val_dict = {}
    for t, v in dic.iteritems():
        val_lst.append(v)
        val_dict[t.name] = v
    sys.stdout.write('Input:\n')
    pprint({k: v for k, v in val_dict.iteritems() if k.startswith('g1')})
    if any(k.startswith('g2') for k in val_dict.keys()):
        pprint({k: v for k, v in val_dict.iteritems() if k.startswith('g2')})


with tf.Session().as_default() as sess:
    g11 = tf.placeholder(tf.float32, name='g11')
    g12 = tf.placeholder(tf.float32, name='g12')
    g13 = tf.placeholder(tf.float32, name='g13')
    g21 = tf.placeholder(tf.float32, name='g21')
    g22 = tf.placeholder(tf.float32, name='g22')
    g23 = tf.placeholder(tf.float32, name='g23')

    grads_and_vars1 = [(g11, 'v1'),
                       (g12, 'v2'),
                       (g13, 'v3')]
    grads_and_vars2 = [[g21, 'v1'],
                       [g22, 'v2'],
                       [g23, 'v3']]

    dev_grads1 = {}
    dev_grads12 = {}
    for g, v in grads_and_vars1:
        dev_grads1.setdefault(v, []).append(g)
        dev_grads12.setdefault(v, []).append(g)
    print('\ndev_grads1')
    pprint(dev_grads1)

    for g, v in grads_and_vars2:
        dev_grads12.setdefault(v, []).append(g)
    print('\ndev_grads12')
    pprint(dev_grads12)

    # Compute average
    avg1 = average_gradients(dev_grads1, 'avg1')
    print('\navg1')
    pprint(avg1)

    avg12 = average_gradients(dev_grads12, 'avg12')
    print('\navg12')
    pprint(avg12)

    # TEST
    # ONE DEVICE
    print('\nONE DEVICE')
    avg1 = [el0 for el0, el1 in avg1]  # strip non-tensors
    # 1,1,1
    pprint_run({g11: 1, g12: 1, g13: 1}, avg1)
    # 1,2,3
    pprint_run({g11: 1, g12: 2, g13: 3}, avg1)
    # 3,.5,3
    pprint_run({g11: 3, g12: 0.5, g13: 3}, avg1)
    # 0,0,0.1
    pprint_run({g11: 0, g12: 0., g13: 0.1}, avg1)

    # TWO DEVICES
    print('\nTWO DEVICES')
    avg12 = [el0 for el0, el1 in avg12]  # strip non-tensors
    # 1,1,1
    pprint_run({g11: 1, g12: 1, g13: 1,
                g21: 1, g22: 1, g23: 1}, avg12)
    # 1,2,3
    pprint_run({g11: 1, g12: 2, g13: 3,
                g21: 1, g22: 2, g23: 3}, avg12)
    # 3,.5,3
    pprint_run({g11: 3, g12: 0.5, g13: 3,
                g21: 3, g22: 0.5, g23: 3}, avg12)
    # 0,0,0.1
    pprint_run({g11: 0, g12: 0., g13: 0.1,
                g21: 0, g22: 0., g23: 0.1}, avg12)
    # mix 1
    pprint_run({g11: 3, g12: 2., g13: 2,
                g21: 1, g22: 1., g23: 0.5}, avg12)
    # mix 2
    pprint_run({g11: 5, g12: 2., g13: 0,
                g21: 2, g22: 5., g23: 0.5}, avg12)
