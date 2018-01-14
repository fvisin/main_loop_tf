from pprint import pprint
import sys

import numpy as np

from main_loop_tf.main import zip_longest
from main_loop_tf.utils import split_in_chunks

bs = 10
hw = 3

minibatch = {
    'data': np.array([np.ones((hw, hw)) * i for i in range(bs)]),
    'labels': np.array([np.ones((hw, hw)) * 10 * i for i in range(bs)])}
sys.stdout.write('Data:\n')
pprint(minibatch['data'])
sys.stdout.write('Labels:\n')
pprint(minibatch['labels'])


def test(nsplits):
    print('##### NSPLIT: %d' % nsplits)
    ret = split_in_chunks(minibatch, nsplits)
    assert len(ret) == nsplits
    for i in range(nsplits):
        print('## Split #%d:' % i)
        sys.stdout.write('Data:\n')
        pprint(ret[i]['data'])
        sys.stdout.write('Labels:\n')
        pprint(ret[i]['labels'])
    print('\n\n')


test(5)
test(3)


feed_dict = {}
placeholders = [{'data': 'datap1', 'labels': 'labelsp1'},
                {'data': 'datap2', 'labels': 'labelsp2'},
                {'data': 'datap3', 'labels': 'labelsp3'}]
ret = split_in_chunks(minibatch, 2)
for p_dict, batch_dict in zip_longest(placeholders,
                                      ret,
                                      fillvalue=ret[0]):
    for p_name, p_obj in p_dict.iteritems():
        feed_dict[p_obj] = batch_dict[p_name]

pprint(feed_dict)
