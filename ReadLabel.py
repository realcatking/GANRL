'''
Read labels of nodes
'''

import scipy.io as sio
from six.moves import zip

def read_label(path):
    filetype = path.strip().split('.')[-1]
    if filetype == 'mat':
        mat_list = sio.loadmat(path)
        if 'Label' in mat_list:
            label = mat_list['Label']
        elif 'label' in mat_list:
            label = mat_list['label']
        else:
            raise Exception('unrecognized label name in MATLAB file')
        label_dict = dict(zip(range(label.size),list(label.reshape(-1))))
    else:
        f = open(path,'r')
        items = f.read().strip().split()
        label_dict = dict(zip(items[::2],items[1::2]))
        f.close()

    return label_dict
