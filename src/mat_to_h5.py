import os
import h5py

def h5py_dataset_iterator(g, prefix=''):
    for key in g.keys():
        item = g[key]
        path = '{}/{}'.format(prefix, key)
        if isinstance(item, h5py.Dataset):
            yield (path, item)
        elif isinstance(item, h5py.Group) and '#' not in key:
            yield from h5py_dataset_iterator(item, path)


with h5py.File(os.environ.get('MAT_FILE'), 'r') as f:
    for (path, dataset) in h5py_dataset_iterator(f, 'data'):
        print(path, dataset)
