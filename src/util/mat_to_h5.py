import h5py


def _h5py_dataset_iterator(g, prefix=''):
    for key in g.keys():
        item = g[key]
        path = '{}/{}'.format(prefix, key)
        if isinstance(item, h5py.Dataset):
            yield (path, item)
        elif isinstance(item, h5py.Group) and '#' not in key:
            yield from _h5py_dataset_iterator(item, path)


def get_mat_data(filename: str) -> dict:
    '''Incomplete function'''
    with h5py.File(filename, 'r') as f:
        for (path, dataset) in _h5py_dataset_iterator(f, 'data'):
            print(path, dataset)
    return {}
