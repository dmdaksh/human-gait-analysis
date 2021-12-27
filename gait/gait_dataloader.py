import torch


class GaitDataLoader:
    def __init__(self,
                 *tensors,
                 batch_size=32,
                 shuffle=False,
                 drop_last=False):
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if not self.drop_last and remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.n_batches:
            raise StopIteration

        batch = tuple(t[self.i * self.batch_size:(self.i + 1) *
                        self.batch_size] for t in self.tensors)
        self.i += 1
        return batch

    def __len__(self):
        return self.n_batches
