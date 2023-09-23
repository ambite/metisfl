from metisfl.model.model_dataset import ModelDataset

import tensorflow as tf

def construct_dataset_pipeline(dataset: ModelDataset, batch_size, is_train=False):
    _x, _y = dataset.get_x(), dataset.get_y()
    if isinstance(_x, tf.data.Dataset):
        if is_train:
            # Shuffle all records only if dataset is used for training.
            _x = _x.shuffle(dataset.get_size())
        # If the input is of tf.Dataset we only need to return the input x,
        # we do not need to set a value for target y.
        _x, _y = _x.batch(batch_size), None
    return _x, _y
