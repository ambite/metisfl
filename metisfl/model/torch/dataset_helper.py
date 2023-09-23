
from metisfl.model.model_dataset import ModelDataset
from metisfl.common.logger import MetisLogger


def construct_dataset_pipeline(dataset: ModelDataset):
    _x = dataset.get_x()
    _y = dataset.get_y()
    if _x and _y:
        return _x, _y
    elif _x:
        return _x
    else:
        MetisLogger.warning("Not a well-formatted input dataset: {}, {}".format(_x, _y))
        return None
