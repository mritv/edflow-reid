import numpy as np
import tensorflow as tf
from PIL import Image
from edflow.iterators.resize import resize_float32

TRIP_W = 256
TRIP_H = 256


def initialize_model(model, checkpoint, session=None):
    """
    Loads weights from a checkpointfile and initializes the model.
    This function is just for the case of restoring the market-1501 pretrained
    model because we have to map variable names correctly. For newly written
    checkpoints use the RestoreCheckpointHook.
    """

    sess = session or tf.Session()
    sess.run(tf.global_variables_initializer())

    if checkpoint is None:
        raise ValueError(
            "The reIdEvaluator needs a checkpoint from which "
            "to initialize the model."
        )

    var_map = {}
    for v in model.variables:
        vn = v.name.strip(model.model_name).strip("/").strip(":0")
        var_map[vn] = v

    tf.train.Saver(var_map).restore(sess, checkpoint)


def resize(image, w=TRIP_W, h=TRIP_H):
    ims = []
    for im in image:
        if w == h // 2 and im.shape[0] == im.shape[1]:
            im = resize_float32(im, h)
            im = im[:, h // 4 : h // 4 + w, :]
        else:
            im = Image.fromarray(im, mode="RGB").resize((w, h))
            im = np.array(im)
        ims += [im]
    ims = np.array(ims)

    return ims

