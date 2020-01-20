import os
from importlib import import_module

import tensorflow as tf

TRIP_CHECK = os.environ.get("REID_CHECKPOINT", "checkpoint/checkpoint-25000")

TRIP_W = 256
TRIP_H = 256


class reIdModel(object):
    """ReID model as from the paper "In defense of the triplet loss\""""
    @property
    def name(self):
        return "triplet_reid"

    def __init__(
        self,
        model_name="triplet_reid",
        imported_model_name="resnet_v1_50",
        imported_head_name="fc1024",
        w=TRIP_W,
        h=TRIP_H,
        nc=3,
        edim=128,
        is_train=False,

    ):
        """Args:
            imported_model_name (str): Which base model to use.
            imported_head_name (str)L Which output head to use.
            w (int): Image width.
            h (int): Image height.
            nc (int): Image channels.A
            edim (int): Embedding dimension.
            is_train (bool): Is it training or not?
        """
        model = import_module(
            "edflow_reid.triplet_reid.triplet_reid.nets." + imported_model_name
        )
        head = import_module("edflow_reid.triplet_reid.triplet_reid.heads." + imported_head_name)

        self.model_name = model_name

        with tf.variable_scope(self.model_name):
            self.images = tf.placeholder(tf.float32, shape=[None, h, w, nc])
            input_images = self.images
            endpoints, body_prefix = model.endpoints(
                input_images, is_training=is_train, prefix=self.model_name + "/"
            )
            with tf.name_scope("head"):
                endpoints = head.head(endpoints, edim, is_training=is_train)

        self.embeddings = endpoints

        globs = tf.global_variables()
        self.variables = [v for v in globs if self.model_name in v.name]

    @property
    def inputs(self):
        # (bs, h, w, c) in [-1, 1]
        return {"image": self.images}

    @property
    def outputs(self):
        # (bs, 128
        return {"embeddings": self.embeddings}
