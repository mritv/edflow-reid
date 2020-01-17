import os

import numpy as np
import tensorflow as tf
from edflow.data.util import adjust_support
from edflow.iterators.batches import make_batches
from edflow.util import retrieve
from tqdm import tqdm

from edflow_reid.reid_callback.reID_model import reIdModel
from edflow_reid.reid_callback.utils import resize, initialize_model

TRIP_CHECK = os.environ.get("REID_CHECKPOINT", "checkpoint/checkpoint-25000")

TRIP_W = 256
TRIP_H = 256


def get_embedding(model, session, image: np.ndarray):
    """
    Embeds the given image as array using the market dataset.
    :param image:
    :return: embedding
    """

    if image.shape[-1] == 4:
        image = image[..., :3]
    fetches = model.outputs
    feeds = {model.inputs["image"]: resize(image)}

    return session.run(fetches, feed_dict=feeds)["embeddings"]


def run_through_dataset(
    dataset, support: str = "-1->1", batch_size: int = 50, image_key: str = "image"
):
    """

    :param dataset: DatasetMixin which contains the images.
    :param support: Support of images. One of '-1->1', '0->1' or '0->255'
    :param batch_size: The images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the disposable hardware.
    :param image_key: Dataset key containing the image to be embedded
    :return: np.ndarray embedding.
    """

    dataset_length = len(dataset)
    if batch_size > dataset_length:
        print(Warning("Setting batch size to length of the dataset.."))
        batch_size = dataset_length

    batches = make_batches(dataset, batch_size, shuffle=False)
    n_batches = len(batches)
    n_used_imgs = n_batches * batch_size
    embeddings = [] # np.empty((n_used_imgs, 128))

    sess_config = tf.ConfigProto()

    sess_config.gpu_options.allow_growth = True

    session = tf.Session(config=sess_config)

    model = reIdModel()
    if os.path.basename(TRIP_CHECK) == "checkpoint-25000":
        initialize_model(model, TRIP_CHECK, session)

    for i, batch in enumerate(tqdm(batches, desc="reID")):
        if i >= n_batches:
            break
        images = retrieve(batch, image_key)
        images = adjust_support(
            np.array(images),
            future_support="0->255",
            current_support=support,
            clip=True,
        )
        images = images.astype(np.float32)[..., :3]

        batch_embeddings = get_embedding(model, session=session, image=images)["emb"]
        embeddings += [batch_embeddings.reshape(batch_size, -1)]
    return np.array(embeddings)



if __name__ == "__main__":
    from abc_interpolation.data.human_gait import HumanGait_abc

    hg_data = HumanGait_abc({"data_split": "test"})
    run_through_dataset(hg_data, batch_size=10, image_key="frame_anchor_1")
