from edflow.data.dataset_mixin import DatasetMixin
import numpy as np
from edflow_reid.reid_callback.reid_callback import get_reid_score


class TestDataset(DatasetMixin):
    def __init__(self):
        self.labels = self.create_labels()
        self.data = self.create_data(self.labels)

    def get_example(self, idx: int) -> dict:
        example = self.data[idx]
        return example

    def __len__(self) -> int:
        return len(self.data)

    @staticmethod
    def create_labels() -> dict:
        labels = dict()
        labels["data"] = np.zeros((10, 256, 256, 3))
        labels["keypoints"] = np.ones((10, 2))
        return labels

    @staticmethod
    def create_data(labels: dict) -> np.ndarray:
        keys = list(labels.keys())
        data = [{}] * len(labels[keys[0]])
        for i, dictionary in enumerate(data):
            for key in keys:
                dictionary[key] = labels[key][i]
        return np.array(data)


def test_reid_callback():
    test_score = get_reid_score(
        root,
        data_in,
        data_out,
        config,
        im_in_key="data",
        im_out_key="data",
        name="reid",
    )


if __name__ == "__main__":
    data = TestDataset()
    print(data[0])
