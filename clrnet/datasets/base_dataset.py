import os.path as osp
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision
import logging

try:
    from .registry import DATASETS
    from .process import Process
except:
    from registry import DATASETS
    from process import Process
from clrnet.utils.visualization import imshow_lanes
from mmcv.parallel import DataContainer as DC


def resize_lanes(lanes, from_size, to_size, format="wh"):
    """
    Args:
        lanes (list of numpy arrayes): arr: [N, 2]
        from_size: (w, h)
        to_size: (w, h)

    Returns:
        _type_: _description_
    """
    if format != "wh":
        from_size = from_size[::-1]
        to_size = to_size[::-1]

    resized_lanes = []
    for lane in lanes:
        # lane: arr, [N, 2]
        xs = lane[:, 0]
        ys = lane[:, 1]

        resized_xs = xs * to_size[0] / from_size[0]
        resized_ys = ys * to_size[1] / from_size[1]

        resized_lane = np.stack([resized_xs, resized_ys], axis=1)
        resized_lanes.append(resized_lane)

    return resized_lanes


@DATASETS.register_module
class BaseDataset(Dataset):
    def __init__(
        self,
        data_root,
        split,
        ori_img_size,
        sample_y,
        processes=None,
        is_training=False,
        cfg=None,
    ):
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)
        self.data_root = data_root
        self.split = split
        # ! be used to restore graph
        self.ori_img_size = ori_img_size  # (h, w)
        self.ori_img_h = ori_img_size[0]
        self.ori_img_w = ori_img_size[1]
        self.sample_y = sample_y
        self.training = is_training  # sometime use "train" for val
        self.processes = Process(processes, cfg)

    # ! debug
    # def resize_lanes(self, lanes):
    #     return resize_lanes(
    #         lanes,
    #         from_size=(self.cfg.ori_img_h, self.cfg.ori_img_w),  # ! model img size
    #         to_size=self.ori_img_size,
    #         format="hw",
    #     )

    def view(self, predictions, img_metas):
        # a batch of data
        img_metas = [item for img_meta in img_metas.data for item in img_meta]
        for lanes, img_meta in zip(predictions, img_metas):
            img_name = img_meta["img_name"]
            img = cv2.imread(osp.join(self.data_root, img_name))
            out_file = osp.join(
                self.cfg.work_dir, "visualization", img_name.replace("/", "_")
            )

            lanes = [lane.to_array(self.sample_y, self.ori_img_size) for lane in lanes]

            imshow_lanes(img, lanes, out_file=out_file)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        data_info = self.data_infos[idx]
        img = cv2.imread(data_info["img_path"])
        # ! resize to model input size
        img = cv2.resize(img, (self.cfg.ori_img_w, self.cfg.ori_img_h))
        # ! resize end
        img = img[self.cfg.cut_height :, :, :]
        sample = data_info.copy()
        sample.update({"img": img})

        if self.training:
            label = cv2.imread(sample["mask_path"], cv2.IMREAD_UNCHANGED)
            # label = cv2.resize(label, (self.cfg.ori_img_w, self.cfg.ori_img_h))
            if (
                len(label.shape) > 2
            ):  # when label is [h, w, c], means not grayscale mask
                label = label[
                    :, :, 0
                ]  # only use one channel, label: [h, w, c] -> [h, w, 1]
            label = label.squeeze()  # [h, w, 1] -> [h, w]
            label = label[self.cfg.cut_height :, :]
            sample.update({"mask": label})

            if self.cfg.cut_height != 0:
                new_lanes = []
                for i in sample["lanes"]:
                    lanes = []
                    for p in i:
                        lanes.append((p[0], p[1] - self.cfg.cut_height))
                    new_lanes.append(lanes)
                sample.update({"lanes": new_lanes})

        sample = self.processes(sample)
        meta = {
            "full_img_path": data_info["img_path"],
            "img_name": data_info["img_name"],
        }
        if self.training:
            meta.update({"mask_path": data_info["mask_path"]})
        meta = DC(meta, cpu_only=True)
        sample.update({"meta": meta})

        return sample
