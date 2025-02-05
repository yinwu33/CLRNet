import os.path as osp
import os
import numpy as np
import cv2
import torch
import torchvision
import logging
from pathlib import Path

try:
    from .registry import DATASETS
    from .process import Process
    from .base_dataset import BaseDataset
except:
    from registry import DATASETS
    from process import Process
    from base_dataset import BaseDataset
from clrnet.utils.visualization import imshow_lanes
from mmcv.parallel import DataContainer as DC


@DATASETS.register_module
class DemoDataset(BaseDataset):
    def __init__(
        self, data_root, ori_img_size, sample_y=None, processes=None, cfg=None, **kwargs
    ):
        if sample_y == None:
            sample_y = range(cfg.ori_img_h, cfg.cut_height, -20)

        super().__init__(
            data_root=data_root,
            split="test",
            ori_img_size=ori_img_size,
            sample_y=sample_y,
            processes=processes,
            cfg=cfg,
        )

        self.load_annotations()

    def load_annotations(self):
        # grab all png from self.data_root
        folder = Path(self.data_root)

        self.data_infos = []
        for img_path in folder.glob("*.png"):
            img_name = img_path.name
            self.data_infos.append(
                {
                    "img_path": str(img_path),
                    "img_name": img_name,
                    "lanes": [],
                    "relative_path": img_path.relative_to(self.data_root),
                }
            )

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        data_info = self.data_infos[idx]
        img = cv2.imread(data_info["img_path"])
        img = cv2.resize(img, (self.cfg.ori_img_w, self.cfg.ori_img_h))
        img = img[self.cfg.cut_height :, :, :]
        sample = data_info.copy()
        sample.update({"img": img})

        sample = self.processes(sample)
        meta = {
            "full_img_path": data_info["img_path"],
            "img_name": data_info["img_name"],
        }

        meta = DC(meta, cpu_only=True)
        sample.update({"meta": meta})

        return sample
    
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
            
            out_lane_file = osp.join(
                self.cfg.work_dir, "out", img_name.replace("/", "_")
            )
            
            blank_image = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
            imshow_lanes(blank_image, lanes, out_file=out_lane_file, width=15)


    def evaluate(self, *args, **kwargs):
        return None
