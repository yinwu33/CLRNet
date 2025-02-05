import os
import os.path as osp
import numpy as np
from .base_dataset import BaseDataset
from .registry import DATASETS
import clrnet.utils.culane_metric as culane_metric
import cv2
from tqdm import tqdm
import logging
import pickle as pkl

LIST_FILE = {
    # "train": "list/train_gt.txt",
    "train": "list/train.txt",  # ! TODO debug, temp for val
    "val": "list/val.txt",
    "test": "list/test.txt",
}



@DATASETS.register_module
class TuSimpleRe(BaseDataset):
    def __init__(
        self,
        data_root,
        split,
        ori_img_size,
        sample_y=range(710, 150, -10),
        processes=None,
        cfg=None,
    ):
        super().__init__(
            data_root,
            split,
            ori_img_size,
            sample_y=sample_y,
            processes=processes,
            cfg=cfg,
        )
        self.list_path = osp.join(data_root, LIST_FILE[split])
        self.split = split
        self.load_annotations()

    def load_annotations(self):
        self.logger.info("Loading CULane annotations...")
        # Waiting for the dataset to load is tedious, let's cache it
        os.makedirs("cache", exist_ok=True)
        cache_path = "cache/tusimplere_{}.pkl".format(self.split)
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as cache_file:
                self.data_infos = pkl.load(cache_file)
                self.max_lanes = max(len(anno["lanes"]) for anno in self.data_infos)
                return

        self.data_infos = []
        with open(self.list_path) as list_file:
            for line in list_file:
                infos = self.load_annotation(line.split())
                self.data_infos.append(infos)

        # cache data infos to file
        with open(cache_path, "wb") as cache_file:
            pkl.dump(self.data_infos, cache_file)

    def load_annotation(self, line):
        infos = {}
        img_line = line[0]
        img_line = img_line[1 if img_line[0] == "/" else 0 : :]
        img_path = os.path.join(self.data_root, img_line)
        infos["img_name"] = img_line
        infos["img_path"] = img_path
        if len(line) > 1:
            mask_line = line[1]
            mask_line = mask_line[1 if mask_line[0] == "/" else 0 : :]
            mask_path = os.path.join(self.data_root, mask_line)
            infos["mask_path"] = mask_path

        if len(line) > 2:
            exist_list = [int(l) for l in line[2:]]
            infos["lane_exist"] = np.array(exist_list)

        anno_path = img_path[:-3] + "lines.txt"  # remove sufix jpg and add lines.txt
        with open(anno_path, "r") as anno_file:
            data = [list(map(float, line.split())) for line in anno_file.readlines()]
        lanes = [
            [
                (lane[i], lane[i + 1])
                for i in range(0, len(lane), 2)
                if lane[i] >= 0 and lane[i + 1] >= 0
            ]
            for lane in data
        ]
        lanes = [list(set(lane)) for lane in lanes]  # remove duplicated points
        lanes = [
            lane for lane in lanes if len(lane) > 2
        ]  # remove lanes with less than 2 points

        lanes = [sorted(lane, key=lambda x: x[1]) for lane in lanes]  # sort by y
        infos["lanes"] = lanes

        return infos

    def get_prediction_string(self, pred):
        ys = np.arange(270, 590, 8) / self.ori_img_h
        out = []
        for lane in pred:
            xs = lane(ys)
            valid_mask = (xs >= 0) & (xs < 1)
            xs = xs * self.ori_img_w
            lane_xs = xs[valid_mask]
            lane_ys = ys[valid_mask] * self.ori_img_h
            lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
            lane_str = " ".join(
                ["{:.5f} {:.5f}".format(x, y) for x, y in zip(lane_xs, lane_ys)]
            )
            if lane_str != "":
                out.append(lane_str)

        return "\n".join(out)

    def get_prediction_arr(self, pred):
        ys = np.arange(270, 590, 8) / self.ori_img_h
        out = []
        for lane in pred:
            xs = lane(ys)
            valid_mask = (xs >= 0) & (xs < 1)
            xs = xs * self.ori_img_w
            lane_xs = xs[valid_mask]
            lane_ys = ys[valid_mask] * self.ori_img_h
            lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]  # list of x, y

            # to [N, 2]
            assert len(lane_xs) == len(lane_ys)
            lane = np.concatenate(
                (lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)), axis=1
            )

            out.append(lane)

        return out

    def get_lane_arr(self, lane):
        ys = np.arange(270, 590, 8) / self.ori_img_h
        xs = lane(ys)
        valid_mask = (xs >= 0) & (xs < 1)
        xs = xs * self.ori_img_w
        lane_xs = xs[valid_mask]
        lane_ys = ys[valid_mask] * self.ori_img_h
        lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]

        assert len(lane_xs) == len(lane_ys)
        lane = np.concatenate((lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)), axis=1)

        return lane

    def evaluate(self, predictions, output_basedir):
        """
        Args:
            predictions (list of list of Lane): 1. list is samples of one dataset, 2. list is lanes in one sample
            pred_meta (list of meta): each meta is a dict with "full_img_path" and "img_name"
            output_basedir (str): output workdir

        Returns:
            _type_: _description_ #TODO
        """
        loss_lines = [[], [], [], []]
        print("Generating prediction output...")
        for idx, pred in enumerate(predictions):
            output_dir = os.path.join(
                output_basedir, os.path.dirname(self.data_infos[idx]["img_name"])
            )
            output_filename = (
                os.path.basename(self.data_infos[idx]["img_name"])[:-3] + "lines.txt"
            )
            os.makedirs(output_dir, exist_ok=True)
            output = self.get_prediction_string(pred)

            with open(os.path.join(output_dir, output_filename), "w") as out_file:
                out_file.write(output)

        # for cate, cate_file in CATEGORYS.items():
        #     result = culane_metric.eval_predictions(
        #         output_basedir,
        #         self.data_root,
        #         os.path.join(self.data_root, cate_file),
        #         iou_thresholds=[0.5],
        #         official=True,
        #         cfg=self.cfg,
        #     )

        result = culane_metric.eval_predictions(
            output_basedir,
            self.data_root,
            self.list_path,
            iou_thresholds=np.linspace(0.5, 0.95, 10),
            official=True,
            cfg=self.cfg,
        )

        return result[0.5]["F1"]
