import os
import os.path as osp
import numpy as np
try:
    from .base_dataset import BaseDataset
    from .registry import DATASETS
except:
    from base_dataset import BaseDataset
    from registry import DATASETS
import clrnet.utils.openlane_metric as openlane_metric
import cv2
from tqdm import tqdm
import logging
import pickle as pkl
import json

LIST_FILE = {
    "train": "list/training.txt",
    "val": "list/validation.txt",
    "test": "list/testing.txt",
}


@DATASETS.register_module
class OpenLane(BaseDataset):
    def __init__(self, data_root, split, processes=None, cfg=None):
        super().__init__(data_root, split, processes=processes, cfg=cfg)
        self.list_path = osp.join(data_root, LIST_FILE[split])
        self.split = split
        self.load_annotations()

    def load_annotations(self):
        self.logger.info("Loading OpenLane annotations...")
        # Waiting for the dataset to load is tedious, let's cache it
        os.makedirs("cache", exist_ok=True)
        cache_path = "cache/openlane_{}.pkl".format(self.split)
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as cache_file:
                self.data_infos = pkl.load(cache_file)
                self.max_lanes = max(len(anno["lanes"]) for anno in self.data_infos)
                return

        self.data_infos = []
        with open(self.list_path) as list_file:
            for line in list_file:
                # line: image relative path to data_root
                infos = self.load_annotation(line.split())
                self.data_infos.append(infos)

        # cache data infos to file
        with open(cache_path, "wb") as cache_file:
            pkl.dump(self.data_infos, cache_file)

    def load_annotation(self, line):
        """
        Args:
            line: list of str, each str is a line in the list file

        Returns:
            infos:{
                img_name: str,
                img_path: str,
                mask_path: str,
                lane_exist: np.array,
                lanes: list of list of tuple (x, y)
            }
        """
        infos = {}
        img_line = line[0]
        img_line = img_line[1 if img_line[0] == "/" else 0 : :]
        img_path = os.path.join(self.data_root, img_line)
        infos["img_name"] = img_line
        infos["img_path"] = img_path

        seg_line = line[1]
        seg_line = seg_line[1 if seg_line[0] == "/" else 0 : :]
        seg_path = os.path.join(self.data_root, seg_line)
        infos["mask_path"] = seg_path

        label_line = line[2]
        label_line = label_line[1 if label_line[0] == "/" else 0 : :]
        anno_path = os.path.join(self.data_root, label_line)
        infos["label_path"] = anno_path

        with open(anno_path, "r") as anno_file:
            data = json.load(anno_file)
            json_lanes = data["lane_lines"]

        lanes = []

        for json_lane in json_lanes:
            json_lane = json_lane["uv"]
            us, vs = json_lane[0], json_lane[1]
            lane = [(u, v) for u, v in zip(us, vs)]
            lanes.append(lane)

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

    def evaluate(self, predictions, output_basedir):
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

        # TODO
        result = openlane_metric.eval_predictions(
            output_basedir,
            self.data_root,
            self.list_path,
            iou_thresholds=np.linspace(0.5, 0.95, 10),
            official=True,
        )

        return result[0.5]["F1"]


if __name__ == "__main__":
    dataset = OpenLane(data_root="./data/openlane", split="train", cfg=None)
    print(len(dataset))