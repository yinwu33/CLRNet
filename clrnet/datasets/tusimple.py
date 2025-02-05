import os.path as osp
import numpy as np
import cv2
import os
import json
import torchvision
from .base_dataset import BaseDataset
from clrnet.utils.tusimple_metric import LaneEval
from .registry import DATASETS
import logging
import random

SPLIT_FILES = {
    "trainval": [
        "label_data_0313.json",
        "label_data_0601.json",
        "label_data_0531.json",
    ],
    "train": ["label_data_0313.json", "label_data_0601.json"],
    "val": ["label_data_0531.json"],
    "test": ["test_label.json"],
}


@DATASETS.register_module
class TuSimple(BaseDataset):
    def __init__(
        self,
        data_root,
        split,
        ori_img_size,  # !
        sample_y=range(710, 150, -10),  # !
        processes=None,
        cfg=None,
    ):
        super().__init__(
            data_root,
            split,
            ori_img_size,
            sample_y,
            processes,
            cfg,
        )
        self.anno_files = SPLIT_FILES[split]
        self.load_annotations()
        self.h_samples = list(range(160, 720, 10))

    def load_annotations(self):
        self.logger.info("Loading TuSimple annotations...")
        self.data_infos = []
        max_lanes = 0
        for anno_file in self.anno_files:
            anno_file = osp.join(self.data_root, anno_file)
            with open(anno_file, "r") as anno_obj:
                lines = anno_obj.readlines()
            for line in lines:
                data = json.loads(line)

                raw_file = data["raw_file"]
                if "test_set/" in raw_file:
                    raw_file = raw_file.replace("test_set/", "")

                y_samples = data["h_samples"]
                gt_lanes = data["lanes"]
                mask_path = raw_file.replace("clips", "seg_label")[:-3] + "png"

                lanes = [
                    [(x, y) for (x, y) in zip(lane, y_samples) if x >= 0]
                    for lane in gt_lanes
                ]
                lanes = [lane for lane in lanes if len(lane) > 0]
                max_lanes = max(max_lanes, len(lanes))
                self.data_infos.append(
                    {
                        "img_path": osp.join(self.data_root, raw_file),
                        "img_name": data["raw_file"],
                        "mask_path": osp.join(self.data_root, mask_path),
                        "lanes": lanes,
                    }
                )

        if self.training:
            random.shuffle(self.data_infos)
        self.max_lanes = max_lanes

    def pred2lanes(self, pred):
        ys = np.array(self.h_samples) / self.cfg.ori_img_h
        lanes = []
        for lane in pred:
            xs = lane(ys)
            invalid_mask = xs < 0
            lane = (xs * self.cfg.ori_img_w).astype(int)
            lane[invalid_mask] = -2
            lanes.append(lane.tolist())

        return lanes

    def pred2tusimpleformat(self, idx, pred, runtime):
        runtime *= 1000.0  # s to ms
        img_name = self.data_infos[idx]["img_name"]
        lanes = self.pred2lanes(pred)
        output = {"raw_file": img_name, "lanes": lanes, "run_time": runtime}
        return json.dumps(output)

    def save_tusimple_predictions(self, predictions, filename, runtimes=None):
        if runtimes is None:
            runtimes = np.ones(len(predictions)) * 1.0e-3
        lines = []
        for idx, (prediction, runtime) in enumerate(zip(predictions, runtimes)):
            line = self.pred2tusimpleformat(idx, prediction, runtime)
            lines.append(line)
        with open(filename, "w") as output_file:
            output_file.write("\n".join(lines))

    def evaluate(self, predictions, output_basedir, runtimes=None):
        pred_filename = os.path.join(output_basedir, "tusimple_predictions.json")
        self.save_tusimple_predictions(predictions, pred_filename, runtimes)
        result, acc = LaneEval.bench_one_submit(pred_filename, self.cfg.test_json_file)
        self.logger.info(result)
        return acc

    def get_prediction_arr(self, pred):
        ys = np.arange(710, 150, -10) / self.ori_img_h
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
