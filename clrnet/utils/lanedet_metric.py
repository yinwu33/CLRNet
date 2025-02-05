import os
import argparse
from functools import partial
from pathlib import Path
import pickle as pkl

import cv2
import numpy as np
from tqdm import tqdm
from p_tqdm import t_map, p_map
from scipy.interpolate import splprep, splev
from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString, Polygon

#! attention for unifed data format only
IMAGE_SHAPE = (720, 1280, 3)  # TODO: configurable 



def draw_lane(lane, img=None, img_shape=None, width=30):
    if img is None:
        img = np.zeros(img_shape, dtype=np.uint8)
    lane = lane.astype(np.int32)
    for p1, p2 in zip(lane[:-1], lane[1:]):
        cv2.line(img, tuple(p1), tuple(p2), color=(255, 255, 255), thickness=width)
    return img


def discrete_max_iou(pred, anno, width=30, img_shape=IMAGE_SHAPE):
    if len(pred) == 0:
        return []

    ious = [0 for _ in range(len(pred))]

    xs = [draw_lane(lane, img_shape=img_shape, width=width) > 0 for lane in pred]
    ys = [draw_lane(lane, img_shape=img_shape, width=width) > 0 for lane in anno]

    for i, x in enumerate(xs):
        max_iou = ious[i]
        for y in ys:
            iou = (x & y).sum() / (x | y).sum()

            max_iou = max(max_iou, iou)

        ious[i] = max_iou

    return ious


def discrete_cross_iou(xs, ys, width=30, img_shape=IMAGE_SHAPE):
    xs = [draw_lane(lane, img_shape=img_shape, width=width) > 0 for lane in xs]
    ys = [draw_lane(lane, img_shape=img_shape, width=width) > 0 for lane in ys]

    ious = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            inter = x & y
            union = x | y
            if union.sum() == 0:
                ious[i, j] = 0
            else:
                ious[i, j] = inter.sum() / union.sum()

    if len(xs) == 0 and len(ys) != 0:
        return ious, 0
    elif len(xs) == 0 and len(ys) == 0:
        return ious, 1
    elif len(xs) != 0 and len(ys) == 0:
        return ious, 0
    else:
        total_x = np.stack(xs).sum(axis=0)
        total_y = np.stack(ys).sum(axis=0)
        intersection = (total_x & total_y).sum()
        union = (total_x | total_y).sum()
        total_iou = intersection / union if union != 0 else 0

        return ious, total_iou


def continuous_cross_iou(xs, ys, width=30, img_shape=IMAGE_SHAPE):
    h, w, _ = img_shape
    image = Polygon([(0, 0), (0, h - 1), (w - 1, h - 1), (w - 1, 0)])
    xs = [
        LineString(lane)
        .buffer(distance=width / 2.0, cap_style=1, join_style=2)
        .intersection(image)
        for lane in xs
    ]
    ys = [
        LineString(lane)
        .buffer(distance=width / 2.0, cap_style=1, join_style=2)
        .intersection(image)
        for lane in ys
    ]

    ious = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            ious[i, j] = x.intersection(y).area / x.union(y).area

    return ious


def interp(points, n=50):
    x = [x for x, _ in points]
    y = [y for _, y in points]
    tck, u = splprep([x, y], s=0, t=n, k=min(3, len(points) - 1))

    u = np.linspace(0.0, 1.0, num=(len(u) - 1) * n + 1)
    return np.array(splev(u, tck)).T


def culane_metric(
    pred,
    anno,
    width=30,
    iou_thresholds=[0.5],
    official=True,
    img_shape=IMAGE_SHAPE,
):
    _metric = {}
    for thr in iou_thresholds:
        tp = 0
        fp = 0 if len(anno) != 0 else len(pred)
        fn = 0 if len(pred) != 0 else len(anno)
        _metric[thr] = [tp, fp, fn]

    interp_pred = np.array(
        [interp(pred_lane, n=5) for pred_lane in pred], dtype=object
    )  # (4, 50, 2)
    interp_anno = np.array(
        [interp(anno_lane, n=5) for anno_lane in anno], dtype=object
    )  # (4, 50, 2)

    if official:
        ious, image_iou = discrete_cross_iou(
            interp_pred, interp_anno, width=width, img_shape=img_shape
        )
    else:
        ious = continuous_cross_iou(
            interp_pred, interp_anno, width=width, img_shape=img_shape
        )

    row_ind, col_ind = linear_sum_assignment(1 - ious)

    _metric = {}
    for thr in iou_thresholds:
        tp = int((ious[row_ind, col_ind] > thr).sum())
        fp = len(pred) - tp
        fn = len(anno) - tp
        _metric[thr] = [tp, fp, fn]

    _metric["ious"] = discrete_max_iou(
        interp_pred, interp_anno, width=width, img_shape=img_shape
    )
    _metric["image_iou"] = image_iou

    return _metric


def load_culane_label_data(label_path):
    with open(label_path, "r") as data_file:
        img_data = data_file.readlines()
    img_data = [line.split() for line in img_data]
    img_data = [list(map(float, lane)) for lane in img_data]
    img_data = [
        [(lane[i], lane[i + 1]) for i in range(0, len(lane), 2)] for lane in img_data
    ]
    img_data = [lane for lane in img_data if len(lane) >= 2]

    return img_data


def load_culane_data(data_dir, file_list_path):
    with open(file_list_path, "r") as f:
        lines = f.readlines()
        # img_paths = []
        label_paths = []

        for line in lines:
            line = line.split()
            # img_paths.append(os.path.join(data_dir, line[0]))
            label_paths.append(os.path.join(data_dir, line[1]))

    data = []
    for label_path in label_paths:
        img_data = load_culane_label_data(label_path)
        data.append(img_data)

    return data


def eval_predictions(
    pred_dir,
    anno_dir,
    list_path,
    iou_thresholds=[0.5],
    width=30,
    official=True,
    sequential=False,
    cfg=None,
):
    import logging

    logger = logging.getLogger(__name__)
    logger.info("Calculating metric for List: {}".format(list_path))
    predictions = load_culane_data(pred_dir, list_path)
    annotations = load_culane_data(anno_dir, list_path)
    if sequential:
        results = map(
            partial(
                culane_metric,
                width=width,
                official=official,
                iou_thresholds=iou_thresholds,
                img_shape=IMAGE_SHAPE,
            ),
            predictions,
            annotations,
        )
    else:
        from multiprocessing import Pool, cpu_count
        from itertools import repeat

        with Pool(cpu_count()) as p:
            results = p.starmap(
                culane_metric,
                zip(
                    predictions,
                    annotations,
                    repeat(width),
                    repeat(iou_thresholds),
                    repeat(official),
                    repeat(IMAGE_SHAPE),
                ),
            )

    # ! save the results
    save_path = Path(list_path).stem + "_results.pkl"
    if cfg is not None:
        save_path = Path(cfg.work_dir) / save_path

    with open(save_path, "wb") as f:
        pkl.dump(results, f)

    # ! end

    mean_f1, mean_prec, mean_recall, total_tp, total_fp, total_fn = 0, 0, 0, 0, 0, 0
    ret = {}
    for thr in iou_thresholds:
        tp = sum(m[thr][0] for m in results)
        fp = sum(m[thr][1] for m in results)
        fn = sum(m[thr][2] for m in results)
        precision = float(tp) / (tp + fp) if tp != 0 else 0
        recall = float(tp) / (tp + fn) if tp != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if tp != 0 else 0
        logger.info(
            "iou thr: {:.2f}, tp: {}, fp: {}, fn: {},"
            "precision: {}, recall: {}, f1: {}".format(
                thr, tp, fp, fn, precision, recall, f1
            )
        )
        mean_f1 += f1 / len(iou_thresholds)
        mean_prec += precision / len(iou_thresholds)
        mean_recall += recall / len(iou_thresholds)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        ret[thr] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        }
    if len(iou_thresholds) > 2:
        logger.info(
            "mean result, total_tp: {}, total_fp: {}, total_fn: {},"
            "precision: {}, recall: {}, f1: {}".format(
                total_tp, total_fp, total_fn, mean_prec, mean_recall, mean_f1
            )
        )
        ret["mean"] = {
            "TP": total_tp,
            "FP": total_fp,
            "FN": total_fn,
            "Precision": mean_prec,
            "Recall": mean_recall,
            "F1": mean_f1,
        }
    return ret


def main():
    args = parse_args()
    for list_path in args.list:
        results = eval_predictions(
            args.pred_dir,
            args.anno_dir,
            list_path,
            width=args.width,
            official=args.official,
            sequential=args.sequential,
        )

        header = (
            "=" * 20 + " Results ({})".format(os.path.basename(list_path)) + "=" * 20
        )
        print(header)
        for metric, value in results.items():
            if isinstance(value, float):
                print("{}: {:.4f}".format(metric, value))
            else:
                print("{}: {}".format(metric, value))
        print("=" * len(header))


def parse_args():
    parser = argparse.ArgumentParser(description="Measure CULane's metric")
    parser.add_argument(
        "--pred_dir",
        help="Path to directory containing the predicted lanes",
        required=True,
    )
    parser.add_argument(
        "--anno_dir",
        help="Path to directory containing the annotated lanes",
        required=True,
    )
    parser.add_argument("--width", type=int, default=30, help="Width of the lane")
    parser.add_argument(
        "--list",
        nargs="+",
        help="Path to txt file containing the list of files",
        required=True,
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run sequentially instead of in parallel",
    )
    parser.add_argument(
        "--official",
        action="store_true",
        help="Use official way to calculate the metric",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
