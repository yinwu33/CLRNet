from pathlib import Path
import os
import json
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

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


def extract_datainfos(tusimple_path: str, split="train"):
    tusimple_path = Path(tusimple_path)

    datainfos = []

    anno_files = SPLIT_FILES[split]

    for anno_file in anno_files:
        anno_file = tusimple_path / anno_file
        with open(anno_file, "r") as anno_obj:
            lines = anno_obj.readlines()
        for line in lines:
            # data is a dict
            # {lanes, h_samples, raw_file}
            data = json.loads(line)

            raw_file = data["raw_file"]
            if "test_set/" in raw_file:
                print(f"Skipping {raw_file}")
                continue
            raw_file = tusimple_path / raw_file

            if not raw_file.exists():
                print(f"File {raw_file} does not exist")
                continue

            h_samples = data["h_samples"]
            lanes_xs = data["lanes"]
            lanes_xs = np.array(lanes_xs, dtype=np.float32)

            lanes = [
                [(x, y) for (x, y) in zip(lane_xs, h_samples) if x >= 0]
                for lane_xs in lanes_xs
            ]

            lanes = [lane for lane in lanes if len(lane) > 0]  # remove empty lanes

            datainfos.append(
                dict(
                    img_path=raw_file,  # absolute path
                    lanes=lanes,  # [N, 2]
                )
            )

    print(f"Extracted {len(datainfos)} data infos")
    return datainfos


def align_format(datainfos, center_y=1280 / 2):
    # since culane max_lanes = 4, tusimple max_lanes = 5
    # we need to remove the one lane
    # remove the lane farthest from the center
    # and reorder the lanes by x from small to large

    for data in datainfos:
        lanes = data["lanes"]
        if len(lanes) > 4:
            # find the lane farthest from the center
            lane_ys = [np.mean(np.array(lane)[:, 1]) for lane in lanes]
            lane_ys = np.array(lane_ys)
            lane_idx = np.argmax(np.abs(lane_ys - center_y))
            lanes.pop(lane_idx)

            # reorder
            lane_ys = [np.mean(np.array(lane)[:, 1]) for lane in lanes]
            lane_ys = np.array(lane_ys)
            lane_idx = np.argsort(lane_ys)
            lanes = [lanes[i] for i in lane_idx]

    return datainfos


def get_seg_mask(lanes, img_size=(720, 1280), width=30):
    # lanes: [N, 2]
    # 0 is background
    # 1, 2, 3, 4 are lanes from left to right
    seg = np.zeros((*img_size, 3), dtype=np.uint8)
    for i, lane in enumerate(lanes):
        for j in range(len(lane) - 1):
            cv2.line(
                seg,
                (round(lane[j][0]), lane[j][1]),
                (round(lane[j + 1][0]), lane[j + 1][1]),
                (i + 1, i + 1, i + 1),
                # (255, 255, 255),  # ! for testing
                thickness=width,
            )
    return seg


def save(datainfos, out_root: str, split="train"):
    out_root = Path(out_root)
    list_dir = out_root / "list"
    split_dir = out_root / split
    img_dir = split_dir / "image"
    label_dir = split_dir / "label"
    seg_dir = split_dir / "seg_label"
    for d in [list_dir, img_dir, label_dir, seg_dir]:
        if not d.exists():
            os.makedirs(d, exist_ok=True)

    list_text = ""

    # check if same img_name exists
    # img_names = [data["img_path"].parent.name for data in datainfos]
    # assert len(img_names) == len(set(img_names)), "Duplicate image names"

    for i, data in tqdm(enumerate(datainfos), total=len(datainfos)):

        # copy image
        img_path = img_dir / f"{i}.jpg"
        img = Image.open(data["img_path"])
        img.save(img_path)

        # save lanes
        lanes = data["lanes"]  # [N, 2]
        label_path = label_dir / f"{i}.lines.txt"
        with open(label_path, "w") as f:
            for lane in lanes:
                for x, y in lane:
                    f.write(f"{x} {y} ")
                f.write("\n")

        # save seg mask
        seg_path = seg_dir / f"{i}.png"
        seg = get_seg_mask(lanes)
        cv2.imwrite(str(seg_path), seg)

        # get rel path
        img_path = img_path.relative_to(out_root)
        label_path = label_path.relative_to(out_root)
        seg_path = seg_path.relative_to(out_root)

        # append image relative path to the out_dir
        list_text += f"{img_path} {label_path} {seg_path}\n"

    with open(list_dir / f"{split}.txt", "w") as f:
        f.write(list_text)


if __name__ == "__main__":
    split = "val"
    tusimple_root = "./data/tusimple"
    out_root = "./data/tusimple_re"
    datainfos = extract_datainfos(tusimple_root, split=split)
    align_format(datainfos)
    save(datainfos, out_root=out_root, split=split)
