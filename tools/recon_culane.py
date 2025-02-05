from pathlib import Path
import os
import json
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm


SPLIT_FILES = {
    "train": "list/train.txt",
    "val": "list/val.txt",
    "test": "list/test.txt",
}

original_size = (590, 1640)
target_size = (720, 1280)


def extract_datainfos(culane_path: str, split: str):
    culane_path = Path(culane_path)

    datainfos = []
    list_file = culane_path / SPLIT_FILES[split]

    with open(list_file, "r") as f:
        lines = f.readlines()

    print("Extracting data infos...")
    for img_rel_path in tqdm(lines, total=len(lines)):
        # each line is a relative path of an image
        if img_rel_path.startswith("/"):
            img_rel_path = img_rel_path[1:]
        img_path = culane_path / img_rel_path.strip()
        label_path = img_path.with_suffix(".lines.txt")

        with open(label_path, "r") as f:
            lines = f.readlines()
        # in format x1, y1, x2, y2, ...
        lanes = [list(map(float, line.split())) for line in lines]
        lanes = [
            [
                (lane[i], lane[i + 1])
                for i in range(0, len(lane), 2)
                if lane[i] >= 0 and lane[i + 1] >= 0
            ]
            for lane in lanes
        ]
        lanes = [list(set(lane)) for lane in lanes]  # remove duplicated points
        lanes = [
            lane for lane in lanes if len(lane) > 2
        ]  # remove lanes with less than 2 points

        lanes = [sorted(lane, key=lambda x: x[1]) for lane in lanes]  # sort by y

        datainfos.append(
            dict(
                img_path=img_path,  # absolute path
                lanes=lanes,
            )
        )

    print()
    return datainfos


def align_format(datainfos):
    # map the keypoints from (590, 1640) to (720, 1280)
    print("Aligning format...")
    for data in tqdm(datainfos, total=len(datainfos)):
        for lane in data["lanes"]:
            for i, (x, y) in enumerate(lane):
                x = x / original_size[1] * target_size[1]
                y = y / original_size[0] * target_size[0]
                lane[i] = (x, y)

    return datainfos


def get_seg_mask(lanes, img_size=target_size, width=30):
    seg = np.zeros((*img_size, 3), dtype=np.uint8)

    for i, lane in enumerate(lanes):
        for j in range(len(lane) - 1):
            cv2.line(
                seg,
                (round(lane[j][0]), round(lane[j][1])),
                (round(lane[j + 1][0]), round(lane[j + 1][1])),
                (i + 1, i + 1, i + 1),
                # (255, 255, 255),  # ! for testing
                thickness=width,
            )
    return seg


def save(datainfos, out_root, split):
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

    print("Saving data...")
    for i, data in tqdm(enumerate(datainfos), total=len(datainfos)):

        img_path = img_dir / f"{i}.jpg"
        label_path = label_dir / f"{i}.lines.txt"
        seg_path = seg_dir / f"{i}.png"
        
        # copy image
        img = Image.open(data["img_path"])
        img = img.resize(target_size[::-1])
        img.save(img_path)

        # # save lanes
        lanes = data["lanes"]  # [N, 2]
        with open(label_path, "w") as f:
            for lane in lanes:
                for x, y in lane:
                    f.write(f"{x} {y} ")
                f.write("\n")

        # save seg mask
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
    split = "test"
    culane_root = "./data/culane"
    out_root = "./data/culane_re"
    datainfos = extract_datainfos(culane_root, split)
    align_format(datainfos)
    save(datainfos, out_root, split)
