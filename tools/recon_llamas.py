from pathlib import Path
import os
import json
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

from llamas_utils import get_horizontal_values_for_four_lanes


SPLIT_FILES = {
    "train": "list/train.txt",
    "val": "list/val.txt",
}

LABEL_DIRS = {
    "train": "labels/train",
    "val": "labels/valid",
}

IMG_ORGINAL_SIZE = (717, 1276)
IMG_TARGET_SIZE = (720, 1280)
WIDTH = 30


def get_img_path(json_path):
    # /foo/bar/test/folder/image_label.ext --> test/folder/image_label.ext
    json_path = str(json_path)
    base_name = "/".join(json_path.split("/")[-3:])
    image_path = os.path.join(
        "color_images", base_name.replace(".json", "_color_rect.png")
    )
    return image_path


def extract_datainfos(llamas_path: str, split: str):
    llamas_path = Path(llamas_path)

    datainfos = []
    list_file = llamas_path / SPLIT_FILES[split]

    label_dir = llamas_path / LABEL_DIRS[split]

    # get all json paths, recursively
    json_paths = list(label_dir.glob("**/*.json"))
    print(f"Found {len(json_paths)} json files")

    for json_path in tqdm(json_paths, total=len(json_paths)):
        lanes = get_horizontal_values_for_four_lanes(json_path)
        lanes = [
            [(x, y) for x, y in zip(lane, range(717)) if x >= 0]
            for lane in lanes
        ]
        lanes = [lane for lane in lanes if len(lane) > 0]
        lanes = [list(set(lane)) for lane in lanes]  # remove duplicated points
        lanes = [
            lane for lane in lanes if len(lane) > 2
        ]  # remove lanes with less than 2 points

        lanes = [sorted(lane, key=lambda x: x[1]) for lane in lanes]  # sort by y
        lanes.sort(key=lambda lane: lane[0][0])

        img_path = get_img_path(json_path)  # relative path
        img_path = llamas_path / img_path

        datainfos.append(
            dict(
                img_path=img_path,  # absolute path
                lanes=lanes,
            )
        )
    return datainfos


def align_format(datainfos):
    # map the keypoints from original size to target size
    print("Aligning format...")
    for data in tqdm(datainfos, total=len(datainfos)):
        for lane in data["lanes"]:
            for i, (x, y) in enumerate(lane):
                x = x / IMG_ORGINAL_SIZE[1] * IMG_TARGET_SIZE[1]
                y = y / IMG_ORGINAL_SIZE[0] * IMG_TARGET_SIZE[0]
                lane[i] = (x, y)

    return datainfos


def get_seg_mask(lanes, img_size, width=WIDTH):
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
        if img.size != IMG_TARGET_SIZE:
            img = img.resize(IMG_TARGET_SIZE[::-1])
        img.save(img_path)

        # save lanes
        lanes = data["lanes"]  # [N, 2]
        with open(label_path, "w") as f:
            for lane in lanes:
                for x, y in lane:
                    f.write(f"{x} {y} ")
                f.write("\n")

        # save seg mask
        seg = get_seg_mask(lanes, img_size=IMG_TARGET_SIZE)
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
    split = "train"
    llamas_root = "./data/llamas"
    out_root = "./data/llamas_re"
    datainfos = extract_datainfos(llamas_root, split)
    align_format(datainfos)
    save(datainfos, out_root, split)
