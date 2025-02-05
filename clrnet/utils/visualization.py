import cv2
import os
import os.path as osp


COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
    (255, 0, 128),
    (0, 128, 255),
    (0, 255, 128),
    (128, 255, 255),
    (255, 128, 255),
    (255, 255, 128),
    (60, 180, 0),
    (180, 60, 0),
    (0, 60, 180),
    (0, 180, 60),
    (60, 0, 180),
    (180, 0, 60),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
]


def imshow_lanes(img, lanes, show=False, out_file=None, width=4):
    """
    Args:
        img (_type_): cv2, [h, w, 3]
        lanes (_type_): list of array, [n, m, 2]
        show (bool, optional): _description_. Defaults to False.
        out_file (_type_, optional): _description_. Defaults to None.
        width (int, optional): _description_. Defaults to 4.
    """
    lanes_xys = []
    for _, lane in enumerate(lanes):
        xys = []
        for x, y in lane:
            if x <= 0 or y <= 0:
                continue
            x, y = int(x), int(y)
            xys.append((x, y))
        lanes_xys.append(xys)
    if len(lanes_xys) != 0:
        lanes_xys.sort(key=lambda xys : xys[0][0])

    for idx, xys in enumerate(lanes_xys):
        for i in range(1, len(xys)):
            cv2.line(img, xys[i - 1], xys[i], COLORS[idx], thickness=width)


    if show:
        cv2.imshow('view', img)
        cv2.waitKey(0)

    if out_file:
        if not osp.exists(osp.dirname(out_file)):
            os.makedirs(osp.dirname(out_file))
        cv2.imwrite(out_file, img)