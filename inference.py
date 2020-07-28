# -*- coding: utf-8 -*-
# @Time    : 2020/6/16 23:51
# @Author  : zonas.wang
# @Email   : zonas.wang@gmail.com
# @File    : inference.py
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import os.path as osp
import time

import tensorflow as tf
import cv2
import glob
import numpy as np
import pyclipper
from shapely.geometry import Polygon
from tqdm import tqdm

from models.model import DBNet
from config import DBConfig
cfg = DBConfig()


def resize_image(image, image_short_side=736):
    height, width, _ = image.shape
    if height < width:
        new_height = image_short_side
        new_width = int(math.ceil(new_height / height * width / 32) * 32)
    else:
        new_width = image_short_side
        new_height = int(math.ceil(new_width / width * height / 32) * 32)
    resized_img = cv2.resize(image, (new_width, new_height))
    return resized_img


def box_score_fast(bitmap, _box):
    h, w = bitmap.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def unclip(box, unclip_ratio=1.5):
    poly = Polygon(box)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded


def get_mini_boxes(contour):
    if not contour.size:
        return [], 0
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [points[index_1], points[index_2],
           points[index_3], points[index_4]]
    return box, min(bounding_box[1])


def polygons_from_bitmap(pred, bitmap, dest_width, dest_height, max_candidates=500, box_thresh=0.7):
    pred = pred[..., 0]
    bitmap = bitmap[..., 0]
    height, width = bitmap.shape
    boxes = []
    scores = []

    contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours[:max_candidates]:
        epsilon = 0.001 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = approx.reshape((-1, 2))
        if points.shape[0] < 4:
            continue
        score = box_score_fast(pred, points.reshape(-1, 2))
        if box_thresh > score:
            continue
        if points.shape[0] > 2:
            box = unclip(points, unclip_ratio=2.0)
            if len(box) > 1:
                continue
        else:
            continue

        box = box.reshape(-1, 2)
        _, sside = get_mini_boxes(box.reshape((-1, 1, 2)))
        if sside < 5:
            continue

        box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
        box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
        boxes.append(box.tolist())
        scores.append(score)
    return boxes, scores


def main():
    BOX_THRESH = 0.5
    mean = np.array([103.939, 116.779, 123.68])

    model_path = "checkpoints/2020-07-24/db_83_2.0894_1.9788.h5"

    img_dir = 'datasets/test/input'
    img_names = os.listdir(img_dir)

    model = DBNet(cfg, model='inference')
    model.load_weights(model_path, by_name=True, skip_mismatch=True)
    for img_name in tqdm(img_names):
        img_path = osp.join(img_dir, img_name)
        image = cv2.imread(img_path)
        src_image = image.copy()
        h, w = image.shape[:2]
        image = resize_image(image)
        image = image.astype(np.float32)
        image -= mean
        image_input = np.expand_dims(image, axis=0)
        image_input_tensor = tf.convert_to_tensor(image_input)
        start_time = time.time()
        p = model.predict(image_input_tensor)[0]
        end_time = time.time()
        print("time: ", end_time - start_time)

        bitmap = p > 0.3
        boxes, scores = polygons_from_bitmap(p, bitmap, w, h, box_thresh=BOX_THRESH)
        for box in boxes:
            cv2.drawContours(src_image, [np.array(box)], -1, (0, 255, 0), 2)
        image_fname = osp.split(img_path)[-1]
        cv2.imwrite('datasets/test/output/' + image_fname, src_image)


if __name__ == '__main__':
    main()

