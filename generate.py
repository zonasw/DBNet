# -*- coding: utf-8 -*-
# @Time    : 2020/6/16 23:50
# @Author  : zonas.wang
# @Email   : zonas.wang@gmail.com
# @File    : generate.py
import math
import json
import os.path as osp

import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon
import imgaug.augmenters as iaa

from transform import transform, crop, resize
from config import DBConfig
cfg = DBConfig()


mean = [103.939, 116.779, 123.68]


def show_polys(image, anns, window_name):
    for ann in anns:
        poly = np.array(ann['poly']).astype(np.int32)
        cv2.drawContours(image, np.expand_dims(poly, axis=0), -1, (0, 255, 0), 2)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image)


def draw_thresh_map(polygon, canvas, mask, shrink_ratio=0.4):
    polygon = np.array(polygon)
    assert polygon.ndim == 2
    assert polygon.shape[1] == 2

    polygon_shape = Polygon(polygon)
    distance = polygon_shape.area * (1 - np.power(shrink_ratio, 2)) / polygon_shape.length
    subject = [tuple(l) for l in polygon]
    padding = pyclipper.PyclipperOffset()
    padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    padded_polygon = np.array(padding.Execute(distance)[0])
    cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)

    xmin = padded_polygon[:, 0].min()
    xmax = padded_polygon[:, 0].max()
    ymin = padded_polygon[:, 1].min()
    ymax = padded_polygon[:, 1].max()
    width = xmax - xmin + 1
    height = ymax - ymin + 1

    polygon[:, 0] = polygon[:, 0] - xmin
    polygon[:, 1] = polygon[:, 1] - ymin

    xs = np.broadcast_to(np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
    ys = np.broadcast_to(np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))

    distance_map = np.zeros((polygon.shape[0], height, width), dtype=np.float32)
    for i in range(polygon.shape[0]):
        j = (i + 1) % polygon.shape[0]
        absolute_distance = compute_distance(xs, ys, polygon[i], polygon[j])
        distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
    distance_map = np.min(distance_map, axis=0)

    xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
    xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
    ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
    ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)
    canvas[ymin_valid:ymax_valid, xmin_valid:xmax_valid] = np.fmax(
        1 - distance_map[
            ymin_valid - ymin:ymax_valid - ymin,
            xmin_valid - xmin:xmax_valid - xmin],
        canvas[ymin_valid:ymax_valid, xmin_valid:xmax_valid])


def compute_distance(xs, ys, point_1, point_2):
    square_distance_1 = np.square(xs - point_1[0]) + np.square(ys - point_1[1])
    square_distance_2 = np.square(xs - point_2[0]) + np.square(ys - point_2[1])
    square_distance = np.square(point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])

    cosin = (square_distance - square_distance_1 - square_distance_2) / \
            (2 * np.sqrt(square_distance_1 * square_distance_2) + 1e-6)
    square_sin = 1 - np.square(cosin)
    square_sin = np.nan_to_num(square_sin)
    result = np.sqrt(square_distance_1 * square_distance_2 * square_sin / (square_distance + 1e-6))

    result[cosin < 0] = np.sqrt(np.fmin(square_distance_1, square_distance_2))[cosin < 0]
    return result


def generate(cfg, train_or_val='train'):
    def init_input():
        batch_images = np.zeros([cfg.BATCH_SIZE, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3], dtype=np.float32)
        batch_gts = np.zeros([cfg.BATCH_SIZE, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE], dtype=np.float32)
        batch_masks = np.zeros([cfg.BATCH_SIZE, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE], dtype=np.float32)
        batch_thresh_maps = np.zeros([cfg.BATCH_SIZE, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE], dtype=np.float32)
        batch_thresh_masks = np.zeros([cfg.BATCH_SIZE, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE], dtype=np.float32)
        # batch_loss = np.zeros([cfg.BATCH_SIZE, ], dtype=np.float32)
        return [batch_images, batch_gts, batch_masks, batch_thresh_maps, batch_thresh_masks]

    data_path = cfg.TRAIN_DATA_PATH if train_or_val=='train' else cfg.VAL_DATA_PATH

    with open(data_path, encoding='utf8') as f:
        data = json.load(f)

    data_root_dir = data["data_root"]
    data_list = data["data_list"]

    image_paths = []
    all_anns = []

    for data_item in data_list:
        img_name = data_item["img_name"]
        annotations = data_item["annotations"]
        anns = []
        for annotation in annotations:
            item = {}
            text = annotation["text"]
            poly = annotation["polygon"]
            if len(poly) < 3:
                continue
            item['text'] = text
            item['poly'] = poly
            anns.append(item)
        image_paths.append(osp.join(data_root_dir, img_name))
        all_anns.append(anns)

    transform_aug = iaa.Sequential([iaa.Affine(rotate=(-10, 10)), iaa.Resize((0.5, 3.0))])
    dataset_size = len(image_paths)
    indices = np.arange(dataset_size)
    if train_or_val=='train':
        np.random.shuffle(indices)

    current_idx = 0
    b = 0
    while True:
        if current_idx >= dataset_size:
            if train_or_val=='train':
                np.random.shuffle(indices)
            current_idx = 0
        if b == 0:
            batch_images, batch_gts, batch_masks, batch_thresh_maps, batch_thresh_masks = init_input()
        i = indices[current_idx]
        image_path = image_paths[i]
        anns = all_anns[i]
        """
        [{'text': 'chinese', 'poly': [[17.86985870232934, 29.2253341902275], [18.465581783660582, 7.2334012599376365], [525.2796724953414, 20.9621104524324], [524.6839494140104, 42.954043382722375]]},
        {'text': 'chinese', 'poly': [[9.746362138723043, 329.1153286941807], [10.667025082598343, 295.12779598373265], [589.454714475228, 310.8061443514931], [588.5340515313526, 344.79367706194114]]}]
        """
        image = cv2.imread(image_path)
        # show_polys(image.copy(), anns, 'before_aug')
        if train_or_val=='train':
            transform_aug = transform_aug.to_deterministic()
            image, anns = transform(transform_aug, image, anns)
            image, anns = crop(image, anns)
        image, anns = resize(cfg.IMAGE_SIZE, image, anns)
        # show_polys(image.copy(), anns, 'after_aug')
        # cv2.waitKey(0)
        anns = [ann for ann in anns if Polygon(ann['poly']).is_valid]
        gt = np.zeros((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), dtype=np.float32)
        mask = np.ones((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), dtype=np.float32)
        thresh_map = np.zeros((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), dtype=np.float32)
        thresh_mask = np.zeros((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), dtype=np.float32)
        for ann in anns:
            poly = np.array(ann['poly'])
            height = max(poly[:, 1]) - min(poly[:, 1])
            width = max(poly[:, 0]) - min(poly[:, 0])
            polygon = Polygon(poly)
            # generate gt and mask
            if polygon.area < 1 or min(height, width) < cfg.MIN_TEXT_SIZE or ann['text'] in cfg.IGNORE_TEXT:
                cv2.fillPoly(mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
                continue
            else:
                distance = polygon.area * (1 - np.power(cfg.SHRINK_RATIO, 2)) / polygon.length
                subject = [tuple(l) for l in ann['poly']]
                padding = pyclipper.PyclipperOffset()
                padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                shrinked = padding.Execute(-distance)
                if len(shrinked) == 0:
                    cv2.fillPoly(mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
                    continue
                else:
                    shrinked = np.array(shrinked[0]).reshape(-1, 2)
                    if shrinked.shape[0] > 2 and Polygon(shrinked).is_valid:
                        cv2.fillPoly(gt, [shrinked.astype(np.int32)], 1)
                    else:
                        cv2.fillPoly(mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
                        continue
            # generate thresh map and thresh mask
            draw_thresh_map(ann['poly'], thresh_map, thresh_mask, shrink_ratio=cfg.SHRINK_RATIO)
        thresh_map = thresh_map * (cfg.THRESH_MAX - cfg.THRESH_MIN) + cfg.THRESH_MIN

        image = image.astype(np.float32)
        image -= mean
        batch_images[b] = image
        batch_gts[b] = gt
        batch_masks[b] = mask
        batch_thresh_maps[b] = thresh_map
        batch_thresh_masks[b] = thresh_mask

        b += 1
        current_idx += 1
        if b == cfg.BATCH_SIZE:
            inputs = [batch_images, batch_gts, batch_masks, batch_thresh_maps, batch_thresh_masks]
            # outputs = batch_loss
            outputs = []
            yield inputs, outputs
            b = 0

