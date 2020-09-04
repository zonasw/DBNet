import os
import os.path as osp
import json

from tqdm import tqdm
from glob import glob


save_ann_path = "annotations.json"
ext = ".jpg"
labelme_data_dir = "/hd2/data/labelme_data"


img_dir = osp.join(labelme_data_dir, "*"+ext)
img_paths = glob(img_dir)

data_list = []
for img_path in tqdm(img_paths):
    img_name = osp.split(img_path)[-1]
    annotations = []
    labelme_ann_path = osp.join(labelme_data_dir, osp.splitext(img_name)[0]+".json")
    with open(labelme_ann_path, "r", encoding="utf8") as f:
        labelme_ann = json.load(f)
    shapes = labelme_ann["shapes"]
    for shape in shapes:
        points = shape["points"]
        text = "unknown"
        annotations.append({"polygon": points, "text": text})
    data_list.append({"img_name": img_name, "annotations": annotations})

ann = {"data_root": labelme_data_dir, "data_list": data_list}
with open(save_ann_path, "w", encoding="utf8") as f:
    json.dump(ann, f, indent=2)

