import os
import os.path as osp
import json
import random

from tqdm import tqdm


val_ratio = 0.01


ann_path = "annotations.json"

with open(ann_path, "r", encoding="utf8") as f:
    datas = json.load(f)

data_root = datas["data_root"]
data_list = datas["data_list"]

data_num = len(data_list)
val_num = int(data_num * val_ratio) if val_ratio < 1 else val_ratio
train_num = data_num - val_num

random.shuffle(data_list)

train_list = data_list[:train_num]
val_list = data_list[train_num:]


train = {"data_root": data_root, "data_list": train_list}
with open("train.json", "w", encoding="utf8") as f:
    json.dump(train, f, indent=2)


val = {"data_root": data_root, "data_list": val_list}
with open("val.json", "w", encoding="utf8") as f:
    json.dump(val, f, indent=2)

