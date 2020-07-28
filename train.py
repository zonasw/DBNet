# -*- coding: utf-8 -*-
# @Time    : 2020/6/16 23:52
# @Author  : zonas.wang
# @Email   : zonas.wang@gmail.com
# @File    : train.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import os.path as osp

from tensorflow.keras import callbacks
from tensorflow.keras import optimizers

from generate import generate
from models.model import DBNet
from config import DBConfig
cfg = DBConfig()


train_generator = generate(cfg, 'train')
val_generator = generate(cfg, 'val')

model = DBNet(cfg, model='training')


load_weights_path = cfg.PRETRAINED_MODEL_PATH
if load_weights_path:
    model.load_weights(load_weights_path, by_name=True, skip_mismatch=True)

model.compile(optimizer=optimizers.Adam(learning_rate=cfg.LEARNING_RATE),
              loss=[None] * len(model.output.shape))
# model.compile(optimizer=optimizers.SGD(learning_rate=cfg.LEARNING_RATE, momentum=0.9),
#                loss=[None] * len(model.output.shape))
model.summary()

# callbacks
checkpoint_callback = callbacks.ModelCheckpoint(
    osp.join(cfg.CHECKPOINT_DIR, 'db_{epoch:02d}_{loss:.4f}_{val_loss:.4f}.h5'))
tensorboard_callback = callbacks.TensorBoard(log_dir=cfg.LOG_DIR,
                                             histogram_freq=1,
                                             write_graph=True,
                                             write_images=True,
                                             update_freq='epoch',   # 'batch'/'epoch'/value_of_int32
                                             profile_batch=2,
                                             embeddings_freq=1,
                                             embeddings_metadata=None)
callbacks = [checkpoint_callback, tensorboard_callback]


model.fit(
    x=train_generator,
    steps_per_epoch=cfg.STEPS_PER_EPOCH,
    initial_epoch=cfg.INITIAL_EPOCH,
    epochs=cfg.EPOCHS,
    verbose=1,
    callbacks=callbacks,
    validation_data=val_generator,
    validation_steps=cfg.VALIDATION_STEPS
)

