# -*- coding: utf-8 -*-
# @Time    : 2020/6/16 23:46
# @Author  : zonas.wang
# @Email   : zonas.wang@gmail.com
# @File    : model.py
from tensorflow import keras as K
from tensorflow.keras import layers as KL
import tensorflow as tf

from models.backbones.resnet import ResNet50
from models.losses import db_loss


def DBNet(cfg, k=50, model='training'):
    assert model in ['training', 'inference'], 'error'

    input_image = KL.Input(shape=[None, None, 3], name='input_image')

    backbone = ResNet50(inputs=input_image, include_top=False, freeze_bn=True)
    C2, C3, C4, C5 = backbone.outputs

    # in2
    in2 = KL.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in2')(C2)
    in2 = KL.BatchNormalization()(in2)
    in2 = KL.ReLU()(in2)
    # in3
    in3 = KL.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in3')(C3)
    in3 = KL.BatchNormalization()(in3)
    in3 = KL.ReLU()(in3)
    # in4
    in4 = KL.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in4')(C4)
    in4 = KL.BatchNormalization()(in4)
    in4 = KL.ReLU()(in4)
    # in5
    in5 = KL.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in5')(C5)
    in5 = KL.BatchNormalization()(in5)
    in5 = KL.ReLU()(in5)

    # P5
    P5 = KL.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(in5)
    P5 = KL.BatchNormalization()(P5)
    P5 = KL.ReLU()(P5)
    P5 = KL.UpSampling2D(size=(8, 8))(P5)
    # P4
    out4 = KL.Add()([in4, KL.UpSampling2D(size=(2, 2))(in5)])
    P4 = KL.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(out4)
    P4 = KL.BatchNormalization()(P4)
    P4 = KL.ReLU()(P4)
    P4 = KL.UpSampling2D(size=(4, 4))(P4)
    # P3
    out3 = KL.Add()([in3, KL.UpSampling2D(size=(2, 2))(out4)])
    P3 = KL.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(out3)
    P3 = KL.BatchNormalization()(P3)
    P3 = KL.ReLU()(P3)
    P3 = KL.UpSampling2D(size=(2, 2))(P3)
    # P2
    out2 = KL.Add()([in2, KL.UpSampling2D(size=(2, 2))(out3)])
    P2 = KL.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(out2)
    P2 = KL.BatchNormalization()(P2)
    P2 = KL.ReLU()(P2)

    fuse = KL.Concatenate()([P2, P3, P4, P5])

    # binarize map
    p = KL.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', use_bias=False)(fuse)
    p = KL.BatchNormalization()(p)
    p = KL.ReLU()(p)
    p = KL.Conv2DTranspose(64, (2, 2), strides=(2, 2), kernel_initializer='he_normal', use_bias=False)(p)
    p = KL.BatchNormalization()(p)
    p = KL.ReLU()(p)
    binarize_map  = KL.Conv2DTranspose(1, (2, 2), strides=(2, 2), kernel_initializer='he_normal',
                                       activation='sigmoid', name='binarize_map')(p)

    # threshold map
    t = KL.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', use_bias=False)(fuse)
    t = KL.BatchNormalization()(t)
    t = KL.ReLU()(t)
    t = KL.Conv2DTranspose(64, (2, 2), strides=(2, 2), kernel_initializer='he_normal', use_bias=False)(t)
    t = KL.BatchNormalization()(t)
    t = KL.ReLU()(t)
    threshold_map  = KL.Conv2DTranspose(1, (2, 2), strides=(2, 2), kernel_initializer='he_normal',
                                        activation='sigmoid', name='threshold_map')(t)

    # thresh binary map
    thresh_binary = KL.Lambda(lambda x: 1 / (1 + tf.exp(-k * (x[0] - x[1]))))([binarize_map, threshold_map])

    if model == 'training':
        input_gt = KL.Input(shape=[cfg.IMAGE_SIZE, cfg.IMAGE_SIZE], name='input_gt')
        input_mask = KL.Input(shape=[cfg.IMAGE_SIZE, cfg.IMAGE_SIZE], name='input_mask')
        input_thresh = KL.Input(shape=[cfg.IMAGE_SIZE, cfg.IMAGE_SIZE], name='input_thresh')
        input_thresh_mask = KL.Input(shape=[cfg.IMAGE_SIZE, cfg.IMAGE_SIZE], name='input_thresh_mask')

        loss_layer = KL.Lambda(db_loss, name='db_loss')(
            [input_gt, input_mask, input_thresh, input_thresh_mask, binarize_map, thresh_binary, threshold_map])

        db_model = K.Model(inputs=[input_image, input_gt, input_mask, input_thresh, input_thresh_mask],
                           outputs=[loss_layer])

        loss_names = ["db_loss"]
        for layer_name in loss_names:
            layer = db_model.get_layer(layer_name)
            db_model.add_loss(layer.output)
            # db_model.add_metric(layer.output, name=layer_name, aggregation="mean")
    else:
        db_model = K.Model(inputs=input_image,
                           outputs=binarize_map)
        """
        db_model = K.Model(inputs=input_image,
                           outputs=thresh_binary)
        """
    return db_model


if __name__ == '__main__':
    from config import DBConfig
    cfg = DBConfig()
    model = DBNet(cfg, model='inference')
    model.summary()
