# -*- coding: utf-8 -*-
# @Time    : 2020/6/16 23:47
# @Author  : zonas.wang
# @Email   : zonas.wang@gmail.com
# @File    : resnet.py
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers as KL


class BatchNormalization(KL.BatchNormalization):
    """
    Identical to KL.BatchNormalization, but adds the option to freeze parameters.
    """
    def __init__(self, freeze, *args, **kwargs):
        self.freeze = freeze
        super(BatchNormalization, self).__init__(*args, **kwargs)

        # set to non-trainable if freeze is true
        self.trainable = not self.freeze

    def call(self, *args, **kwargs):
        # Force test mode if frozen, otherwise use default behaviour (i.e., training=None).
        if self.freeze:
            kwargs['training'] = False
        return super(BatchNormalization, self).call(*args, **kwargs)

    def get_config(self):
        config = super(BatchNormalization, self).get_config()
        config.update({'freeze': self.freeze})
        return config


parameters = {
    "kernel_initializer": "he_normal"
}

class ResNet(tf.keras.Model):
    """
    Constructs a `keras.models.Model` object using the given block count.
    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)
    :param blocks: the network’s residual architecture
    :param block: a residual block (e.g. an instance of `keras_resnet.blocks.basic_2d`)
    :param include_top: if true, includes classification layers
    :param classes: number of classes to classify (include_top must be true)
    :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)
    :param numerical_names: list of bool, same size as blocks, used to indicate whether names of layers should include numbers or letters
    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)
    Usage:
    """
    def __init__(
        self,
        inputs,
        blocks,
        block,
        include_top=True,
        classes=1000,
        freeze_bn=True,
        numerical_names=None,
        *args,
        **kwargs
    ):
        if K.backend.image_data_format() == "channels_last":
            axis = 3
        else:
            axis = 1

        if numerical_names is None:
            numerical_names = [True] * len(blocks)

        x = KL.Conv2D(64, (7, 7), strides=(2, 2), use_bias=False, name="conv1", padding="same")(inputs)
        x = BatchNormalization(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn_conv1")(x)
        x = KL.Activation("relu", name="conv1_relu")(x)
        x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="pool1")(x)

        features = 64

        outputs = []

        for stage_id, iterations in enumerate(blocks):
            for block_id in range(iterations):
                x = block(
                    features,
                    stage_id,
                    block_id,
                    numerical_name=(block_id > 0 and numerical_names[stage_id]),
                    freeze_bn=freeze_bn
                )(x)

            features *= 2

            outputs.append(x)

        if include_top:
            assert classes > 0

            x = KL.GlobalAveragePooling2D(name="pool5")(x)
            x = KL.Dense(classes, activation="softmax", name="fc1000")(x)

            super(ResNet, self).__init__(inputs=inputs, outputs=x, *args, **kwargs)
        else:
            # Else output each stages features
            super(ResNet, self).__init__(inputs=inputs, outputs=outputs, *args, **kwargs)


def resnet_basic(
    filters,
    stage=0,
    block=0,
    kernel_size=3,
    numerical_name=False,
    stride=None,
    freeze_bn=False
):
    """
    A two-dimensional basic block.
    :param filters: the output’s feature space
    :param stage: int representing the stage of this block (starting from 0)
    :param block: int representing this block (starting from 0)
    :param kernel_size: size of the kernel
    :param numerical_name: if true, uses numbers to represent blocks instead of chars (ResNet{101, 152, 200})
    :param stride: int representing the stride used in the shortcut and the first conv layer, default derives stride from block id
    :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)
    Usage:
    """
    if stride is None:
        if block != 0 or stage == 0:
            stride = 1
        else:
            stride = 2

    if K.backend.image_data_format() == "channels_last":
        axis = 3
    else:
        axis = 1

    if block > 0 and numerical_name:
        block_char = "b{}".format(block)
    else:
        block_char = chr(ord('a') + block)

    stage_char = str(stage + 2)

    def f(x):
        y = KL.ZeroPadding2D(padding=1, name="padding{}{}_branch2a".format(stage_char, block_char))(x)

        y = KL.Conv2D(filters, kernel_size, strides=stride, use_bias=False, name="res{}{}_branch2a".format(stage_char, block_char), **parameters)(y)

        y = BatchNormalization(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn{}{}_branch2a".format(stage_char, block_char))(y)

        y = KL.Activation("relu", name="res{}{}_branch2a_relu".format(stage_char, block_char))(y)

        y = KL.ZeroPadding2D(padding=1, name="padding{}{}_branch2b".format(stage_char, block_char))(y)

        y = KL.Conv2D(filters, kernel_size, use_bias=False, name="res{}{}_branch2b".format(stage_char, block_char), **parameters)(y)

        y = BatchNormalization(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn{}{}_branch2b".format(stage_char, block_char))(y)

        if block == 0:
            shortcut = KL.Conv2D(filters, (1, 1), strides=stride, use_bias=False, name="res{}{}_branch1".format(stage_char, block_char), **parameters)(x)

            shortcut = BatchNormalization(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn{}{}_branch1".format(stage_char, block_char))(shortcut)
        else:
            shortcut = x

        y = KL.Add(name="res{}{}".format(stage_char, block_char))([y, shortcut])

        y = KL.Activation("relu", name="res{}{}_relu".format(stage_char, block_char))(y)

        return y

    return f


def resnet_bottleneck(
    filters,
    stage=0,
    block=0,
    kernel_size=3,
    numerical_name=False,
    stride=None,
    freeze_bn=False
):
    """
    A two-dimensional bottleneck block.
    :param filters: the output’s feature space
    :param stage: int representing the stage of this block (starting from 0)
    :param block: int representing this block (starting from 0)
    :param kernel_size: size of the kernel
    :param numerical_name: if true, uses numbers to represent blocks instead of chars (ResNet{101, 152, 200})
    :param stride: int representing the stride used in the shortcut and the first conv layer, default derives stride from block id
    :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)
    Usage:
    """
    if stride is None:
        if block != 0 or stage == 0:
            stride = 1
        else:
            stride = 2

    if K.backend.image_data_format() == "channels_last":
        axis = 3
    else:
        axis = 1

    if block > 0 and numerical_name:
        block_char = "b{}".format(block)
    else:
        block_char = chr(ord('a') + block)

    stage_char = str(stage + 2)

    def f(x):
        y = KL.Conv2D(filters, (1, 1), strides=stride, use_bias=False, name="res{}{}_branch2a".format(stage_char, block_char), **parameters)(x)

        y = BatchNormalization(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn{}{}_branch2a".format(stage_char, block_char))(y)

        y = KL.Activation("relu", name="res{}{}_branch2a_relu".format(stage_char, block_char))(y)

        y = KL.ZeroPadding2D(padding=1, name="padding{}{}_branch2b".format(stage_char, block_char))(y)

        y = KL.Conv2D(filters, kernel_size, use_bias=False, name="res{}{}_branch2b".format(stage_char, block_char), **parameters)(y)

        y = BatchNormalization(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn{}{}_branch2b".format(stage_char, block_char))(y)

        y = KL.Activation("relu", name="res{}{}_branch2b_relu".format(stage_char, block_char))(y)

        y = KL.Conv2D(filters * 4, (1, 1), use_bias=False, name="res{}{}_branch2c".format(stage_char, block_char), **parameters)(y)

        y = BatchNormalization(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn{}{}_branch2c".format(stage_char, block_char))(y)

        if block == 0:
            shortcut = KL.Conv2D(filters * 4, (1, 1), strides=stride, use_bias=False, name="res{}{}_branch1".format(stage_char, block_char), **parameters)(x)

            shortcut = BatchNormalization(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn{}{}_branch1".format(stage_char, block_char))(shortcut)
        else:
            shortcut = x

        y = KL.Add(name="res{}{}".format(stage_char, block_char))([y, shortcut])

        y = KL.Activation("relu", name="res{}{}_relu".format(stage_char, block_char))(y)

        return y

    return f

class ResNet18(ResNet):
    """
    Constructs a `keras.models.Model` according to the ResNet18 specifications.
    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)
    :param blocks: the network’s residual architecture
    :param include_top: if true, includes classification layers
    :param classes: number of classes to classify (include_top must be true)
    :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)
    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)
    Usage:
    """
    def __init__(self, inputs, blocks=None, include_top=True, classes=1000, freeze_bn=False, *args, **kwargs):
        if blocks is None:
            blocks = [2, 2, 2, 2]

        super(ResNet18, self).__init__(
            inputs,
            blocks,
            block=resnet_basic,
            include_top=include_top,
            classes=classes,
            freeze_bn=freeze_bn,
            *args,
            **kwargs
        )


class ResNet34(ResNet):
    """
    Constructs a `keras.models.Model` according to the ResNet34 specifications.
    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)
    :param blocks: the network’s residual architecture
    :param include_top: if true, includes classification layers
    :param classes: number of classes to classify (include_top must be true)
    :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)
    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)
    Usage:
    """
    def __init__(self, inputs, blocks=None, include_top=True, classes=1000, freeze_bn=False, *args, **kwargs):
        if blocks is None:
            blocks = [3, 4, 6, 3]

        super(ResNet34, self).__init__(
            inputs,
            blocks,
            block=resnet_basic,
            include_top=include_top,
            classes=classes,
            freeze_bn=freeze_bn,
            *args,
            **kwargs
        )


class ResNet50(ResNet):
    """
    Constructs a `keras.models.Model` according to the ResNet50 specifications.
    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)
    :param blocks: the network’s residual architecture
    :param include_top: if true, includes classification layers
    :param classes: number of classes to classify (include_top must be true)
    :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)
    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)
    Usage:
    """
    def __init__(self, inputs, blocks=None, include_top=True, classes=1000, freeze_bn=False, *args, **kwargs):
        if blocks is None:
            blocks = [3, 4, 6, 3]

        numerical_names = [False, False, False, False]

        super(ResNet50, self).__init__(
            inputs,
            blocks,
            numerical_names=numerical_names,
            block=resnet_bottleneck,
            include_top=include_top,
            classes=classes,
            freeze_bn=freeze_bn,
            *args,
            **kwargs
        )


class ResNet101(ResNet):
    """
    Constructs a `keras.models.Model` according to the ResNet101 specifications.
    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)
    :param blocks: the network’s residual architecture
    :param include_top: if true, includes classification layers
    :param classes: number of classes to classify (include_top must be true)
    :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)
    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)
    Usage:
    """
    def __init__(self, inputs, blocks=None, include_top=True, classes=1000, freeze_bn=False, *args, **kwargs):
        if blocks is None:
            blocks = [3, 4, 23, 3]

        numerical_names = [False, True, True, False]

        super(ResNet101, self).__init__(
            inputs,
            blocks,
            numerical_names=numerical_names,
            block=resnet_bottleneck,
            include_top=include_top,
            classes=classes,
            freeze_bn=freeze_bn,
            *args,
            **kwargs
        )


class ResNet152(ResNet):
    """
    Constructs a `keras.models.Model` according to the ResNet152 specifications.
    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)
    :param blocks: the network’s residual architecture
    :param include_top: if true, includes classification layers
    :param classes: number of classes to classify (include_top must be true)
    :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)
    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)
    Usage:
    """
    def __init__(self, inputs, blocks=None, include_top=True, classes=1000, freeze_bn=False, *args, **kwargs):
        if blocks is None:
            blocks = [3, 8, 36, 3]

        numerical_names = [False, True, True, False]

        super(ResNet152, self).__init__(
            inputs,
            blocks,
            numerical_names=numerical_names,
            block=resnet_bottleneck,
            include_top=include_top,
            classes=classes,
            freeze_bn=freeze_bn,
            *args,
            **kwargs
        )


class ResNet200(ResNet):
    """
    Constructs a `keras.models.Model` according to the ResNet200 specifications.
    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)
    :param blocks: the network’s residual architecture
    :param include_top: if true, includes classification layers
    :param classes: number of classes to classify (include_top must be true)
    :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)
    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)
    Usage:
    """
    def __init__(self, inputs, blocks=None, include_top=True, classes=1000, freeze_bn=False, *args, **kwargs):
        if blocks is None:
            blocks = [3, 24, 36, 3]

        numerical_names = [False, True, True, False]

        super(ResNet200, self).__init__(
            inputs,
            blocks,
            numerical_names=numerical_names,
            block=resnet_bottleneck,
            include_top=include_top,
            classes=classes,
            freeze_bn=freeze_bn,
            *args,
            **kwargs
        )
