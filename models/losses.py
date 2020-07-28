# -*- coding: utf-8 -*-
# @Time    : 2020/6/16 23:48
# @Author  : zonas.wang
# @Email   : zonas.wang@gmail.com
# @File    : losses.py
import tensorflow as tf
import tensorflow.keras as K


def balanced_crossentropy_loss(pred, gt, mask, negative_ratio=3.):
    pred = pred[..., 0]
    positive_mask = (gt * mask)
    negative_mask = ((1 - gt) * mask)
    positive_count = tf.reduce_sum(positive_mask)
    negative_count = tf.reduce_min([tf.reduce_sum(negative_mask), positive_count * negative_ratio])
    # loss_fun = tf.losses.BinaryCrossentropy()
    # loss = loss_fun(gt, pred)
    # loss = K.losses.binary_crossentropy(gt, pred)
    loss = K.backend.binary_crossentropy(gt, pred)
    positive_loss = loss * positive_mask
    negative_loss = loss * negative_mask
    negative_loss, _ = tf.nn.top_k(tf.reshape(negative_loss, (-1,)), tf.cast(negative_count, tf.int32))

    balanced_loss = (tf.reduce_sum(positive_loss) + tf.reduce_sum(negative_loss)) / (
            positive_count + negative_count + 1e-6)
    return balanced_loss, loss


def dice_loss(pred, gt, mask, weights):
    """
    Args:
        pred: (b, h, w, 1)
        gt: (b, h, w)
        mask: (b, h, w)
        weights: (b, h, w)
    Returns:
    """
    pred = pred[..., 0]
    weights = (weights - tf.reduce_min(weights)) / (tf.reduce_max(weights) - tf.reduce_min(weights) + 1e-6) + 1.
    mask = mask * weights
    intersection = tf.reduce_sum(pred * gt * mask)
    union = tf.reduce_sum(pred * mask) + tf.reduce_sum(gt * mask) + 1e-6
    loss = 1 - 2.0 * intersection / union
    return loss


def l1_loss(pred, gt, mask):
    pred = pred[..., 0]
    mask_sum = tf.reduce_sum(mask)
    loss = K.backend.switch(mask_sum > 0, tf.reduce_sum(tf.abs(pred - gt) * mask) / (mask_sum + 1e-6), tf.constant(0.))
    return loss


def compute_cls_acc(pred, gt, mask):

    zero = tf.zeros_like(pred, tf.float32)
    one = tf.ones_like(pred, tf.float32)

    pred = tf.where(pred < 0.3, x=zero, y=one)
    acc = tf.reduce_mean(tf.cast(tf.equal(pred * mask, gt * mask), tf.float32))

    return acc


def db_loss(args, alpha=5.0, beta=10.0, ohem_ratio=3.0):
    input_gt, input_mask, input_thresh, input_thresh_mask, binarize_map, thresh_binary, threshold_map = args

    threshold_loss = l1_loss(threshold_map, input_thresh, input_thresh_mask)
    binarize_loss, dice_loss_weights = balanced_crossentropy_loss(binarize_map, input_gt, input_mask, negative_ratio=ohem_ratio)
    thresh_binary_loss = dice_loss(thresh_binary, input_gt, input_mask, dice_loss_weights)

    model_loss = alpha * binarize_loss + beta * threshold_loss + thresh_binary_loss
    return model_loss


def db_acc(args):
    input_gt, input_mask, binarize_map, thresh_binary = args
    binarize_acc = compute_cls_acc(binarize_map, input_gt, input_mask)
    thresh_binary_acc = compute_cls_acc(thresh_binary, input_gt, input_mask)
    return binarize_acc, thresh_binary_acc
