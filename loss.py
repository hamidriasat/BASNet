import tensorflow as tf


"""
calculate intersection over union (IoU) between images
"""
def iou(y_true, y_pred):

    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)

    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f + y_pred_f - y_true_f * y_pred_f)

    return intersection / union


"""
IoU loss
"""
def iou_loss(y_true, y_pred):
    return 1 - iou(y_true, y_pred)


"""
Binary crossentropy loss
"""
def bce_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    return bce(y_true, y_pred)


"""
SSIM loss
"""
def ssim_loss(y_true, y_pred):
    return 1 - tf.image.ssim(y_true, y_pred, max_val=1)