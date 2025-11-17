import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import binary_crossentropy
from scipy.ndimage import distance_transform_edt as distance

# Import from our refactored metrics module
from DL_metrics import dice_coef

Tensor = tf.Tensor  # Alias for type hinting

def weighted_bce_loss(y_true: Tensor, y_pred: Tensor, weight: float) -> Tensor:
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))

    loss = (1. - y_true) * logit_y_pred + (1. + (weight - 1.) * y_true) * \
           (K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
    return K.sum(loss) / K.sum(weight)

def weighted_dice_loss(y_true: Tensor, y_pred: Tensor, weight: float = 0.75) -> Tensor:
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    smooth = 1.
    w, m1, m2 = weight * weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * m1) + K.sum(w * m2) + smooth)
    loss = 1. - K.sum(score)
    return loss

def weighted_bce_dice_loss(y_true: Tensor, y_pred: Tensor) -> Tensor:
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    
    averaged_mask = K.pool3d(
        y_true, pool_size=(15, 15, 15), strides=(1, 1, 1), padding='same', pool_mode='avg'
    )
    border = K.cast(K.greater(averaged_mask, 0.005), 'float32') * \
             K.cast(K.less(averaged_mask, 0.995), 'float32')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight += border * 2
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    
    loss = weighted_bce_loss(y_true, y_pred, weight) + weighted_dice_loss(y_true, y_pred, weight)
    return loss

def focal_loss(y_true: Tensor, y_pred: Tensor) -> Tensor:
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    gamma = 4.
    alpha = .15
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    
    return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - \
           K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

def dice_focal_loss(y_true: Tensor, y_pred: Tensor) -> Tensor:
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    dice_ = 1 - dice_coef(y_true, y_pred)
    focal_ = focal_loss(y_true, y_pred)
    loss = dice_ + (1 * focal_)
    return loss

def tversky_loss(y_true: Tensor, y_pred: Tensor) -> Tensor:
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.7
    return 1 - (true_pos + K.epsilon()) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + K.epsilon())

def calc_dist_map(seg: np.ndarray) -> np.ndarray:
    res = np.zeros_like(seg)
    posmask = seg.astype(bool)
    if posmask.any():
        negmask = ~posmask
        res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res

def calc_dist_map_batch(y_true: Tensor) -> np.ndarray:
    y_true_numpy = y_true.numpy()
    return np.array([calc_dist_map(y) for y in y_true_numpy]).astype(np.float32)

def surface_loss(y_true: Tensor, y_pred: Tensor) -> Tensor:
    y_true_dist_map = tf.py_function(
        func=calc_dist_map_batch,
        inp=[y_true],
        Tout=tf.float32
    )
    multipled = y_pred * y_true_dist_map
    return K.mean(multipled)

def dice_loss(y_true: Tensor, y_pred: Tensor) -> Tensor:
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def dice_loss_v2(y_true: Tensor, y_pred: Tensor) -> Tensor:
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    return 1 - dice_coef(y_true, y_pred)

def confusion(y_true: Tensor, y_pred: Tensor) -> tuple[Tensor, Tensor]:
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    smooth = 1
    y_pred_pos = K.clip(y_pred, 0, 1)
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.clip(y_true, 0, 1)
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
    prec = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    return prec, recall

def tp(y_true: Tensor, y_pred: Tensor) -> Tensor:
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pos = K.round(K.clip(y_true, 0, 1))
    tp = (K.sum(y_pos * y_pred_pos) + smooth) / (K.sum(y_pos) + smooth)
    return tp

def tn(y_true: Tensor, y_pred: Tensor) -> Tensor:
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tn = (K.sum(y_neg * y_pred_neg) + smooth) / (K.sum(y_neg) + smooth)
    return tn

def focal_tversky(y_true: Tensor, y_pred: Tensor) -> Tensor:
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    pt_1 = tversky_loss(y_true, y_pred)
    gamma = 0.25
    return K.pow((1 - pt_1), gamma)

def generalized_dice(y_true: Tensor, y_pred: Tensor, smooth: float = 1e-4) -> Tensor:
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    
    w = K.sum(y_true, axis=(0, 1, 2))
    w = 1 / (w ** 2 + 0.00001)
    
    numerator = y_true * y_pred
    numerator = w * K.sum(numerator, axis=(0, 1, 2))
    numerator = K.sum(numerator)
    
    denominator = y_true + y_pred
    denominator = w * K.sum(denominator, axis=(0, 1, 2))
    denominator = K.sum(denominator)
    
    gen_dice_coef = (numerator + smooth) / (denominator + smooth)
    return 2 * gen_dice_coef

def generalized_dice_loss(y_true: Tensor, y_pred: Tensor, smooth: float = K.epsilon()) -> Tensor:
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    return 1 - generalized_dice(y_true=y_true, y_pred=y_pred, smooth=smooth)

def bce_dice_loss(y_true: Tensor, y_pred: Tensor) -> Tensor:
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
