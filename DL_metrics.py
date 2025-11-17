import tensorflow as tf
import tensorflow.keras.backend as K

Tensor = tf.Tensor  # Alias for type hinting

def dice_coef(
    y_true: Tensor, 
    y_pred: Tensor, 
    smooth: float = 1e-5, 
    weight: float = 1
) -> Tensor:
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         = 2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    intersection = K.sum(K.abs(y_true * y_pred), axis=(1, 2, 3))
    return K.mean((weight * 2. * intersection + smooth) /
                  (weight * K.sum(K.square(y_true), axis=(1, 2, 3)) +
                   weight * K.sum(K.square(y_pred), (1, 2, 3)) + smooth))
