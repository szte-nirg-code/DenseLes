import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, Dropout, BatchNormalization, 
    Concatenate, Activation, MaxPooling2D, UpSampling2D
)
from tensorflow.keras.models import Model

def densenet_block(
    n_filters: int, 
    input_layer: tf.Tensor, 
    dropout: float = 0.2
) -> tf.Tensor:
    """
    Creates a 4-layer DenseNet block.
    
    Args:
        n_filters: Number of filters for each Conv2D layer.
        input_layer: The input tensor to the block.
        dropout: Dropout rate.
        
    Returns:
        The output tensor of the DenseNet block.
    """
    g1 = Conv2D(n_filters, (3, 3), padding='same')(input_layer)
    g1 = BatchNormalization()(g1)
    g1 = Dropout(dropout)(g1)
    g1 = Activation('elu')(g1)
    
    c1 = Concatenate()([g1, input_layer])
    
    g2 = Conv2D(n_filters, (3, 3), padding='same')(c1)
    g2 = BatchNormalization()(g2)
    g2 = Dropout(dropout)(g2)
    g2 = Activation('elu')(g2)
    
    c2 = Concatenate()([g2, c1])
    
    g3 = Conv2D(n_filters, (3, 3), padding='same')(c2)
    g3 = BatchNormalization()(g3)
    g3 = Dropout(dropout)(g3)
    g3 = Activation('elu')(g3)
    
    c3 = Concatenate(axis=-1)([g3, input_layer])
    
    g4 = Conv2D(n_filters, (3, 3), padding='same')(c3)
    g4 = BatchNormalization()(g4)
    g4 = Dropout(dropout)(g4)
    g4 = Activation('elu')(g4)
    
    c4 = Concatenate(axis=-1)([g1, g2, g3, g4])
    
    return c4

def dense_model(
    inputs: tf.Tensor, 
    kernel_size: int = 3, 
    n_filter: int = 4, 
    d_filter: int = 8, 
    dropout: float = 0.2
) -> tf.keras.Model:
    """
    Creates the full U-Net-like model with DenseNet blocks.
    
    Args:
        inputs: The input tensor (from tf.keras.layers.Input).
        kernel_size: Kernel size for the initial convolution.
        n_filter: Number of filters for the initial convolution.
        d_filter: Number of filters for the DenseNet blocks.
        dropout: Dropout rate.
        
    Returns:
        A compiled Keras Model.
    """
    c1 = Conv2D(n_filter, (kernel_size, kernel_size), activation='elu', padding='same', kernel_initializer='he_normal')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Dropout(dropout)(c1)
    d1 = densenet_block(d_filter, c1)
    cc1 = Concatenate(axis=-1)([c1, d1])
    
    p1 = MaxPooling2D((2, 2))(cc1)
    d2 = densenet_block(d_filter, p1)
    cc2 = Concatenate(axis=-1)([p1, d2])
    
    p2 = MaxPooling2D((2, 2))(cc2)
    d3 = densenet_block(d_filter, p2)
    
    up1 = UpSampling2D((2, 2))(d3)
    up1 = Concatenate(axis=-1)([cc2, up1])
    
    d4 = densenet_block(d_filter, up1)
    up2 = UpSampling2D((2, 2))(d4)
    up2 = Concatenate(axis=-1)([cc1, up2])
    
    d5 = densenet_block(d_filter, up2)
    
    output = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer='he_normal')(d5)
    
    model = Model(inputs, output)
    
    return model
