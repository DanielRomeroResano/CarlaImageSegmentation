import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, callbacks, initializers, applications

# Suppressing unnecessary warnings
import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras import layers, models, activations, initializers, applications
from tensorflow.keras.layers import BatchNormalization, Activation, Concatenate, Conv2D, AveragePooling2D, UpSampling2D

def convBlock(block_input, num_filters=256, kernel_size=3, dilation_rate=1, padding="same", use_bias=False):
    x = Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding=padding,
        use_bias=use_bias,
        kernel_initializer=initializers.HeNormal(),
    )(block_input)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)
    
    return x

def DilatedPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convBlock(x, kernel_size=1, use_bias=True)
    out_pool = UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = convBlock(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convBlock(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convBlock(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convBlock(dspp_input, kernel_size=3, dilation_rate=18)

    x = Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convBlock(x, kernel_size=1)
    return output

def DeepLab(image_shape, num_classes):
    model_input = tf.keras.Input(shape=image_shape)
    resnet50 = applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=model_input
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedPyramidPooling(x)

    input_a = UpSampling2D(
        size=(image_shape[0] // 4 // x.shape[1], image_shape[1] // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convBlock(input_b, num_filters=48, kernel_size=1)

    x = Concatenate(axis=-1)([input_a, input_b])
    x = convBlock(x)
    x = convBlock(x)
    x = UpSampling2D(
        size=(image_shape[0] // x.shape[1], image_shape[1] // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    model = models.Model(inputs=model_input, outputs=model_output)
    
    return model
