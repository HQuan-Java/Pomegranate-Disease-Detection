import tensorflow as tf
import numpy as np
import os

model = tf.keras.models.load_model('model/pomegranate_final.h5')
print("✅ Model loaded")
print(f"Input shape : {model.input_shape}")
print(f"Output shape: {model.output_shape}")
print(f"Total params: {model.count_params():,}")
model.summary()