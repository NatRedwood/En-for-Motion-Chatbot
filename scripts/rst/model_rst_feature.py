import tensorflow as tf
import tensorflow_hub as hub

MODEL_PATH = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim50-with-normalization/1"
hub_layer = hub.KerasLayer(MODEL_PATH, output_shape=[50], input_shape=[], 
                           dtype=tf.string, trainable=True)

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0, l2=0.001)))
model.add(tf.keras.layers.Dense(1))