import os
os.environ["TF_KERAS"] = '1'
import tensorflow as tf
tf.config.list_physical_devices('GPU')