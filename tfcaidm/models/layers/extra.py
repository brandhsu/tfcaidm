"""Extra layer functions"""

from tensorflow.keras import layers

layer_name = lambda x, name: layers.Lambda(lambda x: x, name=name)(x)
