import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from typing import Optional

class AverageLayer(Layer):
    def __init__(self, **kwargs):
        super(AverageLayer, self).__init__(**kwargs)

    def build(self, inputs_shape):
        inputs_shape = inputs_shape if isinstance(inputs_shape, list) else [inputs_shape]

        if len(inputs_shape) != 1:
            raise ValueError("AverageLayer expect one input.")

        # The first (and required) input is the actual input to the layer
        input_shape = inputs_shape[0]

        # Expected input shape consists of a triplet: (batch, input_length, input_dim)
        if len(input_shape) != 3:
            raise ValueError("Input shape for AverageLayer should be of 3 dimension.")

        self.input_length = int(input_shape[1])
        self.input_dim = int(input_shape[2])

        super(AverageLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        inputs = inputs if isinstance(inputs, list) else [inputs]

        if len(inputs) != 1:
            raise ValueError("AverageLayer expect one input.")

        actual_input = inputs[0]

        # (batch, input_length, input_dim) = mean => (batch, input_dim)
        result = K.mean(actual_input, axis=1)

        return result

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2] # (batch, input_dim)
