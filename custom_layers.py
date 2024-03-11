import tensorflow as tf
from tensorflow.keras import layers

class GatedConv2D(layers.Layer):
    def __init__(self, filters, kernel_size, activation='tanh', **kwargs):
        super(GatedConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        
    def build(self, input_shape):
        self.conv_layer = layers.Conv2D(self.filters, self.kernel_size, padding='same', activation=self.activation)
        self.gate_layer = layers.Conv2D(self.filters, self.kernel_size, padding='same', activation='sigmoid')
    
    def call(self, inputs):
        conv_output = self.conv_layer(inputs)
        gate_output = self.gate_layer(inputs)
        gated_output = conv_output * gate_output
        return gated_output
