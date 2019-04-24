import tensorflow as tf

from libs.caffe_tensorflow.network import Network


class NiN(Network):
    def setup(self):
        """
        Creates a Neural Net with NiN architecture. The input data must have been previously set in
        the constructor of the object as 'data'.
        E.g.:
            net = NiN({'data': input_tensor})

        :return: None
        """
        (self.feed('data')
         .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
         .conv(1, 1, 96, 1, 1, name='mlp1')
         .conv(1, 1, 96, 1, 1, name='mlp2')
         .max_pool(3, 3, 2, 2, name='pool1')
         .dropout(0.5, name='drop1')
         .conv(5, 5, 256, 1, 1, name='conv2')
         .conv(1, 1, 256, 1, 1, name='mlp3')
         .conv(1, 1, 256, 1, 1, name='mlp4')
         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
         .dropout(0.5, name='drop2')
         .conv(3, 3, 384, 1, 1, name='conv3')
         .conv(1, 1, 384, 1, 1, name='mlp5')
         .conv(1, 1, 384, 1, 1, name='mlp6')
         .max_pool(3, 3, 2, 2, padding='VALID', name='pool3')
         .conv(3, 3, 1024, 1, 1, name='conv4')
         .conv(1, 1, 1024, 1, 1, name='mlp7')
         .conv(1, 1, 100, 1, 1, name='mlp8')
         .avg_pool(6, 6, 1, 1, padding='VALID', name='pool4'))

    def get_output(self):
        return tf.squeeze(self.terminals[-1])
