from libs.caffe_tensorflow.network import Network
from utils import default_paths as paths


class VGGNet(Network):
    def setup(self):
        """
        Creates a Neural Net with a VGG16 architecture. The input data must have been previously set in
         the constructor of the object as 'data'.
         E.g.:
            net = VGGNet({'data': input_tensor})

        :return: None
        """
        (self.feed('data')
         .conv(3, 3, 64, 1, 1, padding='SAME', name='conv1_1')
         .conv(3, 3, 64, 1, 1, padding='SAME', name='conv1_2')
         .max_pool(2, 2, 2, 2, padding='SAME', name='pool1')
         .conv(3, 3, 128, 1, 1, padding='SAME', name='conv2_1')
         .conv(3, 3, 128, 1, 1, padding='SAME', name='conv2_2')
         .max_pool(2, 2, 2, 2, padding='SAME', name='pool2')
         .conv(3, 3, 256, 1, 1, padding='SAME', name='conv3_1')
         .conv(3, 3, 256, 1, 1, padding='SAME', name='conv3_2')
         .conv(3, 3, 256, 1, 1, padding='SAME', name='conv3_3')
         .max_pool(2, 2, 2, 2, padding='SAME', name='pool3')
         .conv(3, 3, 512, 1, 1, padding='SAME', name='conv4_1')
         .conv(3, 3, 512, 1, 1, padding='SAME', name='conv4_2')
         .conv(3, 3, 512, 1, 1, padding='SAME', name='conv4_3')
         .max_pool(2, 2, 2, 2, padding='SAME', name='pool4')
         .conv(3, 3, 512, 1, 1, padding='SAME', name='conv5_1')
         .conv(3, 3, 512, 1, 1, padding='SAME', name='conv5_2')
         .conv(3, 3, 512, 1, 1, padding='SAME', name='conv5_3')
         .max_pool(2, 2, 2, 2, padding='SAME', name='pool5')
         .fc(256, name='fc7')
         .fc(101, name='fc8'))

    @property
    def data_path(self):
        # TODO ver si esto se puede poner en otra parte para no usar default_paths en el framework principal
        return paths.get_vgg16_weights_path()

    @property
    def has_transfer_learning(self):
        return True

    @property
    def trainable_layers(self):
        return ["fc7", "fc8"]
