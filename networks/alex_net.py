from libs.caffe_tensorflow.network import Network
from utils import default_paths as paths


class AlexNet(Network):
    def setup(self):
        """
        Creates a Neural Net with a AlexNet architecture. The input data must have been previously set in
         the constructor of the object as 'data'.
         E.g.:
            net = AlexNet({'data': input_tensor})

         Architecture taken from:
            http://vision.stanford.edu/teaching/cs231b_spring1415/slides/alexnet_tugce_kyunghee.pdf
        :return: None
        """
        (self.feed('data')
         .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
         .lrn(2, 2e-05, 0.75, name='norm1')
         .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
         .conv(5, 5, 256, 1, 1, group=2, name='conv2')
         .lrn(2, 2e-05, 0.75, name='norm2')
         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
         .conv(3, 3, 384, 1, 1, name='conv3')
         .conv(3, 3, 384, 1, 1, group=2, name='conv4')
         .conv(3, 3, 256, 1, 1, group=2, name='conv5')
         .max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
         .fc(2048, name='fc6')
         .dropout(keep_prob=0.5, name="dp1")
         .fc(1024, name='fc7')
         .dropout(keep_prob=0.5, name="dp2")
         .fc(101, relu=False, name='fc8'))

    @property
    def data_path(self):
        # TODO ver si esto se puede poner en otra parte para no usar default_paths en el framework principal
        return paths.get_alexnet_weights_path()

    @property
    def has_transfer_learning(self):
        return True

    @property
    def trainable_layers(self):
        return ["fc6", "fc7", "fc8"]
