from libs.caffe_tensorflow.network import Network


class LeNet(Network):
    def setup(self):
        """
        Creates a Neural Net with LeNet-5 architecture. The input shape of the tensor must follow the shape
        [None,32,32,C], where C is the number of color channels of the images, and C>=1. If the images are grayscale
        then C=1. This Tensor must have been previously set in the constructor of the object as 'data'
        E.g.:
            net = LeNet({'data': input_tensor})

        This code was taken and adapted from:
        https://github.com/sujaybabruwad/LeNet-in-Tensorflow

        :return: None
        """
        (self.feed('data')
         .conv(5, 5, 6, 1, 1, padding='VALID', name='conv1')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
         .conv(5, 5, 20, 1, 1, padding='VALID', name='conv2')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool2')
         .fc(120, name='fc1')
         .fc(84, name='fc2')
         .fc(10, relu=False, name='fc3'))
