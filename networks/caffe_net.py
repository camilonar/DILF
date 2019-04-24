from libs.caffe_tensorflow.network import Network


class CaffeNet(Network):
    def setup(self):
        """
         Creates a Neural Net with a simplified CaffeNet architecture. The input data must have been previously set in
         the constructor of the object as 'data'.
         E.g.:
            net = CaffeNet({'data': input_tensor})

         This code was taken and adapted from:
            https://github.com/ethereon/caffe-tensorflow
        :return: None
        """
        (self.feed('data')
         .conv(5, 5, 32, 1, 1, padding='VALID', name='conv1')
         .conv(5, 5, 32, 1, 1, group=2, name='conv2')
         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
         .conv(3, 3, 64, 1, 1, name='conv3')
         .conv(3, 3, 64, 1, 1, name='conv4')
         .max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
         .fc(512, name='fc6')
         .dropout(0.5, name='drop6')
         .fc(200, relu=False, name='fc8'))
