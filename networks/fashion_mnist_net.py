from libs.caffe_tensorflow.network import Network


class FashionMnistNet(Network):
    def setup(self):
        """
        Network to be used with Fashion MNIST dataset

        This code was taken and adapted from:
        https://medium.com/tensorflow/hello-deep-learning-fashion-mnist-with-keras-50fcff8cd74a

        :return: None
        """
        (self.feed('data')
         .conv(2, 2, 64, 1, 1, padding='VALID', name='conv1')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
         .dropout(0.3, name="dp1")
         .conv(2, 2, 32, 1, 1, padding='VALID', name='conv2')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool2')
         .dropout(0.3, name="dp1")
         .fc(256, name='fc1')
         .dropout(0.5, name="dp2")
         .fc(10, relu=False, name='fc3'))
