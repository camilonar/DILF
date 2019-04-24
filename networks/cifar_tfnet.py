from libs.caffe_tensorflow.network import Network


class CifarTFNet(Network):
    def setup(self):
        """
         Architecture taken from:
            https://www.tensorflow.org/tutorials/images/deep_cnn#model_training
        :return: None
        """
        (self.feed('data')
         .conv(5, 5, 64, 1, 1, padding='VALID', name='conv1')
         .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
         .lrn(4, (0.001 / 9.0), 0.75, name='norm1')
         .conv(5, 5, 64, 1, 1, name='conv2')
         .lrn(4, (0.001 / 9.0), 0.75, name='norm2')
         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
         .fc(384, name='fc3')
         .dropout(0.6, name="drop4")
         .fc(192, name='fc5')
         .fc(10, relu=False, name='fc6'))
