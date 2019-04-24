from libs.caffe_tensorflow.network import Network


class SimpleNet(Network):
    def setup(self):
        """
        Creates a SimpleNet

        :return: None
        """
        (self.feed('data')
         .conv(3, 3, 64, 1, 1, padding='VALID', name='conv1')
         .batch_normalization(name="bn1")
         .relu(name="rel1")
         .conv(3, 3, 128, 1, 1, padding='VALID', name='conv2')
         .batch_normalization(name="bn2")
         .relu(name="rel2")
         .conv(3, 3, 128, 1, 1, padding='VALID', name='conv3')
         .batch_normalization(name="bn3")
         .relu(name="rel3")
         .conv(3, 3, 128, 1, 1, padding='VALID', name='conv4')
         .batch_normalization(name="bn4")
         .relu(name="rel4")

         .max_pool(2, 2, 2, 2, name='pool1')
         .dropout(0.1, name='drop1')

         .conv(3, 3, 128, 1, 1, padding='VALID', name='conv5')
         .batch_normalization(name="bn5")
         .relu(name="rel5")
         .conv(3, 3, 128, 1, 1, padding='VALID', name='conv6')
         .batch_normalization(name="bn6")
         .relu(name="rel6")

         .conv(3, 3, 128, 1, 1, padding='VALID', name='conv7')
         .batch_normalization(name="bn7")
         .relu(name="rel7")

         .max_pool(2, 2, 2, 2, name='pool2')
         .dropout(0.1, name='drop2')

         .conv(3, 3, 128, 1, 1, padding='VALID', name='conv8')
         .batch_normalization(name="bn8")
         .relu(name="rel8")
         .conv(3, 3, 128, 1, 1, padding='VALID', name='conv9')
         .batch_normalization(name="bn9")
         .relu(name="rel9")

         .max_pool(2, 2, 2, 2, name='pool3')
         .dropout(0.1, name='drop3')

         .conv(3, 3, 128, 1, 1, padding='VALID', name='conv10')
         .batch_normalization(name="bn10")
         .relu(name="rel10")

         .conv(1, 1, 128, 1, 1, padding='VALID', name='conv11')
         .batch_normalization(name="bn11")
         .relu(name="rel11")

         .conv(1, 1, 128, 1, 1, padding='VALID', name='conv12')
         .batch_normalization(name="bn12")
         .relu(name="rel12")

         .max_pool(2, 2, 2, 2, name='pool4')
         .dropout(0.1, name='drop4')

         .conv(3, 3, 128, 1, 1, padding='VALID', name='conv13')

         .max_pool(2, 2, 2, 2, name='pool5')
         .dropout(0.1, name='drop5')

         .fc(100, relu=False, name='fc3'))
