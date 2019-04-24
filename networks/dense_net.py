from libs.caffe_tensorflow.network import Network


class DenseNet(Network):

    def setup(self):
        """
        Creates a DenseNet for 256x256 images

        :return: None
        """
        growth_k = 12
        num_class = 100
        (self.feed('data')
         .conv(7, 7, growth_k * 2, 2, 2, name="conv_1")
         .dense_block(nb_layers=6, growth_k=growth_k, dropout_rate=0.2, name="dense_1")
         .transition_layer(growth_k=growth_k, dropout_rate=0.2, name="trans_1")

         .dense_block(nb_layers=12, growth_k=growth_k, dropout_rate=0.2, name="dense_2")
         .transition_layer(growth_k=growth_k, dropout_rate=0.2, name="trans_2")

         .dense_block(nb_layers=48, growth_k=growth_k, dropout_rate=0.2, name="dense_3")
         .transition_layer(growth_k=growth_k, dropout_rate=0.2, name="trans_3")
         .dense_block(nb_layers=32, growth_k=growth_k, dropout_rate=0.2, name="dense_final")

         .batch_normalization(name="linear_batch")
         .relu(name="relu")
         .global_average_pooling()
         .flatten()
         .linear(class_num=num_class, name='linear')
         )
