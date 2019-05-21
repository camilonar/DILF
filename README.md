# DILF: Deep Incremental Learning Framework

DILF is a simple framework built over TensorFlow specifically designed to facilitate the implementation and testing of Incremental Learning algorithms that make use of Neural Networks.

DILF stablishes a set of modules and classes that are loosely connected and have clearly defined tasks, which facilitates the separation of concerns of each components and the development of extensible algorithms with replicable results.

DILF is divided in 4 modules, each one related to a core functionality of the framework:
-	**Extraction, Transformation, and Load (ETL) Module**: focused in the loading of data for training and testing.
-	**Network Architectures Module**: for seamless integration of new Network Architectures (adapted from [Caffe-Tensorflow](https://github.com/ethereon/caffe-tensorflow)).
-	**Training Module**: it's responsible for the training of the Network. It's designed so that the addition of new algorithms doesn't affect other modules.
-	**Experiments Module**: creates and executes experiments over multiple Architectures, Training Algorithms and Datasets.

Each module is extensible in itself, however, a number of hook methods are provided so that the addition of new algorithms, datasets and experiments is easy and requires minimum effort. The framework aims to reduce the complexity and verbosity of TensorFlow by providing a baseline structure that incorporates many of the common steps required to build an incremental training algorithm.

# Getting Started
#### Prerequisites

  - Python 3.6+
  - TensorFlow 1.9.0+
  - Numpy 1.14.5+
  
In order to use the matplotlib plotting module, you also need these dependencies:

  - matplotlib
  - colour

#### Basic Installation
Since DILF is based in Python scripts, no installation is requred, besides installation of its dependencies. You can install the dependencies using the following command:
```sh
pip install -r requirements.txt
```
Be mindful that this will install tensorflow-gpu version, which requires CUDA and CuDNN. You can see a guide on the requirements [here](https://www.tensorflow.org/install/gpu).
Alternatively, you can also install Tensorflow CPU version, which doesn't include GPU usage or acceleration.
#### Usage
You can execute the prepackaged Experiments by executing ```program_menu.py``` for a simple menu interface, or ```program_shell.py``` to have full control of the execution. You can execute ```program_menu.py``` with the following command, and then you can follow the instructions that appear in the window:
```sh
python program_menu.py
```
An example execution of ```program_shell.py``` might look like this:
```sh
python program_shell.py --dataset=MNIST --optimizer=TR_BASE --checkpoint_key=0-2000 --summaries_interval=600 --checkpoints_interval=2000 --seed=123 --train_mode=INCREMENTAL --dataset_path=../datasets/MNIST --testing_scenario=0
```
You can see the purpouse of each flag with:
```sh
python program_shell.py -h
```
However, the most important arguments are:
- ```dataset```: with this, you set which dataset is going to be used. Currently supported datasets are: CIFAR (CIFAR-10), MNIST, FASHION_MNIST and CALTECH (Caltech 101)
- ```optimizer```: with this, you set which training algorithm is going to be used. Currently supported algorithms are: TR_BASE (RMSProp) and TR_REP (Training with Representatives)
- ```testing_scenario```: an Experiment may have multiple testing scenarios, which are predefined tests with fixed parameters under the same structure (e.g. RMSProp with lr=0.001, or lr=0.1). You can specify which scenario you want to execute (By default is scenario 0). You can inspect the code of each Experiment for details on its scenarios (a functionality to easily access scenarios descriptions will be added in the future)

You can also use ```utils/read_tensorbooard.py``` to create a TensorBoard folder with the average results of multiple tests. This is useful for investigation, since the average of multiple runs is used when reporting results. To use this function, you can use a command like this:
```sh
python utils\read_tensorboard.py --input_folder=.\summaries\MNIST\TR_BASE_INCREMENTAL_0 --output_folder=results\folder\location -m "accuracy" "loss"
```
# Adding new algorithms with DILF
Here, we present a simple tutorial to implement the algorithm RMSProp for training over MNIST using LeNet. The implementation of other algorithms and the integration of other datasets uses similar steps. However, please note that this is a basic example, and is in fact possible to change and extend any part of the framework if you desire. In order to do that, we strongly recommend reading the framework documentation.

#### Step 1 - Data pipeline
To be able to use the input pipeline and support the new dataset, it is needed to create a new class that inherites from **Data**, implementing the following methods:
- *_build_generic_data_tensor*: builds the tensors corresponding to images and labels (used for data for training and testing)
- *close*: closes any file opened by the pipeline.

Additionaly, it is important to take in account the way in which the data is stored in disk, since the framework already provides Readers for two formats: Directories in Caltech-101 style, and TFRecords. In this case, we use **TFRecordsReader** as Reader for the pipeline.

The implementation of *_build_generic_data_tensor* allows the building of tensors for training and testing in the Data class- In this method, it is possible to apply a number of operations to the data before the training starts, such as: transformation of data, Data augmentation, random shuffle, etc. The framework also includes a helper function named *prepare_basic_dataset* that incorporates many common functions: shuffle, cache, repeat and batch. The implementation of this method is shown below:
```py
def _build_generic_data_tensor(self, reader_data, shuffle, augmentation, testing, skip_count=0):

    filenames = reader_data[0]
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=len(self.general_config.train_configurations))
    
    # Note: the parser must also be implemented
    dataset = dataset.map(self.parser, num_parallel_calls=8)
    dataset = self.prepare_basic_dataset(dataset, shuffle=shuffle, cache=True, repeat=testing, skip_count=skip_count, shuffle_seed=const.SEED)
    
    iterator = dataset.make_initializable_iterator()
    images_batch, target_batch = iterator.get_next()
    return iterator, images_batch, target_batch
```
To use an already existing Pipeline, you only need to invoke the methods *build_train_data_tensor* and *build_test_data_tensor* to obtain the training and testing data respectively. The data is provided as tensors.
 
#### Step 2 - Defining a Network Architecture 
To create a new Network Architecture, it is needed to create a new class that inherites from **Network** and implements the method *setup*, defining and linking each layer in the correct order. **LeNet** implementation is showed here:
```py
class LeNet(Network):
    def setup(self):

        (self.feed('data')
         .conv(5, 5, 6, 1, 1, padding='VALID', name='conv1')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
         .conv(5, 5, 20, 1, 1, padding='VALID', name='conv2')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool2')
         .fc(120, name='fc1')
         .fc(84, name='fc2')
         .fc(10, relu=False, name='fc3'))
```

#### Step 3 - Training a Neural Network 
To implement the algorithm **RMSProp** it is necesary to create a class that inherites from **Trainer** and that implements the following methods:
- *_create_loss*: where the loss operator is defined. E.g. TF's Softmax Cross Entropy
- *_create_optimizer*: where the optimizer is defined
- *_train_batch*: where the samples from a batch are received and the optimizer is applied

A sample implementation is shown below:
```py
class RMSPropTrainer(Trainer):

    def _create_loss(self, tensor_y, net_output):
        return tf.losses.softmax_cross_entropy(tensor_y, net_output)
        
    def _create_optimizer(self, config, loss, var_list=None):
        return tf.train.RMSPropOptimizer(config.learn_rate).minimize(loss, var_list=var_list)
        
    def _train_batch(self, sess, image_batch, target_batch, tensor_x, tensor_y, train_step, loss, increment, iteration, total_it):
        return sess.run([train_step, loss], feed_dict={tensor_x: image_batch, tensor_y: target_batch})
```

To use this component, it is only needed to create an instance of the class passing the Configuration, Data Pipeline abd Neural Network.
 
#### Step 4 – Defining the Experiment
This module links all the components from the other modules together, and also defines the specific configuration of an experiment. The experiment must inherit from the class **Experiment** and implement the following methods:
- *_prepare_data_pipeline*: creates the Pipeline that provides the data
- *_prepare_neural_network*: creates an instance of the model to be trained
- *_prepare_trainer*: creates the object that is tasked with the training of the model
- *_prepare_config*: creates the specific configurations from training and testing. It is necessary to create *one* object of global configuration (**GeneralConfig**) and as many local configuration objects as megabatches (increments) of data has the dataset (**MegabatchConfig**)

A lot of sample implementation can be found inside the ```experiments``` folder, organized by dataset. For example, MNIST has the **MnistExperiment** class in which the neural network and data pipeline are defined, and the **MnistRMSPropExperiment** class which inherites from this class and also defines the trainer and specific configuration for the experiment. This is to show that you can set your experiments in such a structure that reutilization and common configurations for multiple experiments are possible.

It is important to note that if, for example, you desire to execute an experiment that doesn't use LeNet but another Neural Network, or use Adagrad instead of RMSProp, you only need to define this in a new Experiment, without modifying the other components of the framework.

#### Step 5 - Executing the Experiment (from Code)
To execute the Experiment, first it's required to have a folder with the data that is going to be used for training and testing (i.e. the Dataset). Then, an instance of the class must be created, and then the methods *prepare_all* and *execute_experiment* must be executed, like this:
```py
exp = MnistRMSPropExperiment(train_dirs, validation_dir, summaries_interval, ckp_interval, ckp_key)
exp.prepare_all(train_mode, training_scenario)
exp.execute_experiment()
```
While an Experiment is executed, the results of the training are stored in real time in log files that can be accesed by using TensorBoard, like this:
```sh
tensorboard --logdir="log/folder/location"
```