"""
This helper module reads multiple tests results (TensorBoard summaries) and converts them into a single folder with the
average results of the metrics over all the executions
"""
import os
import argparse
import numpy as np
import tensorflow as tf


def _read_data_from_tests(tests_folder: str, metric_tags):
    """
    Reads data from a series of test folders and transforms it into a nested structured that can easily be processed

    :param tests_folder: the root folder where all the tests are located (e.g. ../summaries/CIFAR-10/TR_BASE). This
            folder must have the tests results located in one subfolder each, following the structure of summaries
            provided by the framework
    :param metric_tags: a list of strings containing the metrics to be read (e.g. ['loss', 'accuracy']
    :return: a nested structure with the following structure:
            {increment:
                {metric_tag:
                    {iteration:
                        **[List of values]**}}}
            The List in the innermost part of the structure contains the respective values of the test, and has shape
            [n_tests, 2], where the first column contains the relative time of the event and the second one contains
            the value of the event
    """
    results = {}
    list_tests = os.listdir(tests_folder)
    for test in list_tests:
        path_test = os.path.join(tests_folder, test)
        list_increments = os.listdir(path_test)
        starting_time = -1
        for increment in list_increments:
            path_increment = os.path.join(path_test, increment)
            list_events = os.listdir(path_increment)
            if results.get(increment) is None:
                results[increment] = {tag: dict() for tag in metric_tags}
            for event_file in list_events:
                print("Processing event file: .../{}/{}/{}".format(test, increment, event_file))
                for e in tf.train.summary_iterator(os.path.join(path_increment, event_file)):
                    for v in e.summary.value:
                        if v.tag in metric_tags:
                            if starting_time < 0:
                                starting_time = e.wall_time
                            if results[increment].get(v.tag).get(e.step) is None:
                                results[increment].get(v.tag)[e.step] = []
                            values = results[increment].get(v.tag)[e.step]
                            values.append([e.wall_time - starting_time, v.simple_value])
    return results


def _calculate_average(results):
    """
    Calculates the average value of the results

    :param results: a nested structure, as provided by _read_data_from_tests
    :return:  a nested structure with the following structure:
            {increment:
                {metric_tag:
                    {iteration:
                        [average_relative_time, average_value]}}}
    """
    for inc_key in results:
        inc_dict = results[inc_key]
        for metric_key in inc_dict:
            m_dict = inc_dict[metric_key]
            for it_key in m_dict:
                results[inc_key][metric_key][it_key] = np.average(results[inc_key][metric_key][it_key], axis=0)
    return results


def _write_average_result(results, metric_tags, output_folder):
    """
    Writes the results over a structure of files similar to that one used in Tester. The files created have the metrics
    of the original data split into two: by iteration, and by relative time. The files can be read by TensorBoard

    :param results:  a nested structure, as provided by _calculate_average
    :param metric_tags: a list of strings containing the metrics to be written (e.g. ['loss', 'accuracy']
    :param output_folder: the root folder where the results are going to be written
    :return: None
    """
    sess = tf.InteractiveSession()
    x = tf.placeholder(dtype=tf.float32, shape=(), name='aux_tensor')
    summaries_it = {metric_key: tf.summary.scalar(metric_key + '_iter', x) for metric_key in metric_tags}
    summaries_time = {metric_key: tf.summary.scalar(metric_key + '_time', x) for metric_key in metric_tags}
    for inc_key in results:
        print("Writing increment {}".format(inc_key))
        inc_dict = results[inc_key]
        with tf.summary.FileWriter(os.path.join(output_folder, inc_key),
                                   tf.get_default_graph()) as writer:
            for metric_key in inc_dict:
                m_dict = inc_dict[metric_key]
                for it_key in m_dict:
                    value = results[inc_key][metric_key][it_key]
                    summ_it, summ_time = sess.run((summaries_it[metric_key], summaries_time[metric_key]),
                                                  feed_dict={x: value[1]})
                    writer.add_summary(summ_it, global_step=it_key)
                    writer.add_summary(summ_time, global_step=value[0])
                    print("Metric: {} - It. {} - Time {}: {}".format(metric_key, it_key, value[0], value[1]))
    sess.close()


def create_average_from_tests(tests_folder: str, metric_tags, output_folder):
    """
    Reads the results from a set of tests relating to the same overall Experiment and produces files with the
    average results of the tests. This files can be read by TensorBoard

    :param tests_folder: the root folder where all the tests are located (e.g. ../summaries/CIFAR-10/TR_BASE). This
            folder must have the tests results located in one subfolder each, following the structure of summaries
            provided by the framework
    :param metric_tags: a list of strings containing the metrics to be read (e.g. ['loss', 'accuracy']
    :param output_folder: the root folder where the results are going to be written
    :return: None
    """
    results = _read_data_from_tests(tests_folder, metric_tags)
    results = _calculate_average(results)
    _write_average_result(results, metric_tags, output_folder)


def _shell_exec():
    """
    Executes the program from a shell

    :return: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input_folder',
        type=str,
        help='The root folder where all the tests are located (e.g. ../summaries/CIFAR-10/TR_BASE).',
        required=True)
    parser.add_argument(
        '-o',
        '--output_folder',
        type=str,
        help='The root folder where the results are going to be written.',
        required=True)
    parser.add_argument('-m', '--metrics_list', nargs='+', type=str,
                        help='List of strings containing the metrics to be read (e.g. [\'loss\', \'accuracy\']',
                        required=True)
    args = vars(parser.parse_args())
    print(args)
    create_average_from_tests(args['input_folder'], args['metrics_list'], args['output_folder'])
    return 0


if __name__ == '__main__':
    _shell_exec()
