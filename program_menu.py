"""
This module acts as a menu interface for performing the experiments
"""

import utils.constants as const
import utils.default_paths as paths
import utils.exp_helper as helper


def print_menu():
    """
    Prints the menu for the configuration of a Test

    :return: a str, corresponding to the response of the user
    """
    print()
    print("--------------MENU FOR CONFIGURATION OF TESTS---------------------")
    print("[1] Select Dataset and associated Neural Network")
    print("[2] Select Optimizer")
    print("[3] Select Checkpoint to Load (Optional)")
    print("[4] Select Summary and Checkpoint interval (If not provided then the default values will be used)")
    print("[5] Select Seed (If not provided then the default value will be used)")
    print("[0] Finish configuration and execute Test")
    print("[X] Exit")
    print("------------------------------------------------------------------")
    print()
    response = input("Select an option: ")
    return response


def ask_for_configuration():
    """
    Asks the user for the following configurations for a test:

        - Dataset
        - Optimizer (between the supported types)
        - Load of a checkpoint (Optional)
        - Learning Rate (Optional)
        - Summary Interval (Optional)
        - Checkpoint Interval (Optional)
        - Seed for random numbers (Optional)

    :return: a tuple containing 3 strings, 2 ints and a float: name of the dataset (str), name of the optimizer (str),
        representation of a checkpoint (str), summary interval (int), checkpoint interval (int) and seed (float)
        in that order. The representation of a checkpoint will be returned as None if the user doesn't configure it.
        An example of a return value is: ('MNIST', 'TR_BASE', None, 100, 300, 12345).
    """

    # Creation of variables
    response = "s"
    dataset, optimizer, checkpoint = None, None, None
    summary_interval, ckp_interval = const.SUMMARY_INTERVAL, const.CKP_INTERVAL
    seed = const.SEED

    while response:
        response = print_menu()

        if response == "1":
            dataset = configure_dataset_and_neural_net(dataset)
        elif response == "2":
            optimizer = configure_optimizer(optimizer)
        elif response == "3":
            checkpoint = configure_checkpoint(checkpoint)
        elif response == "4":
            summary_interval, ckp_interval = configure_intervals(summary_interval, ckp_interval)
        elif response == "5":
            seed = configure_seed(seed)
        elif response == "0":
            if dataset and optimizer:
                return dataset, optimizer, checkpoint, summary_interval, ckp_interval, seed
            else:
                print("Both the Dataset and Optimizer must be configured before performing a test")
                continue
        elif response.upper() == 'X':
            break
        else:
            print("Option '{}' is not recognized.".format(response))
            continue
    print("Exiting...")


def configure_dataset_and_neural_net(curr_dataset: str):
    """
        Configures the optimizer appropriately

        :param curr_dataset: the current dataset that has been configured by the user
        :return: a str representing the new dataset if the user has changed it to a valid value, otherwise, the current
            dataset will be returned
        :rtype: str
        """
    print()
    print("----------------------------Configure the Dataset----------------------------")
    print("\n**Note** The dataset is associated with a fixed neural network. For example, if you choose Tiny \n"
          "Imagenet Dataset then CaffeNet will be automatically selected as Model for the training.\n")

    while True:
        print("[M] MNIST (uses LeNet)")
        print("[F] FASHION-MNIST (uses LeNet)")
        print("[C] CIFAR-10 (uses TFModel)")
        print("[D] CIFAR-100 (uses NiN)")
        print("[L] CALTECH-101 (uses AlexNet)")
        print("[I] TINY IMAGENET (uses CaffeNet)")
        print("[X] Cancel Operation and return to Main Menu")
        response = input("Select a dataset: ").upper()
        if response == 'M':
            response = const.DATA_MNIST
        elif response == 'C':
            response = const.DATA_CIFAR_10
        elif response == 'F':
            response = const.DATA_FASHION_MNIST
        elif response == 'L':
            response = const.DATA_CALTECH_101
        elif response == 'D':
            response = const.DATA_CIFAR_100
        elif response == 'I':
            response = const.DATA_TINY_IMAGENET
        elif response.upper() == 'X':
            break
        else:
            print("Option '{}' is not recognized.".format(response))
            continue
        print("The dataset has been set to {}".format(response))
        return response

    print("The dataset hasn't been changed and the current configured value is {}".format(curr_dataset))
    return curr_dataset


def configure_optimizer(curr_optimizer: str):
    """
    Configures the optimizer appropriately

    :param curr_optimizer: the current optimizer that has been configured by the user
    :return: a str representing the new optimizer if the user has changed it to a valid value, otherwise, the current
        optimizer will be returned
    :rtype: str
    """
    print()
    print("----------------------------Configure the Optimizer----------------------------")

    while True:
        print("[B] Simple RMSProp (Base optimizer)")
        print("[N] Incremental Random Representative Sampling (NIL)")
        print("[R] Incremental Representative Sampling with BvSB and Crowding Distance (RILBC)")
        print("[X] Cancel Operation and return to Main Menu")
        response = input("Select an optimizer: ").upper()
        if response == 'B':
            response = const.TR_BASE
        elif response == 'N':
            response = const.TR_NIL
        elif response == 'R':
            response = const.TR_RILBC
        elif response == 'X':
            break
        else:
            print("Option '{}' is not recognized.".format(response))
            continue
        print("The optimizer has been set to {}".format(response))
        return response

    print("The optimizer hasn't been changed and the current configured value is {}".format(curr_optimizer))
    return curr_optimizer


def configure_checkpoint(curr_ckp: str):
    """
    Configures a representation of a checkpoint based in the corresponding increment (mega-batch) and iteration given
    by the user. Notice that this function **DOES NOT** check if the given checkpoint is valid. Also, the full
    checkpoint path must be created at a later moment, since that path is heavily dependent of other parameters like
    Dataset and Optimizer that may change during configuration time.

    :param curr_ckp: a string representing the increment and iteration of a checkpoint. It must be in the form:
        *"[increment]-[iteration]"*, e.g. "1-20"
    :return: a string representing the new checkpoint if the user has changed it, otherwise, the current checkpoint will
        be returned. A value of None will be returned if there was a checkpoint already configured and the user decided
        to not load a checkpoint anymore.
    :rtype: str
    """
    print()
    print("----------------------------Configure the Checkpoint----------------------------")
    while True:
        print("[1] Select a checkpoint to be loaded")
        print("[2] Reset checkpoint")
        print("[X] Cancel Operation and return to Main Menu")
        response = input("Select an option: ").upper()
        if response == '1':
            response = get_checkpoint()
        elif response == '2':
            response = reset_checkpoint(curr_ckp)
            return response
        elif response == 'X':
            break
        else:
            print("Option '{}' is not recognized.".format(response))
            continue
        print("The checkpoint has been changed and corresponds to increment {} and iteration {}."
              .format(response.split("-")[0], response.split("-")[1]))
        return response

    if curr_ckp:
        print("The checkpoint hasn't been changed and currently corresponds to increment {} and iteration {}."
              .format(curr_ckp.split("-")[0], curr_ckp.split("-")[1]))
    else:
        print("No checkpoint has been selected at this moment.")

    return curr_ckp


def get_checkpoint():
    """
    Ask the user for the values of increment and iteration for the checkpoint

    :return: a string representing the increment and iteration, with the format *"[increment]-[iteration]"*, e.g. "0-50"
    :rtype: str
    """
    print()
    while True:
        response = input("Please enter the mega-batch and iteration corresponding to the desired checkpoint with the "
                         "format [mega-batch]-[iteration], e.g. '0-50' (both must be ints equal or greater than 0): ")
        if len(response.split("-")) == 2:
            if 0 <= int(response.split("-")[0]):
                if 0 <= int(response.split("-")[1]):
                    return response
                else:
                    print("Invalid value for the iteration. Must be a int greater or equal than 0")
            else:
                print("Invalid value for the mega-batch. Must be a int greater or equal than 0")
        else:
            print("The format of the checkpoint is [mega-batch]-[iteration], e.g. '0-50', '1-0'")


def reset_checkpoint(curr_ckp: str):
    """
    Resets the checkpoint

    :param curr_ckp: a string representing the increment and iteration of a checkpoint.
    :return: None if the checkpoint is reset, otherwise, the current checkpoint will be returned
    :rtype: str
    """
    print()

    while True:
        response = input(
            "Are you sure you want to reset the checkpoint? This will mean that no checkpoint will be loaded"
            "when the training starts [Y/N]: ").upper()
        if response == 'Y':
            print("The checkpoint has been reset")
            return None
        elif response == 'N':
            print("The checkpoint has NOT been reset")
            return curr_ckp
        else:
            print("Option '{}' is not recognized.".format(response))
            continue


def configure_intervals(curr_s_interval: int, curr_ckp_interval: int):
    """
    Configures the summary interval and checkpoint interval

    :param curr_s_interval: the current summary interval that has been configured by the user
    :param curr_ckp_interval: the current checkpoint interval that has been configured by the user
    :return: a tuple with 2 integers: the new summary interval and the new checkpoint interval if the user has changed
        them to a valid value, otherwise, the current summary and checkpoint intervals will be returned
    """
    print()
    print("----------------------------Configure the Intervals----------------------------")
    print("Press [X] if you want to cancel the operation")

    while True:
        response = input("Set a value for the summary interval (must be a integer greater than 0): ")
        if response.upper() == "X":
            break
        try:
            s_interval = int(response)
            if 0 < s_interval:
                ckp_multiplier = get_checkpoint_multiplier()
                if ckp_multiplier > 0:
                    print("The summary interval has been set to {}".format(s_interval))
                    print("The checkpoint interval has been set to {}".format(s_interval * ckp_multiplier))
                    return s_interval, ckp_multiplier * s_interval
            else:
                print("Invalid value for the summary interval. Must be a int greater than 0")
        except ValueError:
            print("Invalid value for the summary interval. Must be an int")

    print("The summary inxterval hasn't been changed and the current configured value is {}".format(curr_s_interval))
    print(
        "The checkpoint interval hasn't been changed and the current configured value is {}".format(curr_ckp_interval))
    return curr_s_interval, curr_ckp_interval


def get_checkpoint_multiplier():
    """
    Ask the user for the multiplier for the checkpoint interval, since this interval must be a multiple of the summary
    interval

    :return: an int greater than 0 that corresponds to the multiplier selected by the User, or -1 if the User cancels
        the operation
    :rtype: int
    """
    print("\nPress [X] if you want to cancel the operation")
    print("\n The checkpoint interval is a multiplier of the summary interval. For example, if you set a multiplier \n"
          "of 3 and the summary interval is 200, then the checkpoint interval will be set to 600.")
    while True:
        response = input("Set a value for the checkpoint interval multiplier (must be a integer greater than 0): ")
        if response.upper() == "X":
            return -1
        try:
            ckp_multiplier = int(response)
            if 0 < ckp_multiplier:
                return ckp_multiplier
            else:
                print("Invalid value for the checkpoint interval multiplier. Must be a int greater than 0")
        except ValueError:
            print("Invalid value for the checkpoint interval multiplier. Must be an int")


def configure_seed(curr_seed: int):
    """
    Asks the user for the seed for random number generators

    :param curr_seed: the current value of the seed
    :return: an int corresponding to the new seed if the user has changed it to a valid value, otherwise, the current
        seed will be returned
    :rtype: int
    """
    print()
    print("----------------------------Configure the Seed----------------------------")
    print("Press [X] if you want to cancel the operation")

    while True:
        response = input("Set a value for the seed (must be a float): ")
        if response.upper() == "X":
            return curr_seed
        try:
            seed = int(response)
            return seed
        except ValueError:
            print("Invalid value for the seed. Must be a float")


def main():
    """
    Executes the program

    :return: None
    """
    train_mode = const.TRAIN_MODE
    testing_scenario = 0
    dataset, optimizer, checkpoint, s_interval, ckp_interval, seed = ask_for_configuration()
    train_dirs, validation_dir = paths.get_paths_from_dataset(dataset)
    helper.print_config(dataset, optimizer, checkpoint, s_interval, ckp_interval, seed, train_mode,
                        train_dirs, validation_dir, testing_scenario)
    helper.perform_experiment(dataset, optimizer, checkpoint, s_interval, ckp_interval, seed, train_mode,
                              train_dirs, validation_dir, testing_scenario)
    return 0


if __name__ == '__main__':
    main()
