#!/usr/bin/env bash
set -ex

# Aux. Variables definition
# ----------------------------------------------------------------------------
INC="python program_shell.py --seed=12345 --train_mode=INCREMENTAL"
ACU="python program_shell.py --seed=12345 --train_mode=ACUMULATIVE"
RMSPROP="--optimizer=TR_BASE"
NIL="--optimizer=TR_REP"
MNIST_EXT="--dataset=MNIST --dataset_path=../data/extra/MNIST_EXTRA"
CIFAR_EXT="--dataset=CIFAR-10 --dataset_path=../data/extra/CIFAR10_EXTRA"
FASHION_EXT="--dataset=FASHION-MNIST --dataset_path=../data/extra/FASHION-MNIST_EXTRA"
CALTECH_EXT="--dataset=CALTECH-101 --dataset_path=../data/extra/101_ObjectCategories_EXTRA"
MNIST_UNB="--dataset=MNIST --dataset_path=../data/unbalanced/MNIST_UNBALANCED"
CIFAR_UNB="--dataset=CIFAR-10 --dataset_path=../data/unbalanced/CIFAR10_UNBALANCED"
FASHION_UNB="--dataset=FASHION-MNIST --dataset_path=../data/unbalanced/FASHION-MNIST_UNBALANCED"
CALTECH_UNB="--dataset=CALTECH-101 --dataset_path=../data/unbalanced/101_ObjectCategories_UNBALANCED"
_OUTPUT_PATH="../results/summaries/*"

# ----------------------------------------------------------------------------
# Tests in class incremental scenario (a class only appears in one megabatch)

#MNIST
$INC $RMSPROP $MNIST_EXT
$ACU $RMSPROP $MNIST_EXT
$INC $NIL $MNIST_EXT # NIL 1% (Scenario 0)
$INC $NIL $MNIST_EXT -ts=1 # NIL 10% (Scenario 1)
#CIFAR-10
$INC $RMSPROP $CIFAR_EXT
$ACU $RMSPROP $CIFAR_EXT
$INC $NIL $CIFAR_EXT # NIL 1% (Scenario 0)
$INC $NIL $CIFAR_EXT -ts=1 # NIL 10% (Scenario 1)
#Fashion-MNIST
$INC $RMSPROP $FASHION_EXT
$ACU $RMSPROP $FASHION_EXT
$INC $NIL $FASHION_EXT # NIL 1% (Scenario 0)
$INC $NIL $FASHION_EXT -ts=1 # NIL 10% (Scenario 1)
#Caltech 101
$INC $RMSPROP $CALTECH_EXT
$ACU $RMSPROP $CALTECH_EXT
$INC $NIL $CALTECH_EXT # NIL 1% (Scenario 0)
$INC $NIL $CALTECH_EXT -ts=1 # NIL 10% (Scenario 1)

mv "../results/summaries/MNIST" "../results/summaries/MNIST_EXTRA"
mv "../results/summaries/FASHION-MNIST" "../results/summaries/FASHION-MNIST_EXTRA"
mv "../results/summaries/CIFAR-10" "../results/summaries/CIFAR-10_EXTRA"
mv "../results/summaries/CALTECH-101" "../results/summaries/CALTECH-101_EXTRA"
# ----------------------------------------------------------------------------
# Tests in unbalanced classes scenario (a class may appear in multiple megabatches, in different proportions)

#MNIST
$INC $RMSPROP $MNIST_UNB
$ACU $RMSPROP $MNIST_UNB
$INC $NIL $MNIST_UNB # NIL 1% (Scenario 0)
$INC $NIL $MNIST_UNB -ts=1 # NIL 10% (Scenario 1)
#CIFAR-10
$INC $RMSPROP $CIFAR_UNB
$ACU $RMSPROP $CIFAR_UNB
$INC $NIL $CIFAR_UNB # NIL 1% (Scenario 0)
$INC $NIL $CIFAR_UNB -ts=1 # NIL 10% (Scenario 1)
#Fashion-MNIST
$INC $RMSPROP $FASHION_UNB
$ACU $RMSPROP $FASHION_UNB
$INC $NIL $FASHION_UNB # NIL 1% (Scenario 0)
$INC $NIL $FASHION_UNB -ts=1 # NIL 10% (Scenario 1)
#Caltech 101
$INC $RMSPROP $CALTECH_UNB
$ACU $RMSPROP $CALTECH_UNB
$INC $NIL $CALTECH_UNB # NIL 1% (Scenario 0)
$INC $NIL $CALTECH_UNB -ts=1 # NIL 10% (Scenario 1)

for f in $_OUTPUT_PATH
do 
 _FULL_PATH="${f}/*"
 for ff in $_FULL_PATH
 do
  python utils/draw_tests.py -i=$ff -o="" -m="accuracy" -n="GRAPHIC"
 done
done

mv "../results/summaries/MNIST" "../results/summaries/MNIST_UNB"
mv "../results/summaries/FASHION-MNIST" "../results/summaries/FASHION-MNIST_UNB"
mv "../results/summaries/CIFAR-10" "../results/summaries/CIFAR-10_UNB"
mv "../results/summaries/CALTECH-101" "../results/summaries/CALTECH-101_UNB"