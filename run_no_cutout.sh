#! /bin/bash
export CUDA_VISIBLE_DEVICES="0"
nohup python -u train_cutout.py --dataset cifar10 --model resnet18 --data_augmentation --length 16 > logs/log_train_no_cutout.txt &