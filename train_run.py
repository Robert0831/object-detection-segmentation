import torch
import argparse
import os
ROOT = os.getcwd()
import sys
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import config



def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='YOLOv6 PyTorch Training', add_help=add_help)
    parser.add_argument('--data-path', default='./VOC', type=str, help='path of dataset')
    parser.add_argument('--epoch', default=200, type=int, help='epoch num')
    parser.add_argument('--batch', default=16, type=int, help='batch num')
    parser.add_argument('--load', default="", type=str, help='load model')
    parser.add_argument('--trainwhat', default="bac_obj", type=str, help='train "seg"  or "bac_obj" or "fix_bac"' )
    parser.add_argument('--fixbac', default="n", type=str, help='fix bacbone, "tt"' )

    return parser

def chang_conf(args):
    config.NUM_EPOCHS=args.epoch
    config.BATCH_SIZE=args.batch
    config.DATASET=args.data_path
    if args.load!="":
        config.LOAD_MODEL=True
        config.CHECKPOINT_FILE=args.load
    config.FIX_BAC=args.fixbac
    config.TRAIN_WHAT=args.trainwhat
    config.IMG_DIR=args.data_path +"/images/"
    config.LABEL_DIR=args.data_path +"/labels/"

import train
if __name__ == '__main__':

    args = get_args_parser().parse_args()
    chang_conf(args)

    train.main()



    # print(config.BATCH_SIZE)
    # print(config.NUM_EPOCHS)

