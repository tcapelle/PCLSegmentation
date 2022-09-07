# ==============================================================================
# MIT License
#
# Copyright 2021 Institute for Automotive Engineering of RWTH Aachen University.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

import os.path

import argparse
import wandb

from data_loader import DataLoader
from utils.callbacks import TensorBoard
from utils.util import *
from utils.args_loader import load_model_config

from tensorflow.keras import mixed_precision
from wandb.keras import WandbCallback

def train(arg):
    
    config, model = load_model_config(args.model, args.config)
    wandb.init(project="small_kitti", entity="av-team", job_type="training", config=config)
    train = DataLoader("train", arg.data_path, config).write_tfrecord_dataset().read_tfrecord_dataset()
    val = DataLoader("val", arg.data_path, config).write_tfrecord_dataset().read_tfrecord_dataset()

    # tensorboard_callback = TensorBoard(arg.train_dir, val, profile_batch=(200, 202))
    # checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(os.path.join(arg.train_dir, "checkpoint"))
    wandb_callback = WandbCallback(save_model=False)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=config.LEARNING_RATE,
        decay_steps=config.LR_DECAY_STEPS,
        decay_rate=config.LR_DECAY_FACTOR,
        staircase=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=config.MAX_GRAD_NORM)

    model.compile(optimizer=optimizer, weighted_metrics=[])

    model.fit(train,
            validation_data=val,
            epochs=arg.epochs,
            callbacks=[wandb_callback],
            )
    
    #serialize model
    
    (lidar_input, lidar_mask), label, weight  = train.take(1).get_single_element()
    out = model([lidar_input, lidar_mask])

    model.save(filepath=os.path.join(arg.train_dir, arg.model), save_traces=True)
    wandb.finish()

if __name__ == '__main__':
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        # mixed_precision.set_global_policy('mixed_float16')

    parser = argparse.ArgumentParser(description='Parse Flags for the training script!')
    parser.add_argument('-d', '--data_path', type=str,
                      help='Absolute path to the dataset', default="/mnt/disks/KITTI/small")
    parser.add_argument('-e', '--epochs', type=int, default=10,
                      help='Maximal number of training epochs')
    parser.add_argument('-t', '--train_dir', type=str, default="/src/output",
                      help="Directory where to write the Tensorboard logs and checkpoints")
    parser.add_argument('-m', '--model', type=str, default="squeezesegv2",
                      help='Model name either `squeezesegv2`, `darknet53`, `darknet21`')
    parser.add_argument('-c', '--config', type=str, default="squeezesegv2kitti",
                      help='Which configuration for training `squeezesegv2`, `squeezesegv2kitti`, '
                           ' `darknet53`, `darknet21` ')
    args = parser.parse_args()

    train(args)
