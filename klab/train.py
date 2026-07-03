#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Script to train Cascade networks for KLab

The model configuration is defined based on a couple of parameters (sampling rate,
training data sets, noise levels, ground truth smoothing). A folder is generated
on the hard disk with the name 'model_name'.

Finally, the model is trained, "cascade.train_model( model_name )" and the trained
models are saved to disk.

"""


"""

Configure model and its parameters

"""

MODEL_FOLDER = "Pretrained_models"

cfgs = [
    dict(
        model_name="GC8+_INH_15Hz_smoothing100ms_high_noise",  # Model name (and name of the save folder)
        sampling_rate=15,  # Sampling rate in Hz (round to next integer)
        training_datasets=[
            "DS33-Interneurons2023-m-V1"
        ],
        noise_levels=[
            noise for noise in range(1, 10)
        ],  # int values of noise values (do not use numpy here => representer error!)
        smoothing=0.1,  # std of Gaussian smoothing in time (sec)
        causal_kernel=0,  # causal ground truth smoothing kernel
        # Advanced:
        # For additional parameters, you can find their names in the cascade2p/config.py
        # file in the config_template string
    )
]


"""

Import python packages

"""

import os, sys
# Suppress all traceback information
sys.tracebacklimit = 1
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
if os.getcwd() != PROJECT_ROOT:
    os.chdir(PROJECT_ROOT)
print('Current working directory: {}'.format( os.getcwd() ))
# perform checks to catch most likly import errors
from cascade2p import checks  # TODO: put all of this in one function

print("\nChecks for packages:")
checks.check_packages()

from cascade2p import cascade, config


def get_expected_model_files(model_config):
    return [
        "Model_NoiseLevel_{}_Ensemble_{}.pth".format(int(noise_level), ensemble)
        for noise_level in model_config["noise_levels"]
        for ensemble in range(model_config["ensemble_size"])
    ]


def model_training_finished(model_path):
    config_path = os.path.join(model_path, "config.yaml")

    if not os.path.isfile(config_path):
        return False, "config.yaml is missing"

    model_config = config.read_config(config_path)
    missing_model_files = [
        file_name
        for file_name in get_expected_model_files(model_config)
        if not os.path.isfile(os.path.join(model_path, file_name))
    ]

    if missing_model_files:
        return False, "{} trained model files are missing".format(
            len(missing_model_files)
        )

    training_finished = str(model_config.get("training_finished", "")).strip().lower()

    if training_finished != "yes":
        return False, 'training_finished is "{}" but all trained model files exist'.format(
            model_config.get("training_finished", "")
        )

    return True, "training_finished is Yes and all trained model files exist"


"""

Generate folders and train models

"""

for cfg in cfgs:
    model_name = cfg["model_name"]
    model_path = os.path.join(MODEL_FOLDER, model_name)
    config_path = os.path.join(model_path, "config.yaml")
    should_skip, skip_reason = model_training_finished(model_path)

    if should_skip:
        print(
            '\nSkipping model "{}" because {} in {}.'.format(
                model_name, skip_reason, os.path.abspath(config_path)
            )
        )
        continue

    if os.path.isdir(model_path):
        print(
            '\nTraining model "{}" because {}.'.format(model_name, skip_reason)
        )

    if not os.path.isdir(model_path):
        cascade.create_model_folder(cfg, model_folder=MODEL_FOLDER)
    elif not os.path.isfile(config_path):
        config.write_config(cfg, config_path)
        print(
            '\nCreated missing config for model "{}" at {}.'.format(
                model_name, os.path.abspath(model_path)
            )
        )

    print('\nTo load this model, use the model name "{}"'.format(model_name))

    cascade.train_model(model_name, model_folder=MODEL_FOLDER)
