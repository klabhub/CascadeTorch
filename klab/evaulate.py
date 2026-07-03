#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Demo script to quantify and visualize the quality of a trained Cascade model.

The script evaluates a trained model on selected ground-truth datasets using the
same resampled, noise-matched path used in the benchmark example. It computes
correlation, error, and bias for each dataset, then generates summary plots and
example prediction traces.

"""


"""

Configure evaluation

"""

MODEL_NAME = "GC8_EXC_30Hz_smoothing25ms_high_noise"
MODEL_FOLDER = "Pretrained_models"
GROUND_TRUTH_FOLDER = "Ground_truth"

# Use dataset folder names under Ground_truth, or explicit paths to individual
# *_mini.mat files. Set to None to evaluate all datasets listed in the model's
# config.yaml file.
EVALUATION_DATASETS = None

# Use one of the model's trained noise levels. Set to None to use the first one.
NOISE_LEVEL = None

# Number of samples to visualize from the pooled evaluation sequence.
TRACE_PLOT_SAMPLES = 1500

# Output folder for the metrics table and figures.
OUTPUT_FOLDER = "model_evaluation"


"""

Import python packages

"""

import os
import shutil
import sys
import tempfile

if "Demo scripts" in os.getcwd():
    sys.path.append(os.path.abspath(".."))
    os.chdir("..")
print("Current working directory: {}".format(os.getcwd()))

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter

from cascade2p import checks

print("\nChecks for packages:")
checks.check_packages()

from cascade2p import cascade, config, utils


def sanitize_label(label):
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in label)


def resolve_evaluation_target(target_name, ground_truth_folder):
    if os.path.isfile(target_name):
        return dict(label=os.path.splitext(os.path.basename(target_name))[0], file_path=target_name)

    if target_name.endswith(".mat"):
        relative_file_path = os.path.join(ground_truth_folder, target_name)
        if os.path.isfile(relative_file_path):
            return dict(
                label=os.path.splitext(os.path.basename(relative_file_path))[0],
                file_path=relative_file_path,
            )

    dataset_folder = os.path.join(ground_truth_folder, target_name)
    if os.path.isdir(dataset_folder):
        return dict(label=target_name, folder_path=dataset_folder)

    raise FileNotFoundError(
        'Could not find evaluation target "{}" as a dataset folder or .mat file.'.format(target_name)
    )


def evaluate_dataset(model_name, cfg, target_name, noise_level, model_folder, ground_truth_folder):
    target = resolve_evaluation_target(target_name, ground_truth_folder)

    temporary_directory = None
    if "file_path" in target:
        temporary_directory = tempfile.TemporaryDirectory(prefix="cascade_eval_")
        copied_file_path = os.path.join(temporary_directory.name, os.path.basename(target["file_path"]))
        shutil.copy2(target["file_path"], copied_file_path)
        ground_truth_folders = [temporary_directory.name]
    else:
        ground_truth_folders = [target["folder_path"]]

    calcium_windows, ground_truth = utils.preprocess_groundtruth_artificial_noise_balanced(
        ground_truth_folders=ground_truth_folders,
        before_frac=cfg["before_frac"],
        windowsize=cfg["windowsize"],
        after_frac=1 - cfg["before_frac"],
        noise_level=noise_level,
        sampling_rate=cfg["sampling_rate"],
        smoothing=cfg["smoothing"] * cfg["sampling_rate"],
        omission_list=[],
        permute=0,
        verbose=cfg["verbose"],
        replicas=0,
        causal_kernel=cfg["causal_kernel"],
    )

    center_index = int(np.round(cfg["before_frac"] * cfg["windowsize"]))
    center_index = np.clip(center_index, 0, cfg["windowsize"] - 1)
    calcium_trace = calcium_windows[:, center_index, 0]

    predicted = cascade.predict(
        model_name,
        calcium_trace[None, :],
        model_folder=model_folder,
        verbosity=cfg["verbose"],
    )
    predicted = np.squeeze(predicted)
    ground_truth = np.squeeze(ground_truth)

    valid = ~np.isnan(predicted) & ~np.isnan(ground_truth)
    predicted = predicted[valid]
    ground_truth = ground_truth[valid]
    calcium_trace = calcium_trace[valid]

    smoothing_sigma = cfg["smoothing"] * cfg["sampling_rate"]
    predicted_smooth = gaussian_filter(predicted.astype(float), sigma=smoothing_sigma)
    ground_truth_smooth = gaussian_filter(ground_truth.astype(float), sigma=smoothing_sigma)

    error_diff = predicted_smooth - ground_truth_smooth
    signal = np.sum(np.abs(ground_truth_smooth))
    if signal == 0:
        error_value = np.nan
        bias_value = np.nan
    else:
        error_value = np.sum(np.abs(error_diff)) / signal
        bias_value = np.sum(error_diff) / signal

    if predicted.size < 2 or np.std(predicted) == 0 or np.std(ground_truth) == 0:
        correlation_value = np.nan
    else:
        correlation_value = np.corrcoef(ground_truth, predicted, rowvar=False)[0, 1]

    if temporary_directory is not None:
        temporary_directory.cleanup()

    return dict(
        dataset=target["label"],
        correlation=correlation_value,
        error=error_value,
        bias=bias_value,
        calcium=calcium_trace,
        ground_truth=ground_truth,
        prediction=predicted,
        ground_truth_smooth=ground_truth_smooth,
        prediction_smooth=predicted_smooth,
    )


def plot_metric_summary(results, output_folder):
    datasets = [result["dataset"] for result in results]
    correlation = [result["correlation"] for result in results]
    error = [result["error"] for result in results]
    bias = [result["bias"] for result in results]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    x = np.arange(len(datasets))

    axes[0].bar(x, correlation, color="#4C72B0")
    axes[0].set_ylabel("Correlation")
    axes[0].set_title("Model quality by evaluation dataset")

    axes[1].bar(x, error, color="#DD8452")
    axes[1].set_ylabel("Error")

    axes[2].bar(x, bias, color="#55A868")
    axes[2].set_ylabel("Bias")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(datasets, rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, "metric_summary.png"), dpi=200)


def plot_prediction_examples(result, sampling_rate, trace_plot_samples, output_folder):
    trace_plot_samples = min(trace_plot_samples, len(result["ground_truth"]))
    time = np.arange(trace_plot_samples) / sampling_rate

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(time, result["calcium"][:trace_plot_samples], color="#4C72B0")
    axes[0].set_ylabel("dF/F")
    axes[0].set_title("Example evaluation trace: {}".format(result["dataset"]))

    axes[1].plot(time, result["ground_truth_smooth"][:trace_plot_samples], label="Ground truth", color="#55A868")
    axes[1].plot(time, result["prediction_smooth"][:trace_plot_samples], label="Prediction", color="#C44E52", alpha=0.85)
    axes[1].set_ylabel("Smoothed rate")
    axes[1].legend(loc="upper right")

    residual = result["prediction_smooth"][:trace_plot_samples] - result["ground_truth_smooth"][:trace_plot_samples]
    axes[2].plot(time, residual, color="#8172B2")
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_ylabel("Residual")
    axes[2].set_xlabel("Time (s)")

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_folder, "example_trace_{}.png".format(sanitize_label(result["dataset"]))),
        dpi=200,
    )


def plot_prediction_scatter(results, output_folder):
    pooled_ground_truth = np.concatenate([result["ground_truth_smooth"] for result in results])
    pooled_prediction = np.concatenate([result["prediction_smooth"] for result in results])

    if pooled_ground_truth.size > 20000:
        subset = np.linspace(0, pooled_ground_truth.size - 1, 20000, dtype=int)
        pooled_ground_truth = pooled_ground_truth[subset]
        pooled_prediction = pooled_prediction[subset]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(pooled_ground_truth, pooled_prediction, s=6, alpha=0.25, color="#4C72B0", edgecolors="none")

    combined_min = np.nanmin([pooled_ground_truth.min(), pooled_prediction.min()])
    combined_max = np.nanmax([pooled_ground_truth.max(), pooled_prediction.max()])
    ax.plot([combined_min, combined_max], [combined_min, combined_max], color="black", linewidth=1)

    ax.set_xlabel("Ground truth (smoothed)")
    ax.set_ylabel("Prediction (smoothed)")
    ax.set_title("Prediction quality across all evaluated samples")

    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, "prediction_scatter.png"), dpi=200)


def save_metrics_table(results, output_folder):
    metrics_path = os.path.join(output_folder, "metrics_summary.csv")
    with open(metrics_path, "w") as handle:
        handle.write("dataset,correlation,error,bias\n")
        for result in results:
            handle.write(
                "{},{:.6f},{:.6f},{:.6f}\n".format(
                    result["dataset"],
                    result["correlation"],
                    result["error"],
                    result["bias"],
                )
            )


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    cfg_file = os.path.join(MODEL_FOLDER, MODEL_NAME, "config.yaml")
    cfg = config.read_config(cfg_file)

    evaluation_datasets = EVALUATION_DATASETS
    if evaluation_datasets is None:
        evaluation_datasets = list(cfg["training_datasets"])

    noise_level = NOISE_LEVEL
    if noise_level is None:
        noise_level = int(cfg["noise_levels"][0])

    print("\nEvaluating model: {}".format(MODEL_NAME))
    print("Evaluation datasets: {}".format(", ".join(evaluation_datasets)))
    print("Noise level used for evaluation: {}".format(noise_level))

    results = []
    for dataset_name in evaluation_datasets:
        print("\nProcessing dataset {}".format(dataset_name))
        result = evaluate_dataset(
            MODEL_NAME,
            cfg,
            dataset_name,
            noise_level,
            model_folder=MODEL_FOLDER,
            ground_truth_folder=GROUND_TRUTH_FOLDER,
        )
        results.append(result)
        print(
            "Correlation: {:.3f} | Error: {:.3f} | Bias: {:.3f}".format(
                result["correlation"], result["error"], result["bias"]
            )
        )

    save_metrics_table(results, OUTPUT_FOLDER)
    plot_metric_summary(results, OUTPUT_FOLDER)
    plot_prediction_scatter(results, OUTPUT_FOLDER)

    best_result = max(
        results,
        key=lambda result: -np.inf if np.isnan(result["correlation"]) else result["correlation"],
    )
    plot_prediction_examples(
        best_result,
        sampling_rate=cfg["sampling_rate"],
        trace_plot_samples=TRACE_PLOT_SAMPLES,
        output_folder=OUTPUT_FOLDER,
    )

    print("\nSaved evaluation outputs to {}".format(os.path.abspath(OUTPUT_FOLDER)))
    plt.show()


if __name__ == "__main__":
    main()