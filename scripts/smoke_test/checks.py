"""
Expected-output manifest + lightweight content sniffs for the smoke test.
Checks "did the right files show up and are they non-empty/openable", not
"is the model actually good" -- this is a wiring test, not a quality test.
"""
import csv
import glob
import os
from dataclasses import dataclass

import numpy as np
import torch

NO_LAYER_IMPROVED_MSG = "No layer set improved on the initial validation accuracy"


@dataclass
class Expectation:
    stage: str
    pattern: str
    kind: str  # "pt" | "npy" | "csv" | "pdf"
    min_count: int = 1
    soft: bool = False  # True => missing file(s) is a WARN, not a FAIL


@dataclass
class CheckResult:
    stage: str
    target: str
    status: str  # "PASS" | "WARN" | "FAIL"
    detail: str


def _sniff_pt(path):
    try:
        state_dict = torch.load(path, weights_only=True, map_location="cpu")
        return len(state_dict) > 0
    except Exception:
        return False


def _sniff_npy(path):
    try:
        return np.load(path).size > 0
    except Exception:
        return False


def _sniff_csv(path):
    try:
        with open(path, newline="") as f:
            return next(csv.DictReader(f), None) is not None
    except Exception:
        return False


def _sniff_pdf(path):
    try:
        if os.path.getsize(path) == 0:
            return False
        with open(path, "rb") as f:
            return f.read(4) == b"%PDF"
    except Exception:
        return False


_SNIFFERS = {"pt": _sniff_pt, "npy": _sniff_npy, "csv": _sniff_csv, "pdf": _sniff_pdf}


def build_manifest(models_dir, files_dir, images_dir, suffix):
    m = []

    m.append(Expectation("baseline", os.path.join(models_dir, f"baseline_autoencoder_{suffix}.pt"), "pt"))
    m.append(Expectation("baseline", os.path.join(models_dir, f"baseline_unet_{suffix}.pt"), "pt"))
    m.append(Expectation("baseline", os.path.join(models_dir, f"baseline_classifier_{suffix}.pt"), "pt"))
    m.append(Expectation("baseline", os.path.join(models_dir, f"baseline_classifier_{suffix}_metrics.csv"), "csv"))
    m.append(Expectation("baseline", os.path.join(images_dir, f"baseline_autoencoder_examples_{suffix}.pdf"), "pdf"))

    m.append(Expectation("base_classifier", os.path.join(models_dir, f"base_classifier_{suffix}.pt"), "pt"))
    m.append(Expectation("base_classifier", os.path.join(models_dir, f"base_classifier_{suffix}_metrics.csv"), "csv"))

    m.append(Expectation("mixed_classifier", os.path.join(models_dir, f"mixed_classifier_{suffix}.pt"), "pt"))
    m.append(Expectation("mixed_classifier", os.path.join(models_dir, f"mixed_classifier_{suffix}_metrics.csv"), "csv"))

    m.append(Expectation("contrast_classifier", os.path.join(models_dir, f"contrast_body_{suffix}.pt"), "pt"))
    m.append(Expectation("contrast_classifier", os.path.join(models_dir, f"contrast_full_{suffix}.pt"), "pt"))
    m.append(Expectation("contrast_classifier", os.path.join(models_dir, f"contrast_full_{suffix}_metrics.csv"), "csv"))

    m.append(Expectation("pre_alignment_plots", os.path.join(images_dir, f"TSNE_{suffix}.pdf"), "pdf"))
    m.append(Expectation("pre_alignment_plots", os.path.join(images_dir, f"DIV_{suffix}.pdf"), "pdf"))

    for model_name in ("mixed", "contrast"):
        # per-layer sweep checkpoints always get written regardless of which
        # layer ends up "best" -- these are hard requirements
        m.append(Expectation(f"{model_name}_sweep_checkpoints",
                              os.path.join(models_dir, f"{model_name}_unet_FINAL_{suffix}-*.pt"), "pt"))
        m.append(Expectation(f"{model_name}_sweep_checkpoints",
                              os.path.join(models_dir, f"{model_name}_classifier_FINAL_{suffix}-*.pt"), "pt"))

        # the sweep *summary* outputs are only written if some layer set beat
        # the initial (pre-alignment) validation accuracy -- on tiny 1-epoch
        # synthetic data that's overwhelmingly likely but not guaranteed, so
        # these are soft (see NO_LAYER_IMPROVED_MSG handling in scan_log)
        m.append(Expectation(f"{model_name}_sweep_outputs",
                              os.path.join(files_dir, f"{model_name}-single_layer-divergence_plots.pdf"), "pdf", soft=True))
        m.append(Expectation(f"{model_name}_sweep_outputs",
                              os.path.join(files_dir, f"{model_name}-single_layer-accuracy_vs_divergence.pdf"), "pdf", soft=True))
        m.append(Expectation(f"{model_name}_sweep_outputs",
                              os.path.join(files_dir, f"{model_name}-single_layer-layer_summary.csv"), "csv", soft=True))

    m.append(Expectation("post_alignment_plots", os.path.join(images_dir, f"mixed_examples_{suffix}.pdf"), "pdf"))
    m.append(Expectation("post_alignment_plots", os.path.join(images_dir, f"contrast_examples_{suffix}.pdf"), "pdf"))
    m.append(Expectation("post_alignment_plots", os.path.join(images_dir, f"TSNE_UNET_{suffix}.pdf"), "pdf"))
    m.append(Expectation("post_alignment_plots", os.path.join(images_dir, f"DIV_UNET_{suffix}.pdf"), "pdf"))

    return m


def check_expectations(manifest):
    results = []
    for exp in manifest:
        matches = sorted(glob.glob(exp.pattern))
        if len(matches) < exp.min_count:
            status = "WARN" if exp.soft else "FAIL"
            results.append(CheckResult(exp.stage, exp.pattern, status,
                                        f"expected >= {exp.min_count} file(s), found {len(matches)}"))
            continue

        sniff = _SNIFFERS[exp.kind]
        bad = [m for m in matches if not sniff(m)]
        if bad:
            results.append(CheckResult(exp.stage, exp.pattern, "FAIL",
                                        f"{len(bad)}/{len(matches)} file(s) failed content sniff, e.g. {bad[0]}"))
        else:
            results.append(CheckResult(exp.stage, exp.pattern, "PASS", f"{len(matches)} file(s) OK"))
    return results


def scan_log(log_path):
    """Returns (unexpected_error_lines, no_layer_improved_count)."""
    errors = []
    no_layer_improved_count = 0
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if NO_LAYER_IMPROVED_MSG in line:
                no_layer_improved_count += 1
            elif " - ERROR - " in line:
                errors.append(line.rstrip())
    return errors, no_layer_improved_count


def evaluate(tmp_root, classifier_id, suffix, proc):
    output_root = os.path.join(tmp_root, classifier_id)
    models_dir = os.path.join(output_root, "models")
    files_dir = os.path.join(output_root, "files")
    images_dir = os.path.join(output_root, "images")
    logs_dir = os.path.join(output_root, "logs")

    results = []

    results.append(CheckResult("process", "exit_code",
                                "PASS" if proc.returncode == 0 else "FAIL",
                                f"exit code {proc.returncode}"))

    combined_output = (proc.stdout or "") + (proc.stderr or "")
    if "Warning: plot could not be saved" in combined_output:
        results.append(CheckResult("process", "stdout/stderr", "FAIL",
                                    "a plot failed to save (plot_ebsw swallows this as a bare print, "
                                    "not visible via log/exit-code otherwise)"))

    log_files = sorted(glob.glob(os.path.join(logs_dir, "*.log")))
    no_layer_improved_count = 0
    if not log_files:
        results.append(CheckResult("logs", logs_dir, "FAIL", "no log file found"))
    else:
        errors, no_layer_improved_count = scan_log(log_files[0])
        if errors:
            for line in errors:
                results.append(CheckResult("logs", log_files[0], "FAIL", line))
        else:
            results.append(CheckResult("logs", log_files[0], "PASS", "no unexpected ERROR lines"))

    manifest = build_manifest(models_dir, files_dir, images_dir, suffix)
    results.extend(check_expectations(manifest))

    if no_layer_improved_count:
        results.append(CheckResult(
            "sweep_outputs", "-", "WARN",
            f"'{NO_LAYER_IMPROVED_MSG}' logged {no_layer_improved_count}x -- expected occasionally on "
            f"tiny/1-epoch synthetic data; that model's sweep-summary files are optional this run",
        ))

    return results
