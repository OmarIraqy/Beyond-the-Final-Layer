"""Analyze per-class image-size distributions across Urban ReID splits."""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
from PIL import Image
from omegaconf import OmegaConf

# Add project root to path for direct script execution.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import load_config
from src.data.dataset import IDX_TO_CLASS, UrbanReIDDataset


PERCENTILES = [5, 25, 50, 75, 90, 95]


def _percentile_dict(values):
    return {
        str(percentile): round(float(np.percentile(values, percentile)), 2)
        for percentile in PERCENTILES
    }


def _compute_padding_fraction(widths, heights, canvas_hw):
    target_h, target_w = int(canvas_hw[0]), int(canvas_hw[1])
    padding_hits = 0
    for width, height in zip(widths, heights):
        scale = min(target_w / width, target_h / height)
        resized_w = max(1, int(round(width * scale)))
        resized_h = max(1, int(round(height * scale)))
        if resized_h < target_h or resized_w < target_w:
            padding_hits += 1
    return padding_hits / len(widths) if widths else 0.0


def _raw_size_map(samples):
    grouped = defaultdict(lambda: {"widths": [], "heights": []})
    for sample in samples:
        class_name = IDX_TO_CLASS.get(sample.class_label, "Unknown")
        with Image.open(sample.img_path) as image:
            width, height = image.size
        grouped[class_name]["widths"].append(width)
        grouped[class_name]["heights"].append(height)
    return grouped


def _summarize_group(widths, heights, canvas_hw):
    widths_array = np.asarray(widths, dtype=np.float32)
    heights_array = np.asarray(heights, dtype=np.float32)
    aspect_ratios = widths_array / heights_array
    short_sides = np.minimum(widths_array, heights_array)
    long_sides = np.maximum(widths_array, heights_array)

    return {
        "count": int(len(widths)),
        "canvas_hw": [int(canvas_hw[0]), int(canvas_hw[1])],
        "width": _percentile_dict(widths_array),
        "height": _percentile_dict(heights_array),
        "aspect_ratio": _percentile_dict(aspect_ratios),
        "short_side": _percentile_dict(short_sides),
        "long_side": _percentile_dict(long_sides),
        "padding_fraction": round(_compute_padding_fraction(widths, heights, canvas_hw), 4),
        "raw_widths": list(widths),
        "raw_heights": list(heights),
    }


def _summarize_sizes(raw_sizes, class_sizes):
    return {
        class_name: _summarize_group(
            values["widths"],
            values["heights"],
            class_sizes.get(class_name, [0, 0]),
        )
        for class_name, values in raw_sizes.items()
    }


def _strip_raw(summary):
    return {
        class_name: {
            key: value
            for key, value in class_stats.items()
            if key not in {"raw_widths", "raw_heights"}
        }
        for class_name, class_stats in summary.items()
    }


def _build_markdown_summary(overall_summary, train_class_sizes, test_class_sizes):
    lines = [
        "# Class Size Summary",
        "",
        "| Class | Count | Median HxW | AR p50 | Train Canvas | Train Pad Frac | Test Canvas | Test Pad Frac |",
        "| --- | ---: | --- | ---: | --- | ---: | --- | ---: |",
    ]

    for class_name in sorted(overall_summary.keys()):
        stats = overall_summary[class_name]
        median_hw = f"{int(round(stats['height']['50']))}x{int(round(stats['width']['50']))}"
        train_canvas = train_class_sizes.get(class_name, [0, 0])
        test_canvas = test_class_sizes.get(class_name, [0, 0])
        train_pad_frac = _compute_padding_fraction(stats["raw_widths"], stats["raw_heights"], train_canvas)
        test_pad_frac = _compute_padding_fraction(stats["raw_widths"], stats["raw_heights"], test_canvas)
        lines.append(
            f"| {class_name} | {stats['count']} | {median_hw} | {stats['aspect_ratio']['50']:.2f} | "
            f"{train_canvas[0]}x{train_canvas[1]} | {train_pad_frac:.2%} | "
            f"{test_canvas[0]}x{test_canvas[1]} | {test_pad_frac:.2%} |"
        )

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml", help="Config file used to locate the dataset")
    parser.add_argument("--output-dir", default="outputs/size_stats", help="Directory for JSON and markdown outputs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset = UrbanReIDDataset(cfg)
    splits = {
        "train": dataset.train,
        "query": dataset.query,
        "gallery": dataset.gallery,
    }

    split_summaries = {}
    overall_sizes = defaultdict(lambda: {"widths": [], "heights": []})
    for split_name, samples in splits.items():
        class_sizes = cfg.input.class_sizes_train if split_name == "train" else cfg.input.class_sizes_test
        raw_sizes = _raw_size_map(samples)
        split_summaries[split_name] = _summarize_sizes(raw_sizes, class_sizes)
        for class_name, values in raw_sizes.items():
            overall_sizes[class_name]["widths"].extend(values["widths"])
            overall_sizes[class_name]["heights"].extend(values["heights"])

    overall_summary = _summarize_sizes(overall_sizes, cfg.input.class_sizes_train)
    train_class_sizes = OmegaConf.to_container(cfg.input.class_sizes_train, resolve=True)
    test_class_sizes = OmegaConf.to_container(cfg.input.class_sizes_test, resolve=True)
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "dataset_root": cfg.dataset.root,
        "train_class_sizes": train_class_sizes,
        "test_class_sizes": test_class_sizes,
        "splits": {split_name: _strip_raw(summary) for split_name, summary in split_summaries.items()},
        "overall": _strip_raw(overall_summary),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "class_size_stats.json")
    with open(json_path, "w") as handle:
        json.dump(payload, handle, indent=2)

    markdown_path = os.path.join(args.output_dir, "class_size_stats.md")
    with open(markdown_path, "w") as handle:
        handle.write(_build_markdown_summary(overall_summary, train_class_sizes, test_class_sizes))

    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()