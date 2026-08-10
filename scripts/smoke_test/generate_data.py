"""
Generates a small synthetic two-domain image dataset on disk, in the same flat
`<domain>/<class_name>/<files>` layout the real pipeline expects (see
datasets.CombinedDataset / helpers.build_splits). Images are procedurally
drawn shapes -- real, valid, openable image files, just synthetic/tiny-scale
content, so the pipeline exercises its actual image-loading/transform code
path without needing real photographs.
"""
import math
import os
import random
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import helpers  # noqa: E402  (repo root import, needs sys.path set up above)

CLASS_NAMES = [
    "circle", "square", "triangle", "star", "cross",
    "diamond", "hexagon", "pentagon", "arrow", "heart",
]

_COLORS = [
    (220, 20, 60), (30, 144, 255), (60, 179, 113), (255, 165, 0), (147, 112, 219),
    (255, 105, 180), (0, 206, 209), (218, 165, 32), (139, 69, 19), (105, 105, 105),
]

IMAGE_SIZE = 64


def _regular_polygon(center, radius, sides, rotation=0.0):
    cx, cy = center
    return [
        (cx + radius * math.cos(rotation + i * (2 * math.pi / sides)),
         cy + radius * math.sin(rotation + i * (2 * math.pi / sides)))
        for i in range(sides)
    ]


def _draw_class_shape(draw, class_name, color, rng):
    cx = IMAGE_SIZE // 2 + rng.randint(-6, 6)
    cy = IMAGE_SIZE // 2 + rng.randint(-6, 6)
    r = rng.randint(16, 22)

    if class_name == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif class_name == "square":
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif class_name == "triangle":
        draw.polygon(_regular_polygon((cx, cy), r, 3, rng.uniform(0, 2 * math.pi)), fill=color)
    elif class_name == "star":
        outer = _regular_polygon((cx, cy), r, 5, -math.pi / 2)
        inner = _regular_polygon((cx, cy), r * 0.45, 5, -math.pi / 2 + math.pi / 5)
        points = [p for pair in zip(outer, inner) for p in pair]
        draw.polygon(points, fill=color)
    elif class_name == "cross":
        w = max(4, r // 3)
        draw.rectangle([cx - r, cy - w, cx + r, cy + w], fill=color)
        draw.rectangle([cx - w, cy - r, cx + w, cy + r], fill=color)
    elif class_name == "diamond":
        draw.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], fill=color)
    elif class_name == "hexagon":
        draw.polygon(_regular_polygon((cx, cy), r, 6), fill=color)
    elif class_name == "pentagon":
        draw.polygon(_regular_polygon((cx, cy), r, 5, -math.pi / 2), fill=color)
    elif class_name == "arrow":
        draw.polygon([
            (cx - r, cy - r // 3), (cx + r // 3, cy - r // 3), (cx + r // 3, cy - r),
            (cx + r, cy), (cx + r // 3, cy + r), (cx + r // 3, cy + r // 3),
            (cx - r, cy + r // 3),
        ], fill=color)
    elif class_name == "heart":
        draw.pieslice([cx - r, cy - r, cx, cy], 180, 360, fill=color)
        draw.pieslice([cx, cy - r, cx + r, cy], 180, 360, fill=color)
        draw.polygon([(cx - r, cy - r * 0.15), (cx + r, cy - r * 0.15), (cx, cy + r)], fill=color)
    else:
        raise ValueError(f"Unknown class name: {class_name}")


def _make_image(class_idx, class_name, seed, domain_style):
    rng = random.Random(seed)
    bg = tuple(rng.randint(230, 255) for _ in range(3))
    img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=bg)
    draw = ImageDraw.Draw(img)

    base_color = _COLORS[class_idx]
    # source domain gets a wider color jitter + a rotation, as a mild stand-in
    # for a real cross-domain shift; target stays "cleaner"
    jitter = 15 if domain_style == "target" else 35
    color = tuple(max(0, min(255, c + rng.randint(-jitter, jitter))) for c in base_color)
    _draw_class_shape(draw, class_name, color, rng)

    if domain_style == "source":
        img = img.rotate(rng.randint(-20, 20), fillcolor=bg)

    return img


def _generate_domain(root, domain_name, samples_per_class, seed):
    domain_dir = os.path.join(root, "data", domain_name)
    os.makedirs(domain_dir, exist_ok=True)
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(domain_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        for i in range(samples_per_class):
            sample_seed = seed * 100_000 + class_idx * 1_000 + i
            img = _make_image(class_idx, class_name, sample_seed, domain_name)
            img.save(os.path.join(class_dir, f"{domain_name}_{class_name}_{i}.png"))
    return domain_dir


def _missing_classes(paths):
    seen = {os.path.basename(os.path.dirname(p)) for p in paths}
    return set(CLASS_NAMES) - seen


def build_dataset(root, samples_per_class=12, seed=72, train_pct=0.6, val_pct=0.2, max_attempts=5):
    """
    Generates target+source synthetic domains under `root`, and pre-flight
    checks that every class survives the same stratified train/val/test split
    helpers.build_splits will perform at pipeline runtime -- retrying with a
    bumped seed if a class comes up empty in any split (small sample counts +
    stratified splitting can occasionally strand a class at 0 in one split).
    """
    split_pcts = [train_pct, val_pct, 1 - train_pct - val_pct]
    missing = set()

    for attempt in range(max_attempts):
        attempt_seed = seed + attempt
        target_dir = _generate_domain(root, "target", samples_per_class, attempt_seed)
        source_dir = _generate_domain(root, "source", samples_per_class, attempt_seed + 1)

        target_splits = helpers.build_splits(folder=target_dir, split_pcts=split_pcts, seed=attempt_seed)
        source_splits = helpers.build_splits(folder=source_dir, split_pcts=split_pcts, seed=attempt_seed)

        missing = set()
        for split in (*target_splits, *source_splits):
            missing |= _missing_classes(split)

        if not missing:
            return target_dir, source_dir

        shutil.rmtree(target_dir, ignore_errors=True)
        shutil.rmtree(source_dir, ignore_errors=True)

    raise RuntimeError(
        f"Could not generate a synthetic dataset where every class survives train/val/test "
        f"stratified splitting after {max_attempts} attempts (missing classes: {sorted(missing)}). "
        f"Try a larger --samples-per-class."
    )
