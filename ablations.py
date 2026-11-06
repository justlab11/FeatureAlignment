"""
Ablations run on top of the normal pipeline. Each entry is a set of config overrides that run_main.py applies to the
generated run config for every combination in the meta config (`ablations: [normal, unfreeze_body, ...]`).

To add one: add the config field it needs in type_defs.py and the code that reads it, then an entry below.
"""
from copy import deepcopy

ABLATIONS: dict[str, dict] = {
    # the method as described: classifier body frozen, align/classify repeated until it stops helping
    "normal": {},
    # head fine-tuning also trains the body, at body_lr_scale x the classifier lr (BatchNorm stats stay fixed)
    "unfreeze_body": {"unet": {"body_lr_scale": 0.01}},
    # everything frozen, one align + head-fit pass per span (no inner loop)
    "no_inner_loop": {"unet": {"max_inner_iterations": 1}},
    # random shifts / crops on the training images (all three classifiers are retrained with them, so this run also
    # moves the base / mixed / contrast baselines); see transforms.train_augmentations
    "augment": {"dataset": {"augment": "geometric"}},
    # both of the above: body trained at a low lr, one align + head-fit pass per span
    "unfreeze_no_inner": {"unet": {"body_lr_scale": 0.01, "max_inner_iterations": 1}},
}

def validate(names: list[str]) -> None:
    unknown = [n for n in names if n not in ABLATIONS]
    if unknown:
        raise ValueError(f"Unknown ablation(s) {unknown}; choose from {list(ABLATIONS)}")

def _merge(base: dict, overrides: dict) -> dict:
    for key, value in overrides.items():
        if isinstance(value, dict):
            base[key] = _merge(base.get(key, {}), value)
        else:
            base[key] = value
    return base

def apply(config: dict, name: str) -> dict:
    """Returns a copy of the run config (as a dict) with the ablation's overrides applied."""
    validate([name])
    out = _merge(deepcopy(config), ABLATIONS[name])
    out["ablation"] = name
    return out
