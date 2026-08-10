"""
Smoke-tests the FeatureAlignment training pipeline against a small synthetic
dataset. Generates data + a matching config, runs the real `main.py` entry
point via subprocess (with every epoch count capped so it finishes in
seconds instead of the ~4000 tiny-epoch runtime the pipeline defaults to),
then checks that the expected checkpoints/plots/metrics landed and no
unexpected errors were logged.

Usage:
    python scripts/smoke_test/run_smoke_test.py [--keep] [--epoch-cap N] [--samples-per-class N]

Run with the SAME Python interpreter/venv you use for real experiments --
`main.py` is invoked via `sys.executable`, so whatever env runs this script
is the env the pipeline actually runs under.
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_config  # noqa: E402
import generate_data  # noqa: E402
import checks  # noqa: E402

import helpers  # noqa: E402
import type_defs  # noqa: E402


def run_schema_drift_check():
    """
    Bonus check: do the checked-in configs/*.yaml files still validate
    against the current type_defs schema? (Several didn't, as of the
    investigation that led to this script -- kept here as a standing
    regression guard, informational rather than pass/fail-gating.)

    META_config.yaml is the one meta-config consumed by run_main.py, so it's
    validated against type_defs.MetaConfig; every other configs/*.yaml file
    is a single-experiment config consumed by main.py, validated against
    type_defs.Config.
    """
    print("\n=== Config schema-drift check (configs/*.yaml vs current type_defs schemas) ===")
    config_dir = REPO_ROOT / "configs"
    any_checked = False
    for path in sorted(config_dir.glob("*.yaml")):
        any_checked = True
        schema = type_defs.MetaConfig if path.name == "META_config.yaml" else type_defs.Config
        try:
            data = helpers.load_yaml(str(path))
            schema(**data)
            print(f"  [PASS] {path.name} (as {schema.__name__})")
        except Exception as e:
            detail = str(e).splitlines()[0]
            print(f"  [FAIL] {path.name} (as {schema.__name__}) -- {detail}")
    if not any_checked:
        print("  (no configs/*.yaml files found)")


def run_pipeline(config_path, tmp_root, epoch_cap, device, timeout):
    import os

    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    env["FEATUREALIGNMENT_EPOCH_CAP"] = str(epoch_cap)
    if device != "auto":
        env["FEATUREALIGNMENT_FORCE_DEVICE"] = device

    cmd = [sys.executable, str(REPO_ROOT / "main.py"), "--config_fname", str(config_path)]
    print(f"\n=== Running: {' '.join(cmd)}")
    print(f"    cwd={tmp_root}")
    return subprocess.run(cmd, cwd=str(tmp_root), env=env, capture_output=True, text=True, timeout=timeout)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--keep", action="store_true", help="Keep the generated tmp directory instead of deleting it.")
    parser.add_argument("--epoch-cap", type=int, default=1, help="Cap every epoch count in the pipeline (default: 1).")
    parser.add_argument("--samples-per-class", type=int, default=12, help="Synthetic images per class per domain (default: 12).")
    parser.add_argument("--loss", choices=["ebsw", "mmdfuse"], default="ebsw", help="unet.loss to test (default: ebsw).")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu",
                         help="Device to run on (default: cpu -- GPU dispatch overhead dominates on data this "
                              "tiny, so CPU is usually faster here; use 'auto' to exercise the real device-selection logic).")
    parser.add_argument("--timeout", type=int, default=900,
                         help="Subprocess timeout in seconds -- a safety net against a genuine hang, "
                              "not the primary speed control (default: 900).")
    parser.add_argument("--skip-schema-check", action="store_true", help="Skip the configs/*.yaml schema-drift check.")
    args = parser.parse_args()

    if not args.skip_schema_check:
        run_schema_drift_check()

    tmp_root = Path(tempfile.mkdtemp(prefix="featurealignment_smoke_"))
    print(f"\nWorking directory: {tmp_root}")

    exit_code = 1
    try:
        target_dir, source_dir = generate_data.build_dataset(str(tmp_root), samples_per_class=args.samples_per_class)
        print(f"Generated synthetic data: target={target_dir}, source={source_dir}")

        config_path, meta = generate_config.build_config(
            target_dir=target_dir,
            source_dir=source_dir,
            output_dir=str(tmp_root / "configs"),
            samples_per_class=args.samples_per_class,
            loss=args.loss,
        )
        print(f"Generated config: {config_path}")

        try:
            proc = run_pipeline(config_path, tmp_root, args.epoch_cap, args.device, args.timeout)
        except subprocess.TimeoutExpired as e:
            print(f"\nFAIL -- pipeline did not finish within {args.timeout}s (treat as a hang, not a slow-but-working run).")
            if e.stdout:
                print("--- stdout (tail) ---")
                print("\n".join(e.stdout.splitlines()[-40:]) if isinstance(e.stdout, str) else e.stdout[-4000:])
            sys.exit(1)

        print(f"\nExit code: {proc.returncode}")
        if proc.returncode != 0:
            print("--- stdout (tail) ---")
            print("\n".join(proc.stdout.splitlines()[-40:]))
            print("--- stderr (tail) ---")
            print("\n".join(proc.stderr.splitlines()[-40:]))

        results = checks.evaluate(
            tmp_root=str(tmp_root),
            classifier_id=meta["classifier_id"],
            suffix=meta["suffix"],
            proc=proc,
        )

        print("\n=== Stage results ===")
        for r in results:
            print(f"  [{r.status}] {r.stage}: {r.detail}")

        hard_fail = proc.returncode != 0 or any(r.status == "FAIL" for r in results)
        exit_code = 1 if hard_fail else 0
        print(f"\n{'FAIL' if hard_fail else 'PASS'} -- smoke test {'failed' if hard_fail else 'passed'}.")

    finally:
        if args.keep:
            print(f"\n--keep set: leaving generated files at {tmp_root}")
        else:
            shutil.rmtree(tmp_root, ignore_errors=True)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
