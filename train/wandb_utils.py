import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_WANDB_PROJECT = "data science practice proj"


def add_wandb_args(parser: argparse.ArgumentParser, default_job_type: str) -> None:
    parser.add_argument(
        "--wandb-mode",
        type=str,
        choices=("disabled", "offline", "online"),
        default="online",
        help="W&B logging mode. Use disabled to keep the current behavior.",
    )
    parser.add_argument("--wandb-project", type=str, default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-job-type", type=str, default=default_job_type)
    parser.add_argument(
        "--wandb-batch-name",
        type=str,
        default=None,
        help="Shared group name for a batch of runs. Auto-generated when omitted.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Explicit W&B run name. Auto-generated from job type, batch name, and seed when omitted.",
    )
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    parser.add_argument("--wandb-notes", type=str, default=None)
    parser.add_argument(
        "--wandb-log-every-steps",
        type=int,
        default=5,
        help="Log training metrics to W&B every N optimizer steps during train epochs.",
    )


def prepare_wandb_identity(args: argparse.Namespace, default_job_type: str) -> Dict[str, str]:
    job_type = getattr(args, "wandb_job_type", None) or default_job_type
    batch_name = args.wandb_batch_name or datetime.now().strftime(f"{job_type}-batch-%Y%m%d-%H%M%S")
    seed = getattr(args, "seed", None)
    run_name = args.wandb_run_name or _build_default_run_name(job_type=job_type, batch_name=batch_name, seed=seed)

    args.wandb_job_type = job_type
    args.wandb_batch_name = batch_name
    args.wandb_run_name = run_name
    return {
        "project": args.wandb_project,
        "job_type": job_type,
        "batch_name": batch_name,
        "run_name": run_name,
    }


def init_wandb_run(
    args: argparse.Namespace,
    *,
    default_job_type: str,
    config: Dict[str, Any],
    output_dir: Path,
):
    identity = prepare_wandb_identity(args, default_job_type=default_job_type)
    if args.wandb_mode == "disabled":
        return None, identity

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb logging was enabled, but the `wandb` package is not installed. "
            "Install it or pass --wandb-mode disabled."
        ) from exc

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        job_type=identity["job_type"],
        group=identity["batch_name"],
        name=identity["run_name"],
        tags=args.wandb_tags,
        notes=args.wandb_notes,
        mode=args.wandb_mode,
        dir=str(output_dir),
        config=_to_jsonable(config),
    )
    run.define_metric("step")
    run.define_metric("epoch")
    run.define_metric("step/*", step_metric="step")
    run.define_metric("epoch/*", step_metric="epoch")
    return run, identity


def update_wandb_summary(run, values: Dict[str, Any]) -> None:
    if run is None:
        return
    for key, value in values.items():
        run.summary[key] = value


def finish_wandb_run(run) -> None:
    if run is None:
        return
    run.finish()


def _build_default_run_name(job_type: str, batch_name: str, seed: Optional[int]) -> str:
    seed_suffix = f"-seed{seed}" if seed is not None else ""
    return f"{job_type}-{batch_name}{seed_suffix}"


def _to_jsonable(payload: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(payload, default=_json_default, ensure_ascii=False))


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value)
