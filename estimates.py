import logging
from typing import Callable, List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

BYTES = 4  # fp32
OVERHEAD = 1.2  # allocator fragmentation, cuDNN workspace and anything the counts below miss
CUDA_CONTEXT_BYTES = 500 * 1024**2  # per-process CUDA context; not part of torch's allocated memory
NUM_PROJECTIONS = 256  # ISEBSW default in losses.py


def _fmt(num_bytes: float) -> str:
    return f"{num_bytes / 1024**2:,.1f} MiB"


def _count(params) -> int:
    return sum(p.numel() for p in params)


def _saved_bytes_per_sample(model: nn.Module, x: torch.Tensor, params_trainable: bool) -> int:
    """
    Bytes autograd keeps for the backward pass of one forward, per sample. Every tensor the graph saves is counted
    once (views of the same memory are deduplicated), excluding the weights themselves.

    params_trainable=True gives the training case; False keeps the weights fixed but still lets gradients flow to the
    input (the frozen classifier during alignment), which saves fewer activations.
    """
    weight_ptrs = {p.data_ptr() for p in model.parameters()}
    saved = {}

    def pack(t):
        if t.data_ptr() not in weight_ptrs:
            saved[(t.untyped_storage().data_ptr(), t.storage_offset(), t.numel())] = t.numel() * t.element_size()
        return t

    previous = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(params_trainable)

    x = x.clone().requires_grad_(not params_trainable)
    model.train()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = model(x)
    del out

    for p, flag in zip(model.parameters(), previous):
        p.requires_grad_(flag)

    return sum(saved.values()) // x.size(0)


def _rep_numels(outputs: List[torch.Tensor]) -> List[int]:
    """Elements per sample of every representation a model returns."""
    return [int(o[0].numel()) for o in outputs]


def run_estimates(
    classifier_factory: Callable[[], nn.Module],
    unet_factory: Callable[[], nn.Module],
    proj_head: nn.Module,
    sample: torch.Tensor,
    batch_size: int,
) -> dict:
    """
    Passes a sample image through fresh copies of the classifier and the UNet, then logs their sizes and an
    estimate of the GPU memory each training stage needs. Nothing is trained and the real models are untouched.
    """
    B = batch_size
    sample = sample[:2].detach().cpu()
    if sample.size(0) < 2:
        sample = sample.repeat(2, 1, 1, 1)

    logger.info("Running image sample through classifier...")
    classifier = classifier_factory()
    with torch.no_grad():
        classifier.eval()
        cls_reps = _rep_numels(classifier(sample))
    cls_saved_train = _saved_bytes_per_sample(classifier, sample, True)
    cls_saved_frozen = _saved_bytes_per_sample(classifier, sample, False)

    logger.info("Running image sample through unet...")
    unet = unet_factory()
    with torch.no_grad():
        unet.eval()
        unet_reps = _rep_numels(unet(sample))
    unet_saved = _saved_bytes_per_sample(unet, sample, True)

    # parameter counts
    P_c = _count(classifier.parameters())
    P_head = _count(classifier.model.fc.parameters()) if hasattr(classifier, "model") else 0
    P_body = P_c - P_head
    P_u = _count(unet.parameters())
    P_proj = _count(proj_head.parameters())

    input_numel = int(sample[0].numel())
    H_c = sum(cls_reps)  # all representations the classifier hooks hold on to
    H_u = sum(unet_reps)

    # ------------------------------------------------------------------ stage estimates (bytes)
    def weights(p):
        return BYTES * p

    def sgd(p):  # weights + gradients + momentum
        return 3 * BYTES * p

    def adam(p):  # weights + gradients + two moments
        return 4 * BYTES * p

    stages = {}

    # Base classifier: target samples only
    stages["Base classifier training"] = (
        sgd(P_c) + B * (cls_saved_train + BYTES * input_numel)
    )
    # Mixed: target + source in the same step
    stages["Mixed classifier training"] = (
        sgd(P_c) + 2 * B * (cls_saved_train + BYTES * input_numel)
    )
    # Contrast: body trainable, projection head, Adam, 2B images
    stages["Contrast classifier training"] = (
        adam(P_c) + adam(P_proj) + 2 * B * (cls_saved_train + BYTES * input_numel)
        + 4 * BYTES * (2 * B) ** 2  # similarity / mask matrices of the contrastive loss
    )
    # Head fine-tuning with a frozen body: no activations saved for the body, hooks hold the representations
    stages["Head fine-tuning (frozen body)"] = (
        weights(P_body) + sgd(P_head) + 2 * B * BYTES * H_c
    )
    # Same, inside the cascade: the (frozen) UNet runs first on each batch
    stages["Head fine-tuning in cascade (UNet + frozen body)"] = (
        weights(P_u) + stages["Head fine-tuning (frozen body)"] + 2 * B * BYTES * H_u
    )

    # Alignment: UNet trainable, classifier fixed but gradients flow through it to the UNet
    def alignment(span_layers: List[int]) -> float:
        loss_terms = NUM_PROJECTIONS * BYTES * sum(cls_reps[l] for l in span_layers)  # random projections, kept for backward
        return (
            adam(P_u) + weights(P_c)
            + B * unet_saved
            + B * (cls_saved_frozen + BYTES * input_numel)  # classifier on the aligned source batch
            + B * BYTES * H_c  # target pass under no_grad, hooks keep every representation
            + loss_terms
        )

    num_layers = len(cls_reps) - 1  # the cascade covers the input and the layers before the logits
    stages[f"Alignment training, widest span (layers 0-{num_layers - 1})"] = alignment(list(range(num_layers)))
    stages[f"Alignment training, narrowest span (layer {num_layers - 1})"] = alignment([num_layers - 1])

    # Divergence measurement after each span: forward only, widest span's projections
    stages["Divergence measurement"] = (
        weights(P_c) + weights(P_u) + B * BYTES * (H_c + H_u)
        + NUM_PROJECTIONS * BYTES * sum(cls_reps[:num_layers])
    )

    scaled = {name: OVERHEAD * b for name, b in stages.items()}
    peak_name = max(scaled, key=scaled.get)
    process_bytes = scaled[peak_name] + CUDA_CONTEXT_BYTES

    # ------------------------------------------------------------------ log
    logger.info("")
    logger.info("Model sizes")
    logger.info(f"\tInput shape (per sample): {tuple(sample.shape[1:])} = {input_numel:,} elements")
    logger.info(f"\tClassifier parameters: {P_c:,} (body {P_body:,}, head {P_head:,})")
    logger.info(f"\tUNet parameters: {P_u:,}")
    logger.info(f"\tProjection head parameters (contrastive): {P_proj:,}")
    logger.info(f"\tClassifier representation sizes (elements per sample, layers 0-{len(cls_reps) - 1}): {cls_reps}")
    logger.info(f"\tUNet representation sizes (elements per sample): {unet_reps}")
    logger.info(f"\tSaved for backward per sample - classifier trainable: {_fmt(cls_saved_train)}, "
                f"classifier frozen (alignment): {_fmt(cls_saved_frozen)}, UNet: {_fmt(unet_saved)}")

    logger.info("")
    logger.info(f"Estimated GPU memory per stage (batch size {B}; Mixed/Contrast use 2x{B} images per step; "
                f"x{OVERHEAD} overhead)")
    for name, b in scaled.items():
        marker = "  <- peak" if name == peak_name else ""
        logger.info(f"\t{name}: {_fmt(b)}{marker}")
    logger.info(f"\tEstimated per-process total (peak + {_fmt(CUDA_CONTEXT_BYTES)} CUDA context): {_fmt(process_bytes)}")

    logger.info("")

    return {
        "batch_size": B,
        "input_numel": input_numel,
        "parameters": {"classifier": P_c, "classifier_body": P_body, "classifier_head": P_head,
                       "unet": P_u, "projection_head": P_proj},
        "classifier_representation_sizes": cls_reps,
        "unet_representation_sizes": unet_reps,
        "saved_bytes_per_sample": {"classifier_trainable": cls_saved_train,
                                   "classifier_frozen": cls_saved_frozen, "unet": unet_saved},
        "stage_memory_bytes": scaled,
        "peak_stage": peak_name,
        "process_memory_bytes": process_bytes,
        "overhead": OVERHEAD,
        "cuda_context_bytes": CUDA_CONTEXT_BYTES,
    }
