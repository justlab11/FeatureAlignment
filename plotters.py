import torch
import numpy as np
import logging
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from typing import List, Callable
from os import path

from losses import ISEBSW, mmdfuse
from type_defs import DataLoaderSet, EmbeddingSet, ModelSet

logger = logging.getLogger(__name__)

def _span_labels(num_spans: int):
    """
    Labels for the layer sets of the cascade. Span k covers layers (num_spans-k)..(num_spans-1),
    from only the last layer (first) to every layer including the input (last).
    """
    last = num_spans - 1
    labels = []
    for k in range(1, num_spans + 1):
        start = num_spans - k
        if start == last:
            labels.append(f"{last}")
        elif start == 0:
            labels.append(f"Input-{last}")
        else:
            labels.append(f"{start}-{last}")
    return labels

def _to_hwc(img):
    """Converts a CHW tensor/array to a clipped HWC numpy array that imshow can show."""
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    return np.clip(np.transpose(img, (1, 2, 0)), 0, 1)

class TSNE_Plotter:
    def __init__(self, dataloaders: DataLoaderSet, embed_size: int, bs:int, max_points: int = 2000):
        self.val_loader = dataloaders.val_loader
        self.embed_size = embed_size
        self.bs = bs
        # t-SNE on every validation sample is slow, so each domain is subsampled to this many points
        self.max_points = max_points

    def _get_embeds(self, model, device, unet_model=None):
        embeds_base = np.zeros((len(self.val_loader.dataset), self.embed_size))
        embeds_aux = np.zeros((len(self.val_loader.dataset), self.embed_size))
        labels = np.zeros(len(self.val_loader.dataset))

        model.to(device)
        model.eval()

        if unet_model:
            unet_model.to(device)
            unet_model.eval()

        for i, (base, aux, z) in enumerate(self.val_loader):
            base = base.to(device)
            aux = aux.to(device)
            with torch.no_grad():
                # [-2] is the penultimate layer (the last entry is the logits)
                outputs_base = model(base)[-2].cpu().numpy()

                if unet_model:
                    aux = unet_model(aux)[-1]

                outputs_aux = model(aux)[-2].cpu().numpy()

                start = i * self.bs
                end = start + outputs_base.shape[0]
                embeds_base[start:end] = outputs_base.reshape(outputs_base.shape[0], -1)
                embeds_aux[start:end] = outputs_aux.reshape(outputs_aux.shape[0], -1)
                labels[start:end] = z.cpu().reshape(-1)

        model_embeds = EmbeddingSet(
            base_embeds = embeds_base,
            aux_embeds = embeds_aux,
            labels = labels
        )

        return model_embeds

    def _add_colorbar(self, fig, ax, color_dict, c_labels):
        colors = [color_dict[i] for i in range(20)]
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(np.arange(21), cmap.N)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, ticks=np.arange(0.5, 20))
        cbar.set_ticklabels(c_labels)
        cbar.set_label("Classes")

    def _subsample(self, embeds: EmbeddingSet, idx):
        return EmbeddingSet(
            base_embeds=embeds.base_embeds[idx],
            aux_embeds=embeds.aux_embeds[idx],
            labels=embeds.labels[idx],
        )

    def plot_tsne(self, models: ModelSet, accuracies: ModelSet, device,
                  filename, base, aux, unet_models: None|ModelSet=None):

        if unet_models:
            base_embeds = self._get_embeds(
                model=models.base,
                unet_model=unet_models.base,
                device=device,
            )

            mixed_embeds = self._get_embeds(
                model=models.mixed,
                unet_model=unet_models.mixed,
                device=device,
            )

            contrast_embeds = self._get_embeds(
                model=models.contrast,
                unet_model=unet_models.contrast,
                device=device,
            )

        else:
            base_embeds = self._get_embeds(
                model=models.base,
                device=device,
            )

            mixed_embeds = self._get_embeds(
                model=models.mixed,
                device=device,
            )

            contrast_embeds = self._get_embeds(
                model=models.contrast,
                device=device,
            )

        # the same validation samples are shown for every model
        num_samples = len(base_embeds.labels)
        keep = np.sort(np.random.default_rng(42).choice(
            num_samples, size=min(num_samples, self.max_points), replace=False
        ))
        base_embeds = self._subsample(base_embeds, keep)
        mixed_embeds = self._subsample(mixed_embeds, keep)
        contrast_embeds = self._subsample(contrast_embeds, keep)

        base_model_embeds = np.concatenate(
            (base_embeds.base_embeds, base_embeds.aux_embeds)
        )
        mixed_model_embeds = np.concatenate(
            (mixed_embeds.base_embeds, mixed_embeds.aux_embeds)
        )

        contrast_base_norms = np.linalg.norm(contrast_embeds.base_embeds, axis=1, keepdims=True)
        contrast_base_norms = np.maximum(contrast_base_norms, 1e-12)
        contrast_embeds.base_embeds = contrast_embeds.base_embeds / contrast_base_norms

        contrast_aux_norms = np.linalg.norm(contrast_embeds.aux_embeds, axis=1, keepdims=True)
        contrast_aux_norms = np.maximum(contrast_aux_norms, 1e-12)
        contrast_embeds.aux_embeds = contrast_embeds.aux_embeds / contrast_aux_norms

        contrast_model_embeds = np.concatenate(
            (contrast_embeds.base_embeds, contrast_embeds.aux_embeds)
        )

        base_labels = np.concatenate(
            (base_embeds.labels*2, base_embeds.labels*2+1)
        )
        mixed_labels = np.concatenate(
            (mixed_embeds.labels*2, mixed_embeds.labels*2+1)
        )
        contrast_labels = np.concatenate(
            (contrast_embeds.labels*2, contrast_embeds.labels*2+1)
        )

        # Perform t-SNE for all three embedding sets
        tsne = TSNE(n_components=2, random_state=42)
        base_tsne = tsne.fit_transform(base_model_embeds)
        mixed_tsne = tsne.fit_transform(mixed_model_embeds)
        contrast_tsne = tsne.fit_transform(contrast_model_embeds)

        tsne_data = {
            "base": base_tsne,
            "mixed": mixed_tsne,
            "contrast": contrast_tsne
        }

        labels = {
            "base": base_labels,
            "mixed": mixed_labels,
            "contrast": contrast_labels
        }

        accuracies = accuracies.dict()

        c_labels = [
            "0 - Target", "0 - Source",
            "1 - Target", "1 - Source",
            "2 - Target", "2 - Source",
            "3 - Target", "3 - Source",
            "4 - Target", "4 - Source",
            "5 - Target", "5 - Source",
            "6 - Target", "6 - Source",
            "7 - Target", "7 - Source",
            "8 - Target", "8 - Source",
            "9 - Target", "9 - Source",
        ]

        base_fname = path.splitext(filename)[0]

        for name, tsne_values in tsne_data.items():
            fig, ax = plt.subplots(1, 1, figsize=(8,8))
            plt.rcParams['font.size'] = 16
            tab20 = plt.get_cmap('tab20')
            color_dict = {i: tab20(i/20) for i in range(20)}

            if unet_models and name != "base":
                ax.set_title(f"{name.upper()} + ShiftNet Embeddings (Accuracy: {round(accuracies[name]*100, 2)}%)")
            else:
                ax.set_title(f"{name.upper()} Model Embeddings (Accuracy: {round(accuracies[name]*100, 2)}%)")

            half = len(tsne_values) // 2
            tsne_labels = labels[name]

            target_scatter = ax.scatter(
                tsne_values[:half, 0], tsne_values[:half, 1],
                c=[color_dict[label] for label in tsne_labels[:half]]
            )

            source_scatter = ax.scatter(
                tsne_values[half:, 0],
                tsne_values[half:, 1],
                c=[color_dict[label] for label in tsne_labels[half:]],
                marker="x",
                s=30,
            )

            ax.legend(handles=[
                Line2D([], [], marker="o", linestyle="", color="gray", label="Target"),
                Line2D([], [], marker="x", linestyle="", color="gray", label="Source"),
            ], loc="best", fontsize=12)

            self._add_colorbar(
                fig=fig, ax=ax,
                color_dict=color_dict,
                c_labels=c_labels
            )

            if unet_models:
                new_filename = base_fname + "+" + name + "+shiftnet.pdf"
            else:
                new_filename = base_fname + "+" + name + ".pdf"
            plt.savefig(new_filename, format="pdf", dpi=300)
            plt.close()

class EBSW_Plotter:
    def __init__(self, dataloaders: DataLoaderSet, batch_size: int):
        self.val_loader = dataloaders.val_loader
        self.batch_size = batch_size

    @torch.no_grad()
    def _run_divergence(
        self, divergence: Callable, model, dataloader, layers, device, target_only=True, alignment_model=None
    ):
        """
        Computes a divergence for each batch and layer.

        Args:
            divergence (callable): takes two flattened (n, d) feature tensors and returns a float.
            model (nn.Module): The classifier or base model.
            dataloader (DataLoader): DataLoader for evaluation. Pass a pure-class loader to get class-conditional values.
            layers (list[int] or int): Layers to evaluate.
            device (torch.device): Device for computation.
            target_only (bool): If True, compares two halves of the target batch (the Target/Target reference);
                else compares half of the target batch with half of the (optionally aligned) source batch.
            alignment_model (nn.Module, optional): Alignment model applied to the source samples.

        Returns:
            np.ndarray: Array of shape (num_batches, num_layers). Batches that are too small to split are NaN.
        """
        model.to(device)
        model.eval()

        if alignment_model is not None:
            alignment_model.to(device)
            alignment_model.eval()

        if isinstance(layers, int):
            layers = [i for i in range(layers+1)]

        results = np.full((len(dataloader), len(layers)), np.nan)

        for i, (target, source, _) in enumerate(dataloader):
            if target_only:
                chunks = torch.chunk(target, 2)
                if len(chunks) < 2:
                    continue

                min_len = min(chunks[0].size(0), chunks[1].size(0))
                target_set = chunks[0][:min_len]
                source_set = chunks[1][:min_len]

            else:
                chunks_target = torch.chunk(target, 2)
                chunks_source = torch.chunk(source, 2)

                if len(chunks_target) < 2 or len(chunks_source) < 2:
                    continue

                target_set, source_set = chunks_target[0], chunks_source[0]

            target_set = target_set.to(device)
            source_set = source_set.to(device)

            if alignment_model and not target_only:
                source_set = alignment_model(source_set)[-1]

            target_outputs = model(target_set)[:layers[-1]+1]
            source_outputs = model(source_set)[:layers[-1]+1]

            for j, layer in enumerate(layers):
                target_flat = target_outputs[layer].reshape(target_outputs[layer].size(0), -1)
                source_flat = source_outputs[layer].reshape(source_outputs[layer].size(0), -1)

                results[i, j] = divergence(target_flat, source_flat)

        return results

    def run_isebsw(
        self, model, dataloader, layers, device, target_only=True, alignment_model=None, num_projections=128
    ):
        """
        Computes ISEBSW distances for each batch and layer (see `_run_divergence`).

        Returns:
            np.ndarray: Array of shape (num_batches, num_layers) with ISEBSW values.
        """
        def divergence(target_flat, source_flat):
            return ISEBSW(
                target_flat, source_flat, L=num_projections, device=device
            ).item() / target_flat.size(0)

        return self._run_divergence(
            divergence, model, dataloader, layers, device, target_only, alignment_model
        )

    def run_mmdfuse(
        self, model, dataloader, layers, device, target_only=True, alignment_model=None
    ):
        """
        Computes MMD-Fuse statistics for each batch and layer (see `_run_divergence`).

        Returns:
            np.ndarray: Array of shape (num_batches, num_layers) with MMD-Fuse values.
        """
        def divergence(target_flat, source_flat):
            return mmdfuse(target_flat, source_flat, device=device).item()

        return self._run_divergence(
            divergence, model, dataloader, layers, device, target_only, alignment_model
        )

    def plot_ebsw(
        self,
        models: ModelSet,
        layers: list | int,
        device: str,
        filename: str,
        alignment_models: ModelSet | None = None,
        num_projections: int = 128,
        loss: str = "ebsw",
        log_scale: bool = True
    ):
        """
        Plots the layerwise divergence for Target/Target (reference), Target/Source and, if
        `alignment_models` is given, Target/Aligned-Source for each of the base, mixed and contrast models.
        """
        if loss == "ebsw":
            measure = lambda **kwargs: self.run_isebsw(num_projections=num_projections, **kwargs)
        else:
            measure = self.run_mmdfuse

        include_alignment = alignment_models is not None
        categories = ['base', 'mixed', 'contrast']

        target_target = {}
        target_source = {}
        target_aligned = {}

        for cat in categories:
            model = getattr(models, cat)
            common = dict(model=model, dataloader=self.val_loader, layers=layers, device=device)

            target_target[cat] = measure(target_only=True, alignment_model=None, **common)
            target_source[cat] = measure(target_only=False, alignment_model=None, **common)

            # the base model is never aligned
            if include_alignment and getattr(alignment_models, cat) is not None:
                target_aligned[cat] = measure(
                    target_only=False, alignment_model=getattr(alignment_models, cat), **common
                )

        series = [
            ('Target/Target', '#1f77b4', target_target),
            ('Target/Source', '#ff7f0e', target_source),
        ]
        if target_aligned:
            series.append(('Target/Source Aligned', '#2ca02c', target_aligned))

        fig, axs = plt.subplots(1, 3, figsize=(20, 7))
        fig.suptitle('Layerwise Divergence on the Validation Set', fontsize=22)

        for i, cat in enumerate(categories):
            for j, (label, color, data) in enumerate(series):
                if cat not in data:
                    continue

                means = np.nanmean(data[cat], axis=0)
                stds = np.nanstd(data[cat], axis=0)
                x = np.arange(len(means))

                offset = 0.1 * j - 0.05 * (len(series) - 1)

                axs[i].grid(True, linestyle=":", alpha=.7)
                axs[i].errorbar(
                    x+offset, means, yerr=stds, capsize=5, marker='o',
                    color=color, ecolor=color,
                    markersize=8, linewidth=2, capthick=2,
                    label=f'{label} Samples', linestyle='none'
                )

            if log_scale:
                axs[i].set_yscale('log')

            axs[i].set_title(f'{cat.capitalize()} Model', fontsize=20)
            axs[i].set_xlabel('Layer', fontsize=16)
            axs[i].set_ylabel('Average Divergence', fontsize=16)
            axs[i].tick_params(axis='both', which='major', labelsize=14)
            axs[i].legend(fontsize=12)

        plt.tight_layout()

        try:
            plt.savefig(filename, format="pdf", dpi=300)

        except Exception as e:
            print(f"Warning: plot could not be saved to {filename}: {e}")

        plt.close(fig)

def plot_examples(dataset, alignment_model, filename, device, num_samples=10):
    """
    Plots a grid of example images: target, source, and aligned source.

    The source and target in a column are different samples of the same class (the dataset pairs them randomly).

    Args:
        dataset: Dataset object supporting indexing.
        alignment_model: Model used to transform the source image.
        filename: Output filename for the plot (PDF).
        device: Torch device for model inference.
        num_samples: Number of examples to plot (default: 10).
    """
    alignment_model.eval()
    num_samples = min(num_samples, len(dataset))
    random_samples = np.random.choice(len(dataset), num_samples, replace=False)

    fig, axes = plt.subplots(3, num_samples, figsize=(2*num_samples, 6))

    for i, sample_idx in enumerate(random_samples):
        target, source, label = dataset[sample_idx]

        with torch.no_grad():
            source_adjusted = alignment_model(source.unsqueeze(0).to(device))[-1][0]

        axes[0, i].imshow(_to_hwc(target))
        axes[0, i].axis('off')
        axes[0, i].set_title(f"Sample {i+1}\nLabel: {int(label)}")

        axes[1, i].imshow(_to_hwc(source))
        axes[1, i].axis('off')

        axes[2, i].imshow(_to_hwc(source_adjusted))
        axes[2, i].axis('off')

    # axis('off') hides axis labels, so the row names are drawn as text
    for row, name in enumerate(["Target", "Source", "Aligned"]):
        axes[row, 0].text(-0.08, 0.5, name, transform=axes[row, 0].transAxes,
                          rotation=90, va="center", ha="right", fontsize=12)

    plt.tight_layout()
    plt.savefig(filename, format="pdf", dpi=300)
    plt.close(fig)

def plot_dataset_examples(dataset, filename, device, num_samples=10):
    """
    Plots a grid of example target/source pairs from the dataset
    before any alignment, to verify data loading and transforms look correct.
    """
    num_samples = min(num_samples, len(dataset))
    random_samples = np.random.choice(len(dataset), num_samples, replace=False)

    fig, axes = plt.subplots(2, num_samples, figsize=(2*num_samples, 4))

    for i, sample_idx in enumerate(random_samples):
        target, source, label = dataset[sample_idx]

        axes[0, i].imshow(_to_hwc(target))
        axes[0, i].axis('off')
        axes[0, i].set_title(f"Label: {int(label)}", fontsize=8)

        axes[1, i].imshow(_to_hwc(source))
        axes[1, i].axis('off')

    for row, name in enumerate(["Target", "Source"]):
        axes[row, 0].text(-0.08, 0.5, name, transform=axes[row, 0].transAxes,
                          rotation=90, va="center", ha="right", fontsize=10)

    plt.tight_layout()
    plt.savefig(filename, format="pdf", dpi=300)
    plt.close(fig)

def divergence_plots(target_target, target_source, val_acc_values, fname, test_acc_values=None):
    """
    Plots accuracy and layerwise divergence across the cascade's layer sets.

    Args:
        target_target: array (spans+1, batches, layers) of Target/Target divergences (reference floor).
            Row 0 is before any alignment; row k is after the k-th layer set.
        target_source: array (spans+1, batches, layers) of Target/Source divergences. Row 0 is the
            unaligned source; row k is the source aligned for the k-th layer set.
        val_acc_values: 1D array (spans,) of target validation accuracies in [0, 1].
        fname: Output filename for the plot (PDF).
        test_acc_values: optional 1D array (spans,) of target test accuracies in [0, 1].
    """
    target_target = np.asarray(target_target, dtype=float)
    target_source = np.asarray(target_source, dtype=float)

    # average over batches; NaN batches (too small to split) are ignored
    tt_mean = np.nanmean(target_target[1:], axis=1)
    tt_std  = np.nanstd(target_target[1:], axis=1)
    ts_mean = np.nanmean(target_source[1:], axis=1)
    ts_std  = np.nanstd(target_source[1:], axis=1)
    unaligned_ts = np.nanmean(target_source[0], axis=0)

    n = ts_mean.shape[0]
    num_layers = ts_mean.shape[1]
    x = np.arange(n)
    x_labels = _span_labels(n)
    offset = 0.08

    fig, axes = plt.subplots(num_layers+1, 1, figsize=(6, (num_layers+1)*2+2), sharex=True)

    # Top plot: accuracy
    axes[0].plot(x, np.asarray(val_acc_values) * 100, '-o', color='C2', label='Validation')
    if test_acc_values is not None:
        axes[0].plot(x, np.asarray(test_acc_values) * 100, '-s', color='C3', label='Test')
        axes[0].legend()
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Accuracy on the Target Dataset')
    axes[0].grid(True)

    # One panel per measured layer
    for layer in range(num_layers):
        ax = axes[layer+1]
        ax.errorbar(x - offset, tt_mean[:, layer], yerr=tt_std[:, layer], fmt='-o', label='Target/Target', color='C0')
        ax.errorbar(x + offset, ts_mean[:, layer], yerr=ts_std[:, layer], fmt='-s', label='Target/Aligned Source', color='C1')
        ax.axhline(unaligned_ts[layer], color='C1', linestyle='--', alpha=0.6, label='Target/Source (unaligned)')
        ax.set_ylabel(f'Layer {layer}')
        ax.grid(True)
        if layer == 0:
            ax.legend(fontsize=8)

    axes[-1].set_xlabel('Layer Set')
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(x_labels, rotation=45)
    fig.suptitle("Layerwise Divergence and Accuracy by Layer Set")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(fname, dpi=300, format='pdf', bbox_inches='tight')
    plt.close()

def gap_closed_heatmap(target_target, target_source, fname):
    """
    Heatmap of the fraction of the domain gap closed by each layer set at each layer:

        (unaligned Target/Source - aligned Target/Source) / (unaligned Target/Source - Target/Target)

    1 means the aligned source is as close to the target as two target halves are, 0 means no change,
    and negative values mean the alignment made that layer worse. Cells where there was no gap to close
    (Target/Source not larger than Target/Target) are left blank.

    Takes the same arrays as `divergence_plots`.
    """
    target_target = np.asarray(target_target, dtype=float)
    target_source = np.asarray(target_source, dtype=float)

    floor = np.nanmean(target_target, axis=(0, 1))
    unaligned = np.nanmean(target_source[0], axis=0)
    aligned = np.nanmean(target_source[1:], axis=1)

    gap = unaligned - floor
    with np.errstate(divide="ignore", invalid="ignore"):
        closed = (unaligned[None, :] - aligned) / gap[None, :]
    closed[:, gap <= 1e-12] = np.nan

    n, num_layers = closed.shape
    fig, ax = plt.subplots(figsize=(1 + 0.9 * num_layers, 1 + 0.5 * n + 1))
    im = ax.imshow(np.clip(closed, -1, 1), cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")

    for r in range(n):
        for c in range(num_layers):
            if not np.isnan(closed[r, c]):
                ax.text(c, r, f"{closed[r, c]:.2f}", ha="center", va="center", fontsize=8)

    ax.set_xticks(np.arange(num_layers))
    ax.set_xticklabels([str(i) for i in range(num_layers)])
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(_span_labels(n))
    ax.set_xlabel("Measured layer")
    ax.set_ylabel("Layer set aligned")
    ax.set_title("Fraction of Domain Gap Closed")
    fig.colorbar(im, ax=ax, label="Gap closed (clipped to [-1, 1])")

    plt.tight_layout()
    plt.savefig(fname, dpi=300, format="pdf", bbox_inches="tight")
    plt.close(fig)

def span_accuracy_plot(val_acc_values, test_acc_values, fname, best_index=None, reference_accs: dict | None = None):
    """
    Plots target accuracy for each layer set.

    Args:
        val_acc_values: 1D array (spans,) of validation accuracies in [0, 1].
        test_acc_values: 1D array (spans,) of test accuracies in [0, 1].
        fname: Output filename for the plot (PDF).
        best_index: optional index of the selected span, marked with a star.
        reference_accs: optional {label: accuracy in [0, 1]} drawn as horizontal lines (e.g. the same
            classifier before alignment).
    """
    val_acc_values = np.asarray(val_acc_values) * 100
    test_acc_values = np.asarray(test_acc_values) * 100
    n = len(val_acc_values)
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, val_acc_values, '-o', color='C2', label='Validation')
    ax.plot(x, test_acc_values, '-s', color='C3', label='Test')

    if best_index is not None:
        ax.plot([best_index], [test_acc_values[best_index]], marker="*", markersize=16,
                color="gold", markeredgecolor="black", linestyle="", label="Selected (by validation)")

    for k, (label, acc) in enumerate((reference_accs or {}).items()):
        ax.axhline(acc * 100, linestyle="--", color=f"C{4+k}", alpha=0.8, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(_span_labels(n), rotation=45)
    ax.set_xlabel("Layer Set")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Target Accuracy by Layer Set")
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(fname, dpi=300, format="pdf", bbox_inches="tight")
    plt.close(fig)

def incremental_sample_plots(dataloader, unet_fname_set: List[str], device, output_folder: str, build_unet: Callable):
    """
    Generate side-by-side visualization plots comparing target images, outputs from incremental
    U-Net model checkpoints, and source inputs.

    Takes the first batch from a dataloader, extracts 16 samples, and runs inference using each
    provided U-Net model file (assumed to be training checkpoints or variants). Creates one PDF
    per sample showing the progression from source input (left) to target (right), useful for
    inspecting model training evolution or comparisons.

    Args:
        dataloader (torch.utils.data.DataLoader): Dataloader yielding (x, y, _) batches where
            x are targets and y are source inputs (expects channel-first tensors).
        unet_fname_set (list[str]): List of file paths to U-Net checkpoints, in the order they should appear.
        device (torch.device): Device for model inference (e.g., 'cuda').
        output_folder (str): Directory to save PDF plots (files named 'increment_image-{idx}.pdf').
        build_unet (Callable): Builds an untrained U-Net with the architecture the checkpoints were saved from.

    Returns:
        None. Saves up to 16 PDF files to output_folder.

    Note:
        - Outputs use final layer of U-Net: unet(source)[-1].
        - The source and target in a plot are different samples (the dataset pairs same-class samples at random).
        - Memory-efficient: deletes each model after use.
    """

    for x,y,_ in dataloader:
        target = x[:16]
        source = y[:16]
        break

    outputs = [target]
    for unet_fname in unet_fname_set:
        unet = build_unet()

        unet.to(device)
        unet.load_state_dict(torch.load(unet_fname, weights_only=True))

        unet.eval()
        with torch.no_grad():
            outputs.append(unet(source.to(device))[-1].detach().cpu())
        del unet

    outputs.append(source.cpu())

    num_entries = len(outputs)
    num_items = min(16, target.shape[0])

    for item_idx in range(num_items):
        fig, axs = plt.subplots(1, num_entries, figsize=(num_entries*2, 2))

        for entry_idx, ax in enumerate(axs):
            ax.imshow(_to_hwc(outputs[num_entries-entry_idx-1][item_idx]))
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(path.join(output_folder, f"increment_image-{item_idx}.pdf"), dpi=300, bbox_inches='tight')
        plt.close()
