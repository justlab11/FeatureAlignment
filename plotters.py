import torch
import numpy as np
import logging
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from os import path

from losses import ISEBSW, mmdfuse
from type_defs import DataLoaderSet, EmbeddingSet, ModelSet
from helpers import make_unet

class TSNE_Plotter:
    def __init__(self, dataloaders: DataLoaderSet, embed_size: int, bs:int):
        self.val_loader = dataloaders.val_loader
        self.embed_size = embed_size
        self.bs = bs

    def _get_embeds(self, model, device, unet_model=None):
        embeds_base = np.zeros((len(self.val_loader.dataset), self.embed_size))
        embeds_aux = np.zeros((len(self.val_loader.dataset), self.embed_size))
        labels = np.zeros(len(self.val_loader.dataset))

        self.val_loader.dataset.unique_sources = True

        model.to(device)
        model.eval()

        if unet_model:
            unet_model.to(device)
            unet_model.eval()
        
        for i, (base, aux, z) in enumerate(self.val_loader):
            base = base.to(device)
            aux = aux.to(device)
            with torch.no_grad():
                outputs_base = model(base)[-2].cpu().numpy()
                
                if unet_model:
                    aux = unet_model(aux)[-1]

                outputs_aux = model(aux)[-2].cpu().numpy()
                

                embeds_base[i*self.bs:i*self.bs+self.bs] = outputs_base.squeeze()
                embeds_aux[i*self.bs:i*self.bs+self.bs] = outputs_aux.squeeze()
                labels[i*self.bs:i*self.bs+self.bs] = z.cpu().squeeze()

        model_embeds = EmbeddingSet(
            base_embeds = embeds_base,
            aux_embeds = embeds_aux,
            labels = labels
        )

        return model_embeds
    
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

        plt.rcParams['font.size'] = 16

        # Perform t-SNE for all three embedding sets
        tsne = TSNE(n_components=2, random_state=42)
        base_tsne = tsne.fit_transform(base_model_embeds)
        mixed_tsne = tsne.fit_transform(mixed_model_embeds)
        contrast_tsne = tsne.fit_transform(contrast_model_embeds)

        # Create three subplots side by side
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 8))

        # Plot for base embeddings
        tab20 = plt.cm.get_cmap('tab20')
        color_dict = {i: tab20(i/20) for i in range(20)}

        scatter1a = ax1.scatter(base_tsne[:len(base_tsne)//2, 0], base_tsne[:len(base_tsne)//2, 1], c=[color_dict[label] for label in base_labels[:len(base_tsne)//2]])
        scatter1b = ax1.scatter(base_tsne[len(base_tsne)//2:, 0], base_tsne[len(base_tsne)//2:, 1], c=[color_dict[label] for label in base_labels[len(base_tsne)//2:]], marker="x", s=30)
        ax1.set_title(f'Base Model Embeddings (Accuracy: {round(accuracies.base*100, 2)}%)')
        ax1.set_xlabel('t-SNE feature 1')
        ax1.set_ylabel('t-SNE feature 2')
        colors = [color_dict[i] for i in range(20)]
        cmap = mcolors.ListedColormap(colors)
        bounds = np.arange(21)
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        # Create the colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax1, ticks=np.arange(0.5, 20))

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
        cbar.set_ticklabels(c_labels)
        cbar.set_label("Classes")

        tab20 = plt.cm.get_cmap('tab20')
        color_dict = {i: tab20(i/20) for i in range(20)}

        scatter2a = ax2.scatter(mixed_tsne[:len(mixed_tsne)//2, 0], mixed_tsne[:len(mixed_tsne)//2, 1], c=[color_dict[label] for label in mixed_labels[:len(mixed_tsne)//2]])
        scatter2b = ax2.scatter(mixed_tsne[len(mixed_tsne)//2:, 0], mixed_tsne[len(mixed_tsne)//2:, 1], c=[color_dict[label] for label in mixed_labels[len(mixed_tsne)//2:]], marker="x", s=30)
        ax2.set_title(f'Mixed Model Embeddings (Accuracy: {round(accuracies.mixed*100, 2)}%)')
        ax2.set_xlabel('t-SNE feature 1')
        ax2.set_ylabel('t-SNE feature 2')
        colors = [color_dict[i] for i in range(20)]
        cmap = mcolors.ListedColormap(colors)
        bounds = np.arange(21)
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        # Create the colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax2, ticks=np.arange(0.5, 20))

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
        cbar.set_ticklabels(c_labels)
        cbar.set_label("Classes")


        # Plot for contrast embeddings
        tab20 = plt.cm.get_cmap('tab20')
        color_dict = {i: tab20(i/20) for i in range(20)}

        scatter3a = ax3.scatter(contrast_tsne[:len(contrast_tsne)//2, 0], contrast_tsne[:len(contrast_tsne)//2, 1], c=[color_dict[label] for label in contrast_labels[:len(contrast_tsne)//2]])
        scatter3b = ax3.scatter(contrast_tsne[len(contrast_tsne)//2:, 0], contrast_tsne[len(contrast_tsne)//2:, 1], c=[color_dict[label] for label in contrast_labels[len(contrast_tsne)//2:]], marker="x", s=30)
        ax3.set_title(f'Contrastive Model Embeddings (Accuracy: {round(accuracies.contrast*100, 2)}%)')
        ax3.set_xlabel('t-SNE feature 1')
        ax3.set_ylabel('t-SNE feature 2')
        colors = [color_dict[i] for i in range(20)]
        cmap = mcolors.ListedColormap(colors)
        bounds = np.arange(21)
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        # Create the colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax3, ticks=np.arange(0.5, 20))

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
        cbar.set_ticklabels(c_labels)
        cbar.set_label("Classes")

        fig.suptitle(f"Target - {base.capitalize()} / Source - {aux.capitalize()}")

        plt.tight_layout()
        plt.savefig(filename, format="pdf", dpi=300)

        plt.close()


class EBSW_Plotter:
    def __init__(self, dataloaders: DataLoaderSet, batch_size: int):
        self.val_loader = dataloaders.val_loader
        self.batch_size = batch_size

    def run_isebsw(
        self, model, dataloader, layers, device, target_only=True, alignment_model=None, num_projections=128
    ):
        """
        Computes ISEBSW distances for each batch and layer.

        Args:
            model (nn.Module): The classifier or base model.
            dataloader (DataLoader): DataLoader for evaluation.
            layers (list[int] or int): Layers to evaluate.
            device (torch.device): Device for computation.
            target_only (bool): If True, compares splits of target; else compares target and alignment_model outputs.
            alignment_model (nn.Module, optional): Alignment model for transformed comparisons.
            num_projections (int): Number of projections for ISEBSW.

        Returns:
            np.ndarray: Array of shape (num_batches, num_layers) with ISEBSW values.
        """
        model.to(device)
        model.eval()

        if alignment_model is not None:
            alignment_model.to(device)
            alignment_model.eval()

        if isinstance(layers, int):
            layers = [i for i in range(layers+1)]

        total_isebsw_loss = np.zeros((len(dataloader), len(layers)))

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
                target_flat = target_outputs[layer].view(target_outputs[layer].size(0), -1)
                source_flat = source_outputs[layer].view(source_outputs[layer].size(0), -1)

                total_isebsw_loss[i, j] += ISEBSW(
                    target_flat, source_flat, L=num_projections, device=device
                ).item() / target_flat.size(0)

        return total_isebsw_loss
    

    def plot_ebsw(
        self,
        models: ModelSet,
        layers: list | int,
        device: str,
        filename: str,
        alignment_models: ModelSet | None = None,
        num_projections: int = 128
    ):
        include_alignment = True
        if alignment_models is None:
            alignment_models = ModelSet(
                base=None,
                mixed=None,
                contrast=None
            )
            include_alignment = False

        base_ebsw_inter = self.run_isebsw(
            model=models.base,
            dataloader=self.val_loader,
            layers=layers,
            target_only=True,
            device=device,
            alignment_model=None,
            num_projections=num_projections
        )

        mixed_ebsw_inter = self.run_isebsw(
            model=models.mixed,
            dataloader=self.val_loader,
            layers=layers,
            target_only=True,
            device=device,
            alignment_model=None,
            num_projections=num_projections
        )

        contrast_ebsw_inter = self.run_isebsw(
            model=models.contrast,
            dataloader=self.val_loader,
            layers=layers,
            target_only=True,
            device=device,
            alignment_model=None,
            num_projections=num_projections
        )

        base_ebsw_intra = self.run_isebsw(
            model=models.base,
            dataloader=self.val_loader,
            layers=layers,
            target_only=False,
            device=device,
            alignment_model=None,
            num_projections=num_projections
        )

        mixed_ebsw_intra = self.run_isebsw(
            model=models.mixed,
            dataloader=self.val_loader,
            layers=layers,
            target_only=False,
            device=device,
            alignment_model=None,
            num_projections=num_projections
        )

        contrast_ebsw_intra = self.run_isebsw(
            model=models.contrast,
            dataloader=self.val_loader,
            layers=layers,
            target_only=False,
            device=device,
            alignment_model=None,
            num_projections=num_projections
        )

        if include_alignment:
            align_base_ebsw_intra = self.run_isebsw(
                model=models.base,
                dataloader=self.val_loader,
                layers=layers,
                target_only=False,
                device=device,
                alignment_model=alignment_models.base,
                num_projections=num_projections
            )

            align_mixed_ebsw_intra = self.run_isebsw(
                model=models.mixed,
                dataloader=self.val_loader,
                layers=layers,
                target_only=False,
                device=device,
                alignment_model=alignment_models.mixed,
                num_projections=num_projections
            )

            align_contrast_ebsw_intra = self.run_isebsw(
                model=models.contrast,
                dataloader=self.val_loader,
                layers=layers,
                target_only=False,
                device=device,
                alignment_model=alignment_models.contrast,
                num_projections=num_projections
            )

            align_layer_measurements_inter = ModelSet(
                base=align_base_ebsw_intra,
                mixed=align_mixed_ebsw_intra,
                contrast=align_contrast_ebsw_intra
            ).dict()

        layer_measurements_inter = ModelSet(
            base=base_ebsw_inter,
            mixed=mixed_ebsw_inter,
            contrast=contrast_ebsw_inter
        ).dict()

        layer_measurements_intra = ModelSet(
            base=base_ebsw_intra,
            mixed=mixed_ebsw_intra,
            contrast=contrast_ebsw_intra
        ).dict()

        fig, axs = plt.subplots(1, 3, figsize=(20, 7))
        fig.suptitle('Energy-Based Sliced Wasserstein Loss Initial Layerwise Test Results', fontsize=22)

        categories = ['base', 'mixed', 'contrast']

        if include_alignment:
            mode_labels = ['Target/Target', 'Target/Source', "Target/Source Aligned"]
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            modes = [layer_measurements_inter, layer_measurements_intra, align_layer_measurements_inter]
            
        else:
            mode_labels = ['Target/Target', 'Target/Source']
            colors = ['#1f77b4', '#ff7f0e']
            modes = [layer_measurements_inter, layer_measurements_intra]

        for i, cat in enumerate(categories):
            for j, mode in enumerate(modes):
                if i==0 and j==2:
                    # if we are plotting the base model (i==0) with alignment (j==2)
                    # just continue as base never has alignment so the plot is useless
                    continue
                data = mode[cat]
                means = np.mean(data, axis=0)
                stds = np.std(data, axis=0)
                x = range(len(means))

                if len(modes) == 2:
                    offset = .1*j - 0.05 # proper offset value for two error bars
                else:
                    offset = .1*j - .1

                axs[i].grid(True, linestyle=":", alpha=.7)
                axs[i].errorbar(
                    x+offset, means, yerr=stds, capsize=5, marker='o',
                    color=colors[j], ecolor=colors[j],
                    markersize=8, linewidth=2, capthick=2,
                    label=f'{mode_labels[j]} Samples', linestyle='none'
                )

            axs[i].set_title(f'{cat.capitalize()} Model', fontsize=20)
            axs[i].set_xlabel('Layer', fontsize=16)
            axs[i].set_ylabel('Average Loss', fontsize=16)
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
    Plots a grid of example images: original, source, and aligned output.

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
        source_adjusted = source.unsqueeze(0).to(device)
        source_adjusted = alignment_model(source_adjusted)[-1][0]

        target = np.transpose(target, (1, 2, 0))
        source = np.transpose(source, (1, 2, 0))
        source_adjusted = np.transpose(source_adjusted.detach().cpu(), (1, 2, 0))

        axes[0, i].imshow(target)
        axes[0, i].axis('off')
        axes[0, i].set_title(f"Sample {i+1}\nLabel: {label}")

        axes[1, i].imshow(source)
        axes[1, i].axis('off')

        axes[2, i].imshow(source_adjusted)
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.savefig(filename, format="pdf", dpi=300)
    plt.close(fig)
    
def divergence_plots(inter_data, intra_data, val_acc_values, fname):
    """
    Plots validation accuracy and inter/intra-domain distances across layer sets.

    Args:
        inter_data: 2D array (layers x comparisons) of inter-domain distances.
        intra_data: 2D array (layers x comparisons) of intra-domain distances.
        val_acc_values: 1D array of validation accuracies.
        fname: Output filename for the plot (PDF).
    """
    inter_data = torch.tensor(inter_data)
    intra_data = torch.tensor(intra_data)

    inter_data = inter_data[1:]
    intra_data = intra_data[1:]

    inter_mean = torch.mean(inter_data, dim=1)
    inter_std  = torch.std(inter_data, dim=1)

    intra_mean = torch.mean(intra_data, dim=1)
    intra_std  = torch.std(intra_data, dim=1)

    n = inter_mean.size(0)
    x = np.arange(n)
    # Custom x labels for layer sets
    x_labels = [f"{i-1}" if i==n else (f"{i-1}-{n-1}" if i>1 else f"Input-{n-1}") for i in range(n, 0, -1)]
    offset = 0.08

    fig, axes = plt.subplots(inter_mean.size(1)+1, 1, figsize=(6, (inter_mean.size(1)+1)*2+2), sharex=True)

    # Top plot: validation accuracy
    axes[0].plot(x, val_acc_values, '-o', color='C2')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Validation Accuracy on the Target Dataset')
    axes[0].grid(True)
    axes[0].set_xticks([])

    # Comparison subplots
    for i in range(inter_mean.size(1)):
        ax = axes[i+1]
        ax.errorbar(x - offset, inter_mean[:, i], yerr=inter_std[:, i], fmt='-o', label='Inter-layer', color='C0')
        ax.errorbar(x + offset, intra_mean[:, i], yerr=intra_std[:, i], fmt='-s', label='Intra-layer', color='C1')
        ax.set_ylabel(f'Layer {i+1}')
        ax.grid(True)
        if i == 0:
            ax.legend()

    axes[-1].set_xlabel('Layer Set')
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(x_labels)
    fig.suptitle("Inter- and Intra-Domain Distances with Validation Accuracy")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(fname, dpi=300, format='pdf', bbox_inches='tight')
    plt.close()

def incremental_sample_plots(dataloader, unet_fname_set, device, output_folder):
    build_unet = make_unet(
        size=32,
        attention=False,
        base_channels=2,
        noise_channels=2
    )
    for x,y,_ in dataloader:
        target = x[:16]
        source = y[:16].to(device)
        break

    outputs = [target]
    for unet_fname in unet_fname_set:
        unet = buid_unet()
        
        unet.to(device)
        unet.load_state_dict(torch.load(unet_fname, weights_only=True))

        unet.eval()
        outputs.append(unet(source)[-1].detach().cpu())
        del unet

    outputs.append(source)

    num_entries = len(outputs)
    num_items = 16

    for item_idx in range(num_items):
        fig, axs = plt.subplots(1, num_entries, figsize=(num_entries*2, 2))

        for entry_idx, ax in enumerate(axs):
            img = outputs[num_entries-entry_idx-1][item_idx]
            if img.shape[0] == 3:
                img.np.transpose(img, (1,2,0))

            ax.imshow(img)
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"increment_image-{item_idx}.pdf"), dpi=300, bbox_inches='tight')
        plt.close()
