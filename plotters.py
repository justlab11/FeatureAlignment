import torch
import numpy as np
import logging
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


from losses import ISEBSW
from type_defs import DataLoaderSet, EmbeddingSet, ModelSet

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
                    aux = unet_model(aux)

                outputs_aux = model(aux)[-2].cpu().numpy()
                

                embeds_base[i*self.bs:i*self.bs+self.bs] = outputs_base
                embeds_aux[i*self.bs:i*self.bs+self.bs] = outputs_aux
                labels[i*self.bs:i*self.bs+self.bs] = z.cpu()

        model_embeds = EmbeddingSet(
            base_embeds = embeds_base,
            aux_embeds = embeds_aux,
            labels = labels
        )

        return model_embeds
    
    def plot_tsne(self, classifiers: ModelSet, accuracies: ModelSet, device,
                  filename, base, aux, unet_models: None|ModelSet=None):
        
        if unet_models:
            base_embeds = self._get_embeds(
                model=classifiers.base,
                unet_model=unet_models.base,
                device=device,
            )

            mixed_embeds = self._get_embeds(
                model=classifiers.mixed,
                unet_model=unet_models.mixed,
                device=device,
            )

            contrast_embeds = self._get_embeds(
                model=classifiers.contrast,
                unet_model=unet_models.contrast,
                device=device,
            )

        else:
            base_embeds = self._get_embeds(
                model=classifiers.base,
                device=device,
            )

            mixed_embeds = self._get_embeds(
                model=classifiers.mixed,
                device=device,
            )

            contrast_embeds = self._get_embeds(
                model=classifiers.contrast,
                device=device,
            )

        base_model_embeds = base_embeds.base_embeds
        mixed_model_embeds = np.concatenate(
            (mixed_embeds.base_embeds, mixed_embeds.aux_embeds)
        )
        contrast_model_embeds = np.concatenate(
            (contrast_embeds.base_embeds, contrast_embeds.aux_embeds)
        )

        base_labels = base_embeds.labels
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
        scatter1 = ax1.scatter(base_tsne[:, 0], base_tsne[:, 1], c=base_labels, cmap='tab10', alpha=.7)
        ax1.set_title(f'Base Model Embeddings (Accuracy: {round(accuracies.base*100, 2)}%)')
        ax1.set_xlabel('t-SNE feature 1')
        ax1.set_ylabel('t-SNE feature 2')
        cbar = fig.colorbar(scatter1, ax=ax1)
        ticks = np.arange(0, 10)
        c_labels = [ 
            "0 - Base",
            "1 - Base",
            "2 - Base",
            "3 - Base",
            "4 - Base",
            "5 - Base",
            "6 - Base",
            "7 - Base",
            "8 - Base",
            "9 - Base",
        ]
        cbar.set_ticks(ticks)
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
            "0 - Base", "0 - Aux",
            "1 - Base", "1 - Aux",
            "2 - Base", "2 - Aux",
            "3 - Base", "3 - Aux",
            "4 - Base", "4 - Aux",
            "5 - Base", "5 - Aux",
            "6 - Base", "6 - Aux",
            "7 - Base", "7 - Aux",
            "8 - Base", "8 - Aux",
            "9 - Base", "9 - Aux",
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
            "0 - Base", "0 - Aux",
            "1 - Base", "1 - Aux",
            "2 - Base", "2 - Aux",
            "3 - Base", "3 - Aux",
            "4 - Base", "4 - Aux",
            "5 - Base", "5 - Aux",
            "6 - Base", "6 - Aux",
            "7 - Base", "7 - Aux",
            "8 - Base", "8 - Aux",
            "9 - Base", "9 - Aux",
        ]
        cbar.set_ticklabels(c_labels)
        cbar.set_label("Classes")

        fig.suptitle(f"Base - {base.capitalize()} / Auxiliary - {aux.capitalize()}")

        plt.tight_layout()
        plt.savefig(filename, format="pdf", dpi=300)


class EBSW_Plotter:
    def __init__(self, dataloaders: DataLoaderSet, batch_size: int):
        self.val_loader = dataloaders.val_loader
        self.batch_size = batch_size

    def run_isebsw(self, model, dataloader, layers, device, base_only=True, unet_model=None, num_projections=128):
        dataloader.unique_sources = True
        model.to(device)
        model.eval()

        if unet_model is not None:
            unet_model.to(device)
            unet_model.eval()

        if isinstance(layers, int):
            layers = [i for i in range(layers+1)]

        total_isebsw_loss = np.zeros((len(dataloader), len(layers)))
        total_batches = 0
        
        for i, (base, aux, _) in enumerate(dataloader):
            if base_only:
                dataset_1, dataset_2 = torch.split(base, base.size(0) // 2)
            else:
                dataset_1, _ = torch.split(base, base.size(0) // 2)
                dataset_2, _ = torch.split(aux, aux.size(0) // 2)

            dataset_1 = dataset_1.to(device)
            dataset_2 = dataset_2.to(device)

            if unet_model and not base_only:
                dataset_2 = unet_model(dataset_2)
                
            dataset_1_outputs = model(dataset_1)[:layers[-1]+1]
            dataset_2_outputs = model(dataset_2)[:layers[-1]+1]

            for j, layer in enumerate(layers):
                dataset_1_layer_flat = dataset_1_outputs[layer].view(dataset_1_outputs[layer].size(0), -1)
                dataset_2_layer_flat = dataset_2_outputs[layer].view(dataset_2_outputs[layer].size(0), -1)

                total_isebsw_loss[i, j] += ISEBSW(
                    dataset_1_layer_flat,
                    dataset_2_layer_flat,
                    L=num_projections,
                    device=device
                ).item() / self.batch_size

            total_batches += 1

        return total_isebsw_loss

    def plot_ebsw(self, models: ModelSet, layers: list|int, device: str, filename: str, unet_models: ModelSet|None=None, num_projections: int=128):
        if unet_models == None:
            unet_models = ModelSet(
                base=None,
                mixed=None,
                contrast=None
            )

            base_ebsw_inter = self.run_isebsw(
                model=models.base,
                dataloader=self.val_loader,
                layers=layers,
                device=device,
                unet_model=unet_models.base,
                num_projections=num_projections
            )

            mixed_ebsw_inter = self.run_isebsw(
                model=models.mixed,
                dataloader=self.val_loader,
                layers=layers,
                device=device,
                unet_model=unet_models.mixed,
                num_projections=num_projections
            )

            contrast_ebsw_inter = self.run_isebsw(
                model=models.contrast,
                dataloader=self.val_loader,
                layers=layers,
                device=device,
                unet_model=unet_models.contrast,
                num_projections=num_projections
            )   

            base_ebsw_intra = self.run_isebsw(
                model=models.base,
                dataloader=self.val_loader,
                layers=layers,
                base_only=False,
                device=device,
                unet_model=unet_models.base,
                num_projections=num_projections
            )

            mixed_ebsw_intra = self.run_isebsw(
                model=models.mixed,
                dataloader=self.val_loader,
                layers=layers,
                base_only=False,
                device=device,
                unet_model=unet_models.mixed,
                num_projections=num_projections
            )

            contrast_ebsw_intra = self.run_isebsw(
                model=models.contrast,
                dataloader=self.val_loader,
                layers=layers,
                base_only=False,
                device=device,
                unet_model=unet_models.contrast,
                num_projections=num_projections
            )

            layer_measurements_inter = ModelSet(
                base = base_ebsw_inter,
                mixed = mixed_ebsw_inter,
                contrast = contrast_ebsw_inter
            ).dict()

            layer_measurements_intra = ModelSet(
                base = base_ebsw_intra,
                mixed = mixed_ebsw_intra,
                contrast = contrast_ebsw_intra
            ).dict()

            fig, axs = plt.subplots(1, 3, figsize=(20, 7))
            fig.suptitle('Energy-Based Sliced Wasserstein Loss Initial Layerwise Test Results', fontsize=22)

            categories = ['base', 'mixed', 'contrast']
            modes = ['Base/Base', 'Base/Aux']

            colors = ['#1f77b4', '#ff7f0e']  # Blue, Orange

            for i, cat in enumerate(categories):
                for j, mode in enumerate([layer_measurements_inter, layer_measurements_intra]):
                    data = mode[cat]
                    means = np.mean(data, axis=0)
                    stds = np.std(data, axis=0)

                    x = range(len(means))

                    axs[i].grid(True, linestyle=":", alpha=.7)

                    axs[i].errorbar(x, means, yerr=stds, capsize=5, marker='o', 
                                    color=colors[j], ecolor=colors[j], 
                                    markersize=8, linewidth=2, capthick=2,
                                    label=f'{modes[j]} Samples', linestyle='none')

                # Increase title font size
                axs[i].set_title(f'{cat.capitalize()} Model', fontsize=20)

                # Increase x-label font size
                axs[i].set_xlabel('Layer', fontsize=16)

                # Increase y-label font size
                axs[i].set_ylabel('Average Loss', fontsize=16)

                # Increase tick label font sizes
                axs[i].tick_params(axis='both', which='major', labelsize=14)

                # Add legend
                axs[i].legend(fontsize=12)

            plt.tight_layout()
            plt.savefig(filename, format="pdf", dpi=300)