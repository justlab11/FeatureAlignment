import torch
import numpy as np
import logging
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


from losses import ISEBSW, mmdfuse
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


class EBSW_Plotter:
    def __init__(self, dataloaders: DataLoaderSet, batch_size: int):
        self.val_loader = dataloaders.val_loader
        self.batch_size = batch_size

    def run_isebsw(self, model, dataloader, layers, device, base_only=True, unet_model=None, num_projections=128):
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
                splits = torch.split(base, base.size(0) // 2)
                dataset_1, dataset_2 = splits[0], splits[1]
            else:
                splits_base = torch.split(base, base.size(0) // 2)
                splits_aux = torch.split(aux, aux.size(0) // 2)

                dataset_1, dataset_2 = splits_base[0], splits_aux[0]

            dataset_1 = dataset_1.to(device)
            dataset_2 = dataset_2.to(device)

            if unet_model and not base_only:
                dataset_2 = unet_model(dataset_2)[-1]
                
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
    
    def run_mmdfuse(self, model, dataloader, layers, device, base_only=True, unet_model=None):
        model.to(device)
        model.eval()

        if unet_model is not None:
            unet_model.to(device)
            unet_model.eval()

        if isinstance(layers, int):
            layers = [i for i in range(layers+1)]

        total_mmdfuse_loss = np.zeros((len(dataloader), len(layers)))
        total_batches = 0
        
        for i, (base, aux, _) in enumerate(dataloader):
            if base_only:
                splits = torch.split(base, base.size(0) // 2)
                dataset_1, dataset_2 = splits[0], splits[1]
            else:
                splits_base = torch.split(base, base.size(0) // 2)
                splits_aux = torch.split(aux, aux.size(0) // 2)

                dataset_1, dataset_2 = splits_base[0], splits_aux[0]

            dataset_1 = dataset_1.to(device)
            dataset_2 = dataset_2.to(device)

            if unet_model and not base_only:
                dataset_2 = unet_model(dataset_2)[-1]
                
            dataset_1_outputs = model(dataset_1)[:layers[-1]+1]
            dataset_2_outputs = model(dataset_2)[:layers[-1]+1]

            for j, layer in enumerate(layers):
                dataset_1_layer_flat = dataset_1_outputs[layer].view(dataset_1_outputs[layer].size(0), -1)
                dataset_2_layer_flat = dataset_2_outputs[layer].view(dataset_2_outputs[layer].size(0), -1)

                total_mmdfuse_loss[i, j] += mmdfuse(
                    dataset_1_layer_flat,
                    dataset_2_layer_flat,
                    device=device
                ).item() / self.batch_size

            total_batches += 1

        return total_mmdfuse_loss

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
                base_only=True,
                device=device,
                unet_model=unet_models.base,
                num_projections=num_projections
            )

            mixed_ebsw_inter = self.run_isebsw(
                model=models.mixed,
                dataloader=self.val_loader,
                layers=layers,
                base_only=True,
                device=device,
                unet_model=unet_models.mixed,
                num_projections=num_projections
            )

            contrast_ebsw_inter = self.run_isebsw(
                model=models.contrast,
                dataloader=self.val_loader,
                layers=layers,
                base_only=True,
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
            modes = ['Target/Target', 'Target/Source']

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

def plot_examples(dataset, unet_model, filename, device):
    unet_model.eval()
    # Randomly select 10 samples
    num_samples = 10
    random_samples = np.random.choice(len(dataset), num_samples, replace=False)

    #dataset.dataset.unique_sources = True

    # Create a figure with 10 columns and 3 rows
    fig, axes = plt.subplots(3, num_samples, figsize=(20, 6))

    for i, sample_idx in enumerate(random_samples):
        img_one, img_two, label = dataset[sample_idx]
        img_three = img_two.unsqueeze(0).to(device)
        img_three = unet_model(img_three)[-1][0]

        img_one = np.transpose(img_one, (1, 2, 0))
        img_two = np.transpose(img_two, (1, 2, 0))
        img_three = np.transpose(img_three.detach().cpu(), (1, 2, 0))
        
        # Plot img_one in the first row
        axes[0, i].imshow(img_one)
        axes[0, i].axis('off')
        axes[0, i].set_title(f"Sample {i+1}\nLabel: {label}")
        
        # Plot img_two in the second row
        axes[1, i].imshow(img_two)
        axes[1, i].axis('off')

        axes[2, i].imshow(img_three)
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.savefig(filename, format="pdf", dpi=300)
