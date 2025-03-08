import sys
sys.path.append('..')
import os

import torch
from dgp import get_dataloader
import os
from model import GPT
import pickle as pkl
from sae import SAE
import torch.nn.functional as F
from sklearn.cluster import spectral_clustering as sc
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
pca = PCA(n_components=3)

# Some global variables
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
pos_order = ['Noun', 'Pro.', 'Verb', 'Adv.', 'Adj.', 'Conj.']
titlefont = 10
axisfont = 8
fontname = 'sans-serif'
pos_colors = {
    'Noun': [1.        , 0.41176471, 0.70588235], 
    'Pro.': [0.66666667, 0.52418301, 0.80392157], 
    'Verb': [0.        , 0.74901961, 1. ], 
    'Adv.': [0.33333333, 0.63660131, 0.90196078], 
    'Adj.': [0.5882352941176471, 0.4627450980392157, 0.3843137254901961], 
    'Conj.': [0.7, 0.7, 0.7], 
    }
colors_list = torch.tensor([pos_colors[pos] for pos in pos_order]).to(device).float()


### Fetch model / SAE
def get_sae(sae_id, base_model_cfg):
    """
    Function to fetch the SAE model.
    """
    sae_dict = torch.load('../trained_saes/final/'+sae_id+'/latest_ckpt.pt', map_location='cpu', weights_only=False)
    sae_cfg = sae_dict['config']

    # Broad info to characterize the SAE
    sae_info = {}
    if sae_cfg.sae.sae_type == 'sparsemax_dist':
        sae_info['type'] = 'SpaDE'
        sae_info['gamma'] = sae_cfg.sae.gamma_reg
    elif sae_cfg.sae.sae_type == 'relu':
        sae_info['type'] = 'ReLU'
        sae_info['gamma'] = sae_cfg.sae.gamma_reg
    elif sae_cfg.sae.sae_type == 'jumprelu':
        sae_info['type'] = 'JumpReLU'
        sae_info['gamma'] = sae_cfg.sae.gamma_reg
    elif sae_cfg.sae.sae_type == 'topk':
        sae_info['type'] = 'TopK'
        sae_info['K'] = sae_cfg.sae.kval_topk

    # Load the SAE model
    sae = SAE(
            dimin=base_model_cfg.model.n_embd,
            width=base_model_cfg.model.n_embd*sae_cfg.sae.exp_factor, 
            sae_type=sae_cfg.sae.sae_type,
            kval_topk=sae_cfg.sae.kval_topk if sae_cfg.sae.sae_type=='topk' else None,
            normalize_decoder=(not sae_cfg.sae.sae_type == 'sparsemax_dist'),
            )
    sae.load_state_dict(sae_dict['sae'])

    return sae.eval().to(device), sae_info


### Functions to get activations or sparse codes
def get_pos_data(sequences):
    """
    Function to decompose the data into PoS.
    """
    pos_masks = {pos: [] for pos in pos_order}
    pos_idxs = {pos: [] for pos in pos_order}
    n_pos = {pos: 0 for pos in pos_order}

    for pos, (start, end) in zip(pos_masks.keys(), [(0, 10), (30, 40), (10, 20), (40, 50), (20, 30), (50, 60)]):
        print(f"Getting PoS mask for {pos}: {start}--{end-1}")

        # Get mask for PoS
        pos_masks[pos] = (sequences >= start) & (sequences < end) 

        # Sort by position in sequences
        pos_idxs[pos] = torch.sort(
            torch.nonzero(pos_masks[pos], as_tuple=False)[:,1]
            ).indices 
        
        # Count number of tokens
        n_pos[pos] = len(pos_idxs[pos])

    return pos_masks, pos_idxs, n_pos


def getActivations(base_model, sequences, pos_masks, pos_idxs):
    """
    Function to get activations from the model.
    """
    activations = [] 
    def getActivationHook():
        """
        Function to define hooks for storing activations.
        """
        def hook(model, input, output):
            activations.append(output.detach().clone())
        return hook

    pos_acts = {pos: [] for pos in pos_order}
    act_hook = base_model.transformer.h[0].register_forward_hook(getActivationHook())
    
    with torch.no_grad():
        base_model(sequences.to(device)) 
        token_activations = activations[0][sequences < 60].clone()
        for pos in pos_masks.keys():
            pos_acts[pos] = activations[0][pos_masks[pos]].clone()
            pos_acts[pos] = pos_acts[pos][pos_idxs[pos]]

    token_activations = []
    for pos in pos_order:
        token_activations.append(pos_acts[pos])
    token_activations = torch.cat(token_activations, dim=0)

    act_hook.remove()
    return token_activations, pos_acts



### Functions to compute the metrics
def compute_population_metrics(token_activations, sae):
    """
    Function to compute the variance explained, NMSE, L0, and percent dead neurons on entire dataset.
    """

    with torch.no_grad():
        pred_activations, latent_codes = sae(token_activations, return_hidden=True) 
        variance_remaining = ((pred_activations - token_activations).var() / token_activations.var()).item()
        nmse = ((pred_activations - token_activations).norm() / token_activations.norm()).item()
        L0 = (latent_codes > 1e-5).sum().item() / latent_codes.shape[0]
        percent_dead = ((latent_codes.sum(dim=0) < 1e-5).sum() / latent_codes.shape[1]).item() * 100

    return variance_remaining, nmse, L0, percent_dead
    

def compute_pos_latent_codes(pos_acts, pos_masks, sae):
    """
    Function to compute the variance explained, NMSE, L0, and percent dead neurons by PoS.
    """

    with torch.no_grad():
        pos_latent_codes = {}
        for pos in pos_masks.keys():
            _, pos_latent_codes[pos] = sae(pos_acts[pos], return_hidden=True) 
        
    return pos_latent_codes


def latents_characterization(pos_latent_codes, n_pos):
    """
    Function to characterize the latents.
    """
    # Stack of latent codes, sorted by PoS and position in sequence
    latents_stack = [] # (num_pos, num_latents)
    for pos in pos_order:
        latents_stack.append(pos_latent_codes[pos])
    latents_stack = torch.cat(latents_stack, dim=0)

    # Remove dead latents
    not_dead_latents = (latents_stack.sum(dim=0) > 1e-5)
    latents_stack = latents_stack[:, not_dead_latents]

    # Data and latent stacks
    data_stack = F.normalize(latents_stack, p=2, dim=1)
    latents_stack = F.normalize(latents_stack, p=2, dim=0)

    # Correlation maps
    data_corrmap = data_stack @ data_stack.T # (num_pos, num_pos)
    latents_corrmap = latents_stack.T @ latents_stack # (num_latents, num_latents)

    # Stable rank values
    s = data_corrmap.svd().S
    stable_rank_data = s.sum() / s[0]

    s = latents_corrmap.svd().S
    stable_rank_latents = s.sum() / s[0]

    # Check latent selectivity for each PoS and define a map from latent ID to PoS
    f1_scores = torch.zeros(latents_corrmap.shape[0], len(pos_order))
    start = 0
    for i, pos in enumerate(pos_order):
        n = n_pos[pos]
        f1_scores[:, i] = (data_stack[start:start+n].abs() > 1e-2).float().mean(dim=0)
        start += n

    latents_to_pos = torch.argmax(f1_scores, dim=1).to(device)
    
    return stable_rank_data.item(), data_corrmap.cpu().numpy(), stable_rank_latents.item(), latents_corrmap.cpu().numpy(), latents_to_pos, latents_stack, data_stack, f1_scores


### Plotting functions for data
def plot_data_corrmap(data_corrmap, sae_hparam, n_pos, save_path):
    """
    Function to plot the correlation map.
    """

    plt.figure(figsize=(5, 5), dpi=300)
    plt.imshow(data_corrmap, cmap='magma', vmin=0, vmax=1)
    plt.colorbar(shrink=0.8)
    plt.title(sae_hparam, fontsize=titlefont, fontname=fontname)

    # Add red lines to show PoS boundaries
    start_point = 0
    for pos in pos_order:
        n = n_pos[pos]
        plt.text(-200, start_point + n//2, pos, fontsize=axisfont, color='k', fontname=fontname)
        plt.text(start_point + n//4, data_corrmap.shape[0] + 100, pos, fontsize=axisfont, color='k', fontname=fontname)
        if start_point + n == data_corrmap.shape[0]:
            break
        plt.axvline(x=start_point + n-1, color='#EFEFEF', linestyle='-', linewidth=1)
        plt.axhline(y=start_point + n-1, color='#EFEFEF', linestyle='-', linewidth=1)
        start_point += n

    # Remove xticks and yticks
    plt.xticks([])
    plt.yticks([])

    # Save the plot
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_data_pca(data_stack, latents_to_pos, sae_hparam, save_path):
    """
    Function to plot the PCA.
    """

    ##### Run PCA
    data_pca = pca.fit_transform(data_stack.cpu().numpy()) # (num_pos, 2)

    # Define color by mixing PoS colors according to latent codes
    data_to_color = torch.zeros(data_stack.shape[0], 3)
    for i in range(data_stack.shape[0]):
        data_to_color[i] = colors_list[latents_to_pos[data_stack[i] > 0]].mean(dim=0)
    data_to_color = data_to_color.cpu().numpy()

    ##### 2D PCA
    fig = plt.figure(figsize=(4, 3), dpi=300)
    ax1 = fig.add_subplot(111)

    for i, d in enumerate(data_pca):
        ax1.scatter(d[0], d[1], color=data_to_color[i], s=8, alpha=1)

    # Define a legend using PoS colors
    for pos in pos_order:
        ax1.scatter([], [], color=pos_colors[pos], label=pos)

    # Set axis labels
    ax1.set_xlabel('PC1: {:.2f}'.format(pca.explained_variance_ratio_[0]), fontsize=titlefont, fontname=fontname)
    ax1.set_ylabel('PC2: {:.2f}'.format(pca.explained_variance_ratio_[1]), fontsize=titlefont, fontname=fontname)

    # Make gridlines thinner and lighter
    ax1.grid(False)

    fig.suptitle(sae_hparam, fontsize=titlefont, fontname=fontname)
    ax1.legend(frameon=False, fontsize=axisfont)
 
    # Save the plot
    plt.savefig(save_path+'_2D.png', bbox_inches='tight', dpi=300)
    plt.close()


    ##### 3D PCA
    fig = plt.figure(figsize=(5, 4), dpi=300)
    ax2 = fig.add_subplot(111, projection='3d')

    # Scatter plot
    for i, d in enumerate(data_pca):
        ax2.scatter(d[0], d[1], d[2], color=data_to_color[i], s=8, alpha=1)

    # Define a legend using PoS colors
    for pos in pos_order:
        ax2.scatter([], [], color=pos_colors[pos], label=pos)

    # Put fewer ticks
    ax2.locator_params(axis='x', nbins=4)
    ax2.locator_params(axis='y', nbins=4)
    ax2.locator_params(axis='z', nbins=4)

    # Set axis labels
    ax2.set_xlabel('PC1: {:.2f}'.format(pca.explained_variance_ratio_[0]), fontsize=titlefont, fontname=fontname)
    ax2.set_ylabel('PC2: {:.2f}'.format(pca.explained_variance_ratio_[1]), fontsize=titlefont, fontname=fontname)
    ax2.set_zlabel('PC3: {:.2f}'.format(pca.explained_variance_ratio_[2]), fontsize=titlefont, fontname=fontname)

    # Rotate the plot so that z-axis is on left-hand side
    ax2.view_init(elev=10, azim=260)

    # Set title and legend
    fig.suptitle(sae_hparam, fontsize=titlefont, fontname=fontname, y=0.78)
    ax2.legend(frameon=False, loc=[0.92, 0.4], fontsize=axisfont)

    # Make gridlines thinner and lighter
    ax2.xaxis._axinfo["grid"].update({"linewidth":0.5, "linestyle":'--'})
    ax2.yaxis._axinfo["grid"].update({"linewidth":0.5, "linestyle":'--'})
    ax2.zaxis._axinfo["grid"].update({"linewidth":0.5, "linestyle":'--'})

    # Remove background color
    ax2.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax2.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax2.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    # Save the plot
    plt.savefig(save_path+'_3D.png', bbox_inches='tight', dpi=300)
    plt.close()


### Plotting functions for latents
def plot_latents_corrmap(latents_corrmap, latents_to_pos, latents_stack, sae_hparam, save_path):
    """
    Function to plot the correlation map.
    """

    # Run spectral clustering on latents to identify clusters (Intentionally over-cluster the data to find fine-grained clusters)
    cluster_idx = sc(latents_corrmap, n_clusters=20, random_state=0)
    cluster_idx = torch.tensor(cluster_idx).to(device)

    # Map PoS to cluster IDs that maximally represent the PoS (defined via F1 scores)
    pos_to_clusterid = {}
    for i, pos in enumerate(pos_order):
        pos_to_clusterid[pos] = np.unique(cluster_idx[latents_to_pos == i].cpu().numpy())

    # Reorder latents by PoS according to list identified above
    new_order = []
    done_clusters = []
    n_pos_latents = []
    pos_to_latentidx = {pos: [] for pos in pos_order}
    for pos in pos_order:
        n_pos = 0
        for cluster_id in pos_to_clusterid[pos]:
            if cluster_id in done_clusters:
                continue
            done_clusters.append(cluster_id)
            locs = torch.where(cluster_idx == cluster_id)[0]
            pos_to_latentidx[pos].extend(locs.cpu().numpy())
            new_order.extend(locs.cpu().numpy())
            n_pos += len(locs)
        n_pos_latents.append(n_pos)

    # Reorder latents stack and compute correlation map
    reordered_latents_stack = latents_stack[:, new_order]
    reordered_latents_corrmap = reordered_latents_stack.T @ reordered_latents_stack

    ## Plotting
    plt.figure(figsize=(5, 5), dpi=300)
    plt.imshow(reordered_latents_corrmap.cpu().numpy(), cmap='magma', vmin=0, vmax=1)
    plt.colorbar(shrink=0.8)
    plt.title(sae_hparam, fontsize=titlefont, fontname=fontname)

    # Add red lines to show PoS boundaries
    start = 0
    for i, pos in enumerate(pos_order):
        n_pos = n_pos_latents[i]
        plt.text(-30, start + n_pos//2, pos, fontsize=axisfont, color='k', fontname=fontname)
        plt.text(start + n_pos//2, reordered_latents_corrmap.shape[0] + 20, pos, 
                fontsize=axisfont, color='k', fontname=fontname)
        if start + n_pos == reordered_latents_corrmap.shape[0]:
            break
        plt.axvline(x=start + n_pos-1, color='#EFEFEF', linestyle='-', linewidth=1)
        plt.axhline(y=start + n_pos-1, color='#EFEFEF', linestyle='-', linewidth=1)
        start += n_pos

    # Remove xticks and yticks
    plt.xticks([])
    plt.yticks([])

    # Save the plot
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_latents_pca(latents_to_pos, f1_scores, latents_stack, sae_hparam, save_path):
    """
    Function to plot the PCA.
    """
    l = latents_to_pos.tolist() # (num_latents)
    f = (f1_scores / f1_scores.max(dim=0, keepdim=True).values) # (num_latents, num_pos)

    # Retrieve F1 scores for each latent for its corresponding PoS
    f1 = torch.zeros(l.__len__())
    for i, pos in enumerate(l):
        f1[i] = f[i, pos]

    # Define color to latent codes
    latents_to_color = colors_list[latents_to_pos].cpu().numpy()
    latents_to_size = 30*f1.cpu().numpy()

    # PCA
    latents_pca = pca.fit_transform(latents_stack.T.cpu().numpy()) # (num_pos, 2)

    #### 2D PCA
    fig = plt.figure(figsize=(4, 3), dpi=300)
    ax1 = fig.add_subplot(111)

    # Scatter plot for 2D and 3D
    for i, d in enumerate(latents_pca):
        ax1.scatter(d[0], d[1], color=latents_to_color[i], s=latents_to_size[i])

    # Define a legend using PoS colors
    for pos in pos_order:
        ax1.scatter([], [], color=pos_colors[pos], label=pos)

    # Set axis labels
    ax1.set_xlabel('PC1: {:.2f}'.format(pca.explained_variance_ratio_[0]), fontsize=axisfont, fontname=fontname)
    ax1.set_ylabel('PC2: {:.2f}'.format(pca.explained_variance_ratio_[1]), fontsize=axisfont, fontname=fontname)

    # Make gridlines thinner and lighter
    ax1.grid(False)
    fig.suptitle(sae_hparam, fontsize=titlefont, fontname=fontname)
    ax1.legend(frameon=False, fontsize=axisfont)
 
    # Save the plot
    plt.savefig(save_path+'_2D.png', bbox_inches='tight', dpi=300)
    plt.close()


    ##### 3D PCA
    fig = plt.figure(figsize=(4, 4), dpi=300)
    ax2 = fig.add_subplot(111, projection='3d')

    # Scatter plot
    for i, d in enumerate(latents_pca):
        ax2.scatter(d[0], d[1], d[2], color=latents_to_color[i], s=latents_to_size[i], alpha=1)

    # Define a legend using PoS colors
    for pos in pos_order:
        ax2.scatter([], [], color=pos_colors[pos], label=pos)

    # Put fewer ticks
    ax2.locator_params(axis='x', nbins=4)
    ax2.locator_params(axis='y', nbins=4)
    ax2.locator_params(axis='z', nbins=4)

    # Set axis labels
    ax2.set_xlabel('PC1: {:.2f}'.format(pca.explained_variance_ratio_[0]), fontsize=titlefont, fontname=fontname)
    ax2.set_ylabel('PC2: {:.2f}'.format(pca.explained_variance_ratio_[1]), fontsize=titlefont, fontname=fontname)
    ax2.set_zlabel('PC3: {:.2f}'.format(pca.explained_variance_ratio_[2]), fontsize=titlefont, fontname=fontname)

    # Rotate the plot so that z-axis is on left-hand side
    ax2.view_init(elev=10, azim=260)

    # Set title and legend
    fig.suptitle(sae_hparam, fontsize=titlefont, fontname=fontname, y=0.78)
    ax2.legend(frameon=False, loc=[0.92, 0.4], fontsize=axisfont)

    # Make gridlines thinner and lighter
    ax2.xaxis._axinfo["grid"].update({"linewidth":0.5, "linestyle":'--'})
    ax2.yaxis._axinfo["grid"].update({"linewidth":0.5, "linestyle":'--'})
    ax2.zaxis._axinfo["grid"].update({"linewidth":0.5, "linestyle":'--'})

    # Remove background color
    ax2.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax2.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax2.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path+'_3D.png', bbox_inches='tight', dpi=300)
    plt.close()


# Main function
if __name__ == "__main__":

    # Load the base model
    base_model_dir = '../trained_models/final/5fr59ylf'
    base_model_dict = torch.load(os.path.join(base_model_dir, 'latest_ckpt.pt'), map_location='cpu', weights_only=False)
    base_model_cfg = base_model_dict['config']

    with open(os.path.join(base_model_dir, 'grammar/PCFG.pkl'), 'rb') as f:
        pcfg = pkl.load(f)
    base_model = GPT(base_model_cfg.model, pcfg.vocab_size)
    base_model.load_state_dict(base_model_dict['net'])
    base_model.to(device)
    base_model = base_model.eval()

    # Get the dataloader
    dataloader = get_dataloader(
        language=base_model_cfg.data.language,
        config=base_model_cfg.data.config,
        alpha=base_model_cfg.data.alpha,
        prior_type=base_model_cfg.data.prior_type,
        num_iters=base_model_cfg.data.num_iters * base_model_cfg.data.batch_size,
        max_sample_length=base_model_cfg.data.max_sample_length,
        seed=base_model_cfg.seed,
        batch_size=256,
        num_workers=4,
    )

    for sequences, seq_lengths in dataloader:
        break

    # Define per PoS masks
    pos_masks, pos_idxs, n_pos = get_pos_data(sequences)

    # Get the activations
    token_activations, pos_acts = getActivations(base_model, sequences, pos_masks, pos_idxs)

    # Compile results
    results_dict = {'ReLU': {}, 'JumpReLU': {}, 'SpaDE': {}, 'TopK': {}}
    for sae_id in ['ftg9224x','043mcjiv','5ovxwk0l','j3mwrie4','b87pr4vt','moz259dl',
                   'ldx6ldtt','yyfn9ayz','kzoi5gtj','g5rsdm25','p9sk9kne','w7ukpaol',
                   '8fffwb9h','hhhc77le','fz5wun7o','10k1e6dl','6xk5r9rz','ozskq6ir',
                   'dkzwtbvr','zpl33gr8', 'z9r8qnqe', 'za0zh9yj', '898iol2k']:

        # Define SAE
        sae, sae_info = get_sae(sae_id, base_model_cfg)

        # Compute metrics over the population
        _, nmse, L0, percent_dead = compute_population_metrics(token_activations, sae)

        # Compute metrics by PoS
        pos_latent_codes = compute_pos_latent_codes(pos_acts, pos_masks, sae)

        # Stable rank and monosemanticity
        stable_rank_data, data_corrmap, stable_rank_latents, latents_corrmap, latents_to_pos, latents_stack, data_stack, f1_scores = latents_characterization(pos_latent_codes, n_pos)

        # Save directory
        save_path = 'corrmaps/{}/'.format(sae_info['type'])
        os.makedirs(save_path, exist_ok=True)

        # Plot the correlation map
        save_path = save_path+'K_{}'.format(sae_info['K']) if sae_info['type'] == 'TopK' else save_path+'gamma_{}'.format(sae_info['gamma'])
        sae_hparam = r'$L_0={:.1f}$'.format(L0)
        # if sae_info['type'] == 'TopK':
        #     sae_hparam = r'$k={}$'.format(sae_info['K'])
        # else:
        #     sae_hparam = r'$\gamma={}$'.format(sae_info['gamma'])

        # Plot the data and latents
        plot_data_corrmap(data_corrmap, sae_hparam, n_pos, save_path+'_data.png')
        plot_data_pca(data_stack, latents_to_pos, sae_hparam, save_path+'_data_pca')
        plot_latents_corrmap(latents_corrmap, latents_to_pos, latents_stack, sae_hparam, save_path+'_latents.png')
        plot_latents_pca(latents_to_pos, f1_scores, latents_stack, sae_hparam, save_path+'_latents_pca')