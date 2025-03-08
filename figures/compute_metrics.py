import sys
sys.path.append('..')

import torch
from dgp import get_dataloader
import os
from model import GPT
import pickle as pkl
from sae import SAE
import torch.nn.functional as F
from sklearn.cluster import spectral_clustering as sc
import numpy as np

# Some global variables
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pos_order = ['Noun', 'Pro.', 'Verb', 'Adv.', 'Adj.', 'Conj.']


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

    for pos, (start, end) in zip(pos_masks.keys(), [(0, 10), (30, 40), (10, 20), (40, 50), (20, 30), (50, 60)]):
        print(f"Getting PoS mask for {pos}: {start}--{end-1}")

        # Get mask for PoS
        pos_masks[pos] = (sequences >= start) & (sequences < end) 

        # Sort by position in sequences
        pos_idxs[pos] = torch.sort(
            torch.nonzero(pos_masks[pos], as_tuple=False)[:,1]
            ).indices 

    return pos_masks, pos_idxs


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
    

def compute_pos_metrics(pos_acts, pos_masks, sae):
    """
    Function to compute the variance explained, NMSE, L0, and percent dead neurons by PoS.
    """

    with torch.no_grad():
        pos_latent_codes = {}
        results_by_pos = {}
        for pos in pos_masks.keys():
            results_by_pos[pos] = {'n_tokens': pos_acts[pos].shape[0]}
            pos_pred_activations, pos_latent_codes[pos] = sae(pos_acts[pos], return_hidden=True) 
            results_by_pos[pos]['variance_remaining'] = ((pos_pred_activations - pos_acts[pos]).var() / pos_acts[pos].var()).item()
            results_by_pos[pos]['nmse'] = ((pos_pred_activations - pos_acts[pos]).norm() / pos_acts[pos].norm()).item()
            results_by_pos[pos]['L0'] = (pos_latent_codes[pos] > 1e-5).sum().item() / pos_latent_codes[pos].shape[0]
            results_by_pos[pos]['percent_dead'] = ((pos_latent_codes[pos].sum(dim=0) < 1e-5).sum() / pos_latent_codes[pos].shape[1]).item() * 100
        
    return pos_latent_codes, results_by_pos


def latents_characterization(pos_latent_codes, is_relu=False):
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

    # F1 scores and monosemanticity
    precision = torch.zeros(latents_stack.shape[1], len(pos_order)-1) # (num_latents, num_pos)
    recall = torch.zeros(latents_stack.shape[1], len(pos_order)-1) # (num_latents, num_pos)
    times_a_latent_was_active = (latents_stack > 1e-5).sum(dim=0) # (num_latents)
    f1_scores_per_concept = {}
    start = 0
    for i, pos in enumerate(['Noun', 'Verb', 'Adv.', 'Adj.', 'Conj.']):
        f1_scores_per_concept[pos] = {'mean': [], 'std': []}
        n = pos_latent_codes[pos].shape[0] # Number of tokens for this PoS
        if pos == 'Noun':
            n += pos_latent_codes['Pro.'].shape[0] # Number of tokens for this PoS
        pos_activity = (latents_stack[start:start+n] > 1e-3).sum(dim=0) # Was a latent active for this PoS?
        precision[:,i] = pos_activity / (times_a_latent_was_active+1e-5)
        recall[:,i] = pos_activity / n
        start += n

    # Compute monosemanticity (defined as average F1 score across latents)
    f1_scores_per_latent = 2 * (precision * recall) / (precision + recall + 1e-5)
    f1_scores_per_latent /= (f1_scores_per_latent.sum(dim=1, keepdim=True) + 1e-5)
    mean_monosemanticity = (f1_scores_per_latent.max(dim=1).values).mean()
    std_monosemanticity = (f1_scores_per_latent.max(dim=1).values).std()

    # Populate F1 scores per concept
    K = 15 #if is_relu else 15
    for i, pos in enumerate(['Noun', 'Verb', 'Adv.', 'Adj.', 'Conj.']):
        f1_scores_per_concept[pos]['mean'] = torch.topk(f1_scores_per_latent[:,i], k=K, dim=0).values.mean().item()
        f1_scores_per_concept[pos]['std'] = torch.topk(f1_scores_per_latent[:,i], k=K, dim=0).values.std().item()

    return stable_rank_data.item(), stable_rank_latents.item(), f1_scores_per_concept, mean_monosemanticity, std_monosemanticity


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
    pos_masks, pos_idxs = get_pos_data(sequences)

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
        variance_remaining, nmse, L0, percent_dead = compute_population_metrics(token_activations, sae)

        # Compute metrics by PoS
        pos_latent_codes, pos_results = compute_pos_metrics(pos_acts, pos_masks, sae)

        # Stable rank and monosemanticity
        stable_rank_data, stable_rank_latents, f1_scores_per_concept, mean_monosemanticity, std_monosemanticity = latents_characterization(pos_latent_codes, is_relu=(sae_info['type']=='ReLU'))

        # Assemble results
        name = str(sae_info['K']) if sae_info['type'] == 'TopK' else str(sae_info['gamma'])
        results_dict[sae_info['type']].update({
            name: {
                'run_id': sae_id,
                'variance_remaining': variance_remaining,
                'nmse': nmse,
                'L0': L0,
                'percent_dead': percent_dead,
                'stable_rank_data': stable_rank_data,
                'stable_rank_latents': stable_rank_latents,
                'mean_monosemanticity': mean_monosemanticity,
                'std_monosemanticity': std_monosemanticity,
                'per_pos_results': pos_results,
                'f1_scores': f1_scores_per_concept,
                }
            })

    # Save results
    with open('results.pkl', 'wb') as f:
        pkl.dump(results_dict, f)