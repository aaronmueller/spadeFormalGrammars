import sys
# sys.path.append("./dictionary_learning/")
# sys.path.append("./spadeFormalGrammars/")

from collections import namedtuple
# from dictionary_learning import AutoEncoder, JumpReluAutoEncoder
from sae import SAE, step_fn
from utils.loading import load_sae_inference_only

# RelaxedArchetypalAutoEncoder
# from dictionary_learning.dictionary import IdentityDict
from dataset import Submodule, Dataset
from typing import Literal
from nnsight import LanguageModel
from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sns
import torch as t
import math
import random
import numpy as np
from huggingface_hub import list_repo_files
from tqdm import tqdm
import os
import argparse
sns.set()
sns.set_style("whitegrid")

DICT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/dictionaries"

DictionaryStash = namedtuple("DictionaryStash", ["embed", "attns", "mlps", "resids"])


def _load_pythia_saes_and_submodules(
    model,
    thru_layer: int | None = None,
    separate_by_type: bool = False,
    include_embed: bool = True,
    neurons: bool = False,
    dtype: t.dtype = t.float32,
    device: t.device = t.device("cpu"),
):
    assert len(model.gpt_neox.layers) == 6, "Not the expected number of layers for pythia-70m-deduped"
    if thru_layer is None:
        thru_layer = len(model.gpt_neox.layers)

    attns = []
    mlps = []
    resids = []
    dictionaries = {}
    if include_embed:
        embed = Submodule(
            name = "embed",
            submodule=model.gpt_neox.embed_in,
        )
        if not neurons:
            dictionaries[embed] = AutoEncoder.from_pretrained(
                f"{DICT_DIR}/pythia-70m-deduped/embed/10_32768/ae.pt",
                dtype=dtype,
                device=device,
            )
        else:
            dictionaries[embed] = IdentityDict(512)
    else:
        embed = None
    for i, layer in enumerate(model.gpt_neox.layers[:thru_layer+1]):
        attns.append(
            attn := Submodule(
                name = f"attn_{i}",
                submodule=layer.attention,
                is_tuple=True,
            )
        )
        mlps.append(
            mlp := Submodule(
                name = f"mlp_{i}",
                submodule=layer.mlp,
            )
        )
        resids.append(
            resid := Submodule(
                name = f"resid_{i}",
                submodule=layer,
                is_tuple=True,
            )
        )
        if not neurons:
            dictionaries[attn] = AutoEncoder.from_pretrained(
                f"{DICT_DIR}/pythia-70m-deduped/attn_out_layer{i}/10_32768/ae.pt",
                dtype=dtype,
                device=device,
            )
            dictionaries[mlp] = AutoEncoder.from_pretrained(
                f"{DICT_DIR}/pythia-70m-deduped/mlp_out_layer{i}/10_32768/ae.pt",
                dtype=dtype,
                device=device,
            )
            dictionaries[resid] = AutoEncoder.from_pretrained(
                f"{DICT_DIR}/pythia-70m-deduped/resid_out_layer{i}/10_32768/ae.pt",
                dtype=dtype,
                device=device,
            )
        else:
            dictionaries[attn] = IdentityDict(512)
            dictionaries[mlp] = IdentityDict(512)
            dictionaries[resid] = IdentityDict(512)

    if separate_by_type:
        return DictionaryStash(embed, attns, mlps, resids), dictionaries
    else:
        submodules = (
            [embed] if include_embed else []
         ) + [
            x for layer_dictionaries in zip(attns, mlps, resids) for x in layer_dictionaries
        ]
        return submodules, dictionaries


def load_gemma_sae(
    submod_type: Literal["embed", "attn", "mlp", "resid"],
    layer: int,
    width: Literal["16k", "65k"] = "16k",
    neurons: bool = False,
    dtype: t.dtype = t.float32,
    device: t.device = t.device("cpu"),
):
    if neurons:
        if submod_type != "attn":
            return IdentityDict(2304)
        else:
            return IdentityDict(2048)

    repo_id = "google/gemma-scope-2b-pt-" + (
        "res" if submod_type in ["embed", "resid"] else
        "att" if submod_type == "attn" else
        "mlp"
    )
    if submod_type != "embed":
        directory_path = f"layer_{layer}/width_{width}"
    else:
        directory_path = "embedding/width_4k"

    files_with_l0s = [
        (f, int(f.split("_")[-1].split("/")[0]))
        for f in list_repo_files(repo_id, repo_type="model", revision="main")
        if f.startswith(directory_path) and f.endswith("params.npz")
    ]
    optimal_file = min(files_with_l0s, key=lambda x: abs(x[1] - 100))[0]
    optimal_file = optimal_file.split("/params.npz")[0]
    return JumpReluAutoEncoder.from_pretrained(
        load_from_sae_lens=True,
        release=repo_id.split("google/")[-1],
        sae_id=optimal_file,
        dtype=dtype,
        device=device,
    )


def _load_gemma_saes_and_submodules(
    model,
    thru_layer: int | None = None,
    separate_by_type: bool = False,
    include_embed: bool = True,
    neurons: bool = False,
    dtype: t.dtype = t.float32,
    device: t.device = t.device("cpu"),
):
    assert len(model.model.layers) == 26, "Not the expected number of layers for Gemma-2-2B"
    if thru_layer is None:
        thru_layer = len(model.model.layers)
    
    attns = []
    mlps = []
    resids = []
    dictionaries = {}
    if include_embed:
        embed = Submodule(
            name = "embed",
            submodule=model.model.embed_tokens,
        )
        dictionaries[embed] = load_gemma_sae("embed", 0, neurons=neurons, dtype=dtype, device=device)
    else:
        embed = None
    for i, layer in tqdm(enumerate(model.model.layers[:thru_layer+1]), total=thru_layer+1, desc="Loading Gemma SAEs"):
        attns.append(
            attn := Submodule(
                name=f"attn_{i}",
                submodule=layer.self_attn.o_proj,
                use_input=True
            )
        )
        dictionaries[attn] = load_gemma_sae("attn", i, neurons=neurons, dtype=dtype, device=device)
        mlps.append(
            mlp := Submodule(
                name=f"mlp_{i}",
                submodule=layer.post_feedforward_layernorm,
            )
        )
        dictionaries[mlp] = load_gemma_sae("mlp", i, neurons=neurons, dtype=dtype, device=device)
        resids.append(
            resid := Submodule(
                name=f"resid_{i}",
                submodule=layer,
                is_tuple=True,
            )
        )
        dictionaries[resid] = load_gemma_sae("resid", i, neurons=neurons, dtype=dtype, device=device)

    if separate_by_type:
        return DictionaryStash(embed, attns, mlps, resids), dictionaries
    else:
        submodules = (
            [embed] if include_embed else []
        )+ [
            x for layer_dictionaries in zip(attns, mlps, resids) for x in layer_dictionaries
        ]
        return submodules, dictionaries


def load_saes_and_submodules(
    model,
    thru_layer: int | None = None,
    separate_by_type: bool = False,
    include_embed: bool = True,
    neurons: bool = False,
    dtype: t.dtype = t.float32,
    device: t.device = t.device("cpu"),
):
    model_name = model.config._name_or_path

    if model_name == "EleutherAI/pythia-70m-deduped":
        return _load_pythia_saes_and_submodules(model, thru_layer=thru_layer, separate_by_type=separate_by_type, include_embed=include_embed, neurons=neurons, dtype=dtype, device=device)
    elif model_name == "google/gemma-2-2b":
        return _load_gemma_saes_and_submodules(model, thru_layer=thru_layer, separate_by_type=separate_by_type, include_embed=include_embed, neurons=neurons, dtype=dtype, device=device)
    else:
        raise ValueError(f"Model {model_name} not supported")


def get_mid_layer(submodules, layer):
    target = f"resid_{layer}"
    for submodule in submodules:
        if submodule.name == target:
            return submodule


def get_activations(model, submodule, dictionary, batch):
    with t.no_grad(), model.trace(batch):
        x = submodule.get_activation()
        x_saved = x.save()
    x_hat, f = dictionary(x_saved.value, return_hidden=True)
    # f_saved = f.save()
    return f.detach()


def load_dataset():
    return Dataset("../data/labeled_sentences.jsonl")


def score_identification(acts, labels, lamda=0.1, metric="accuracy"):
    scores = {}
    top_features = {}
    labels = {k: v for k, v in labels.items() if k not in ("formality-high", "formality-neutral", "reading-level-low", "reading-level-high")}
    label_matrix = t.stack([t.Tensor(labels[l]) for l in labels], dim=0)    # N x L

    for label_name in labels:
        if metric == "mcc":
            label_vec = t.Tensor(labels[label_name])   # N
        else:
            label_vec = t.tensor(labels[label_name])
        feature_labels = acts.T > lamda     # F x N
        if metric == "accuracy":
            matches = (feature_labels == label_vec)
            accuracies = matches.sum(dim=1) / label_vec.shape[-1]
            accuracy = accuracies.max()
            top_features[label_name] = accuracies.argmax()
            scores[label_name] = accuracy
        elif metric == "macrof1":
            # Calculate true positives, false positives, false negatives for each feature
            true_positives = (feature_labels & label_vec).sum(dim=1).float()  # F
            false_positives = (feature_labels & ~label_vec).sum(dim=1).float()  # F
            false_negatives = (~feature_labels & label_vec).sum(dim=1).float()  # F
            
            # Calculate precision and recall
            precision = true_positives / (true_positives + false_positives + 1e-10)  # F
            recall = true_positives / (true_positives + false_negatives + 1e-10)  # F
            
            # Calculate F1 scores
            f1_scores = 2 * precision * recall / (precision + recall + 1e-10)  # F
            
            # Find the feature with the max F1 score
            top_feature = f1_scores.argmax()
            max_f1 = f1_scores[top_feature]
            
            top_features[label_name] = top_feature
            scores[label_name] = max_f1
        elif metric == "mcc":
            acts_centered = acts - acts.mean(dim=0, keepdim=True)
            acts_std = acts_centered.norm(dim=0, keepdim=True)
            label_matrix_centered = label_matrix.T - label_matrix.T.mean(dim=0, keepdim=True)
            label_matrix_std = label_matrix_centered.norm(dim=0, keepdim=True)
            # Correct correlation computation
            numerator = acts_centered.T @ label_matrix_centered  # F × L
            denominator = acts_std.T * label_matrix_std  # F × L (broadcasting)

            mask = denominator != 0     # prevent NaNs
            corr_matrix = t.zeros_like(numerator)
            corr_matrix[mask] = numerator[mask] / denominator[mask]

            # Get indices of maximum correlations for each label
            top_feature_indices = corr_matrix.argmax(dim=0)  # Returns indices, shape: (L,)
            top_features = {label_name: top_feature_indices[i].item() for i, label_name in enumerate(list(labels))}

            return corr_matrix, top_features
        else:
            raise ValueError(f"Unrecognized metric: {metric}")

    return scores, top_features


def score_sensitivity(acts, labels, feature_idx, lamda=0.1, target_label="domain-science"):
    # First, find sentences where all labels are the same except the target_label
    print(acts.sum())
    prefix = target_label.split("-")[0]
    label_present = t.Tensor(labels[target_label]).nonzero().squeeze().tolist()
    not_label_present = set(list(range(acts.shape[0]))).difference(label_present)
    pair_indices = []
    for idx1 in label_present:
        label_vec1 = t.Tensor([labels[l][idx1] for l in labels.keys() if not l.startswith(prefix)])
        for idx2 in not_label_present:
            label_vec2 = t.Tensor([labels[l][idx2] for l in labels.keys() if not l.startswith(prefix)])
            if label_vec1.equal(label_vec2):
                pair_indices.append((idx1, idx2))

    sensitive = 0
    total = len(pair_indices)
    for pair in pair_indices:
        idx1, idx2 = pair
        if (acts[idx1][feature_idx] > lamda and acts[idx2][feature_idx] < lamda) or \
            (acts[idx1][feature_idx] < lamda and acts[idx2][feature_idx] > lamda):
            sensitive += 1
    return sensitive / total, total


def plot_distributions(activations, top_features, labels, bins=30, model_name="pythia70m", lamda=0.1):
    random.seed(12)
    for label_name in labels:
        label_vec = t.Tensor(labels[label_name])
        top_feature = top_features[label_name]
        random_feature = random.randint(0, activations.shape[-1])

        class_0_acts = activations.T[top_feature][label_vec == 0]
        class_1_acts = activations.T[top_feature][label_vec == 1]
        var_0, mean_0 = t.var_mean(class_0_acts, dim=-1)
        var_1, mean_1 = t.var_mean(class_1_acts, dim=-1)

        random_class_0_acts = activations.T[random_feature][label_vec == 0]
        random_class_1_acts = activations.T[random_feature][label_vec == 1]
        random_var_0, random_mean_0 = t.var_mean(random_class_0_acts, dim=-1)
        random_var_1, random_mean_1 = t.var_mean(random_class_1_acts, dim=-1)
        
        print(f"{label_name}: {mean_0} ({var_0}) | {mean_1} ({var_1})")
        print(f"\t- Random: {random_mean_0} ({random_var_0}) | {random_mean_1} ({random_var_1})")

        class_0_kde = stats.gaussian_kde(class_0_acts)
        class_1_kde = stats.gaussian_kde(class_1_acts)
        min_act = min(min(class_0_acts), min(class_1_acts))
        max_act = max(max(class_0_acts), max(class_1_acts))
        xx = np.linspace(min_act, max_act, 1000)

        fig, ax1 = plt.subplots(figsize=(10,6))
        
        ax1.hist(class_0_acts, bins=bins, alpha=0.5, color='blue', label='False')
        ax1.hist(class_1_acts, bins=bins, alpha=0.5, color='red', label='True')
        ax1.set_ylabel("Frequency")
        ax2 = ax1.twinx()
        ax2.plot(xx, class_0_kde(xx), color='blue')
        ax2.plot(xx, class_1_kde(xx), color='red')
        ax2.set_ylabel("Density")
        ax2.grid(False)

        plt.title(label_name)
        ax1.legend()

        out_dir = f"results/activation_plots/{model_name}/{lamda}/"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, f"{label_name}.pdf"), format="pdf", bbox_inches="tight")
        plt.cla()
        plt.clf()
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="pythia70m")
    parser.add_argument("--sae", "-s", type=str, default="sae_results/relu_uniform/latest_ckpt.pt")
    parser.add_argument("--id-metric", type=str, choices=["accuracy", "macrof1", "mcc"], default="mcc")
    parser.add_argument("--lamda", type=float, default=0.1)
    parser.add_argument("--randomize-sae", action="store_true")
    parser.add_argument("--identity-baseline", action="store_true")
    args = parser.parse_args()

    device = "cuda"
    dtype = t.float32 if args.model_name == "pythia70m" else t.bfloat16
   
    if args.model_name == "pythia70m":
        model = LanguageModel("EleutherAI/pythia-70m-deduped", device_map=device, dispatch=True, torch_dtype=dtype)
    elif args.model_name == "gemma2":
        model = LanguageModel("google/gemma-2-2b", device_map=device, dispatch=True, attn_implementation="eager", torch_dtype=dtype)
    else:
        raise NotImplementedError()
    
    mid_layer = model.config.num_hidden_layers // 2

    neurons = args.identity_baseline
    
    submodule = Submodule(
        name = f"resid_{mid_layer}",
        submodule=model.gpt_neox.layers[mid_layer],
        is_tuple=True,
    )
    # submodules, dictionaries = load_saes_and_submodules(model, dtype=dtype, device=device, thru_layer=mid_layer, neurons=neurons)
    # submodule = get_mid_layer(submodules, mid_layer)
    # Load for inference only
    dictionary, _, _ = load_sae_inference_only(args.sae, dimin=model.config.hidden_size)
    dictionary.to("cuda:0")
    # Archetypal SAEs
    # dictionaries[submodule] = AutoEncoder.from_pretrained(
    #         "/home/aaron/fromgit/identifiable_language/archetypal/dictionary_learning/weights/pythia-70m/reg/trainer_0/ae.pt",
    #         device=device
    # )
    if args.randomize_sae:
        dictionaries[submodule].encoder.reset_parameters()
        dictionaries[submodule].decoder.reset_parameters()
        # dictionaries[submodule].W_enc.data.normal_(mean=0, std=0.1)
        # dictionaries[submodule].W_dec.data.normal_(mean=0, std=0.1)
        # dictionaries[submodule].b_enc.data.normal_(mean=0, std=0.1)
        # dictionaries[submodule].b_dec.data.normal_(mean=0, std=0.1)
    # dictionary = dictionaries[submodule]


    dataset = load_dataset()
    examples = dataset.examples
    labels = dataset.labels_binary
    num_examples = len(examples)
    batch_size = 1

    n_batches = math.ceil(num_examples / batch_size)
    batches = [
        examples[batch * batch_size : (batch + 1) * batch_size]
        for batch in range(n_batches)
    ]
    labels_batched = [
        {l: labels[l][batch * batch_size : (batch + 1) * batch_size]
        for l in labels}
        for batch in range(n_batches)
    ]

    with t.no_grad(), model.trace("test"):
        x = submodule.get_activation()
        x_saved = x.save()
    print(x_saved.value.shape)
    x_hat, f = dictionary(x_saved.value, return_hidden=True)
    num_hidden = f.detach().shape[-1]
    # f_saved = f.save()
    # num_hidden = f_saved.value.detach().shape[-1]
    acts = t.zeros((num_examples, num_hidden))

    for idx, batch in tqdm(enumerate(batches), desc="Caching activations", total=len(batches)):
        f = get_activations(model, submodule, dictionary, batch).sum(dim=1)
        len_batch = len(batch)
        start_idx = idx * batch_size
        acts[start_idx : start_idx + len_batch] = f

    scores, top_features = score_identification(acts, dataset.labels_binary, 
                                                lamda=args.lamda, metric=args.id_metric)
    if args.id_metric == "mcc":
        top_scores = scores.max(dim=0).values
        # print(top_scores.shape)
        # print(scores, top_features)
        mcc = top_scores.mean().item()
        for i, label in enumerate(list(top_features.keys())):
            if label not in ("domain-science", "sentiment-positive"):
                continue
            print(f"{label}: {top_scores[i]} ({top_features[label]})")
        # for i, label in enumerate(list(top_features.keys())):
        #     print(f"{label}: {top_scores[i]} ({top_features[label]})")
        print(f"MCC: {mcc:.3f}")    
    else:
        print(scores)
        print(sum(list(scores.values())))
        print(sum(list(scores.values())) / len(list(scores.keys())))
        print()
        print(top_features)

    # sensitivities = {}
    # for label in tqdm(dataset.labels_binary, total=len(list(dataset.labels_binary.keys())), desc="Sensitivity of label"):
    #     sensitivities[label], N = score_sensitivity(acts, dataset.labels_binary, top_features[label],
    #                                              lamda=args.lamda, target_label=label)
    # print("Sensitivities: ", sensitivities)
    # print("Sensitivity mean: ", sum(sensitivities.values()) / len(sensitivities.values()))

    # Print conditional distributions
    # plot_distributions(acts, top_features, dataset.labels_binary)
