import torch
from typing import Tuple
from collections import defaultdict
from torch.utils.data import DataLoader
import numpy as np
import json
import os
from .PCFG import PCFG
from transformers import AutoTokenizer
import pickle as pkl

def get_dataloader_json(
        path: str=None,
        corr_config: str=None,
        num_iters: int=1e6,
        max_sample_length: int=64,
        seed: int=42,
        batch_size: int=32,
        num_workers: int=4,
        model_name: str=None
):
    # Create a dataset
    dataset = JSONDataset(
            path=path,
            corr_config=corr_config,
            num_iters=num_iters,
            max_sample_length=max_sample_length,
            seed=seed,
            model_name=model_name
        ) 

    # Create a dataloader
    dataloader = DataLoader(
                dataset,
                sampler=torch.utils.data.RandomSampler(dataset, replacement=True), 
                shuffle=False,
                pin_memory=True,
                batch_size=batch_size,
                num_workers=num_workers,
            )

    return dataloader


def get_dataloader(
        language: str = 'english', # in ['english', 'expr', 'dyck']
        config: dict = {'n_nouns': 10,
                        'n_verbs': 10,
                        'n_adjectives': 10,
                        'n_pronouns': 10,
                        'n_adverbs': 10,
                        'n_conjunctions': 2,
                        'p_conjunctions': 0.15,
                        'n_prepositions': 0,
                        'relative_clauses': False,
                        'transitive_verbs': False
                        }, # config for PCFG. see below for other languages.
        alpha: float = 1e5,
        prior_type: str = 'dirichlet',
        num_iters: int=1e6,
        max_sample_length: int=128,
        seed: int = 42,
        batch_size: int = 32, 
        num_workers: int = 4,
        ):
    """Define the PCFG dataloader.

    Args:
        language: The language of the PCFG. One of ['english', 'expr', 'dyck1', 'dyck2'].
        config: The configuration of the PCFG. The keys depend on the language.
        * For 'english':
            n_nouns: The number of nouns in the vocabulary.
            n_verbs: The number of verbs in the vocabulary.
            n_adjectives: The number of adjectives in the vocabulary.
            n_pronouns: The number of pronouns in the vocabulary.
            n_adverbs: The number of adverbs in the vocabulary.
            n_conjunctions: The number of conjunctions in the vocabulary.
            p_conjunctions: The probability of generating a conjunction.
            n_prepositions: The number of prepositions in the vocabulary.
            relative_clauses: Whether to generate relative clauses (as both adjectives and adverbs).
            transitivity: Whether to distinguish transitive and intransitive verbs.
        * For 'expr':
            n_digits: The number of digits in the vocabulary.
            n_ops: The number of operations in the vocabulary.
            bracket: Whether to include brackets in the vocabulary.
        * For 'dyck':
            n_brackets: The number of types brackets in the vocabulary.
            p_nest: The probability of nesting sequences. Should be ≤ 0.5.
        alpha (float, optional): The concentration parameter for the Dirichlet distribution. Defaults to 1e5.
        prior_type (str, optional): The type of prior distribution. Defaults to 'dirichlet'.
        num_iters (int, optional): The number of iterations to make in the training loop per epoch. Defaults to 1e6.
        max_sample_length (int, optional): The maximum length of a sequence. Defaults to 128.
        seed (int, optional): The random seed. Defaults to 42.
        batch_size (int, optional): The batch size. Defaults to 32.
        num_workers (int, optional): The number of workers. Defaults to 4.

    Returns:
        DataLoader: A pytorch compatible, PCFG dataloader.    
    """

    # Create a dataset
    dataset = PCFGDataset(
            language=language,
            config=config,
            alpha=alpha,
            prior_type=prior_type,
            num_iters=num_iters,
            max_sample_length=max_sample_length,
            seed=seed,
        ) 

    # Create a dataloader
    dataloader = DataLoader(
                dataset,
                sampler=torch.utils.data.RandomSampler(dataset, replacement=True), 
                shuffle=False,
                pin_memory=True,
                batch_size=batch_size,
                num_workers=num_workers,
            )

    return dataloader


class JSONDataset():
    def __init__(self,
        path: str = None,
        seed: int = 42,
        max_sample_length: int = 64,
        num_iters: int = 1e6,
        corr_config: str = None,
        model_name: str = None
    ):
                # Some setup details
        self.num_iters = int(num_iters)
        self.max_sample_length = max_sample_length
        self.rng = np.random.RandomState(seed)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.seed = seed

        # Load correlation config
        if corr_config:
            with open(corr_config) as f:
                self.corr_config = json.load(f)
        else:
            self.corr_config = {}

        # Load examples from JSON.
        self.examples = []
        with open(path, 'r') as json_file:
            for line in json_file:
                self.examples.append(json.loads(line))

        # Build indices for efficient sampling
        self._build_correlation_indices()
        print("Pregenerating sampling order...")
        self._pregenerate_sampling_order()
        print("Done!")

    def _build_correlation_indices(self):
        """Builds three pools of indices based on the correlation config."""
        if not self.corr_config:
            return

        # Group examples by attribute-value pairs
        self.attr_value_indices = defaultdict(list)
        for idx, example in enumerate(self.examples):
            for attr, value in example.items():
                if attr != 'sentence':
                    key = f"{attr}-{value}"
                    self.attr_value_indices[key].append(idx)

        # Use the first correlation rule in the config file.
        attr1_val, target = next(iter(self.corr_config.items()))
        attr2_val, self.correlation_prob = target # This is now the target joint probability

        # Pool 1: Indices for examples with BOTH attr1_val AND attr2_val.
        attr1_examples = set(self.attr_value_indices.get(attr1_val, []))
        attr2_examples = set(self.attr_value_indices.get(attr2_val, []))
        self.both_indices = list(attr1_examples & attr2_examples)

        # Pool 2: Indices for examples with attr1_val but NOT attr2_val.
        self.attr1_only_indices = list(attr1_examples - attr2_examples)

        # Pool 3: Indices for all other examples (those without attr1_val).
        all_indices = set(range(len(self.examples)))
        other_examples = all_indices - attr1_examples
        self.other_indices = list(other_examples)

    def _pregenerate_sampling_order(self):
        """
        Pregenerates the sampling order based on a target joint probability.
        """
        if not self.corr_config or not self.both_indices:
            # Fallback to uniform sampling if no correlation is set or if it's impossible to satisfy.
            self.sampling_indices = self.rng.choice(len(self.examples), size=self.num_iters, replace=True)
            if self.corr_config and not self.both_indices:
                 print("Warning: No examples found with both specified attributes. Falling back to uniform sampling.")
            return

        # --- Define sampling probabilities for the three pools ---

        # 1. The probability of sampling from the "both_indices" pool is the desired correlation.
        p_both = self.correlation_prob
        
        # 2. The remaining probability is distributed between the other two pools
        #    based on their relative sizes in the original dataset.
        p_remaining = 1.0 - p_both
        
        size_attr1_only = len(self.attr1_only_indices)
        size_other = len(self.other_indices)
        total_remaining_size = size_attr1_only + size_other

        if total_remaining_size > 0:
            p_attr1_only = p_remaining * (size_attr1_only / total_remaining_size)
            p_other = p_remaining * (size_other / total_remaining_size)
        else: # Handle case where only "both_indices" has samples
            p_attr1_only, p_other = 0.0, 0.0
            p_both = 1.0 # Force sampling from the only available pool

        # --- Determine number of samples from each pool ---
        n_both = int(round(p_both * self.num_iters))
        n_attr1_only = int(round(p_attr1_only * self.num_iters))
        # The rest go to the "other" pool to ensure total is num_iters
        n_other = self.num_iters - n_both - n_attr1_only
        
        # --- Generate samples from each pool ---
        samples_both = self.rng.choice(self.both_indices, size=n_both, replace=True)

        if size_attr1_only > 0:
            samples_attr1_only = self.rng.choice(self.attr1_only_indices, size=n_attr1_only, replace=True)
        else:
            samples_attr1_only = np.array([], dtype=int)
            
        if size_other > 0:
            samples_other = self.rng.choice(self.other_indices, size=n_other, replace=True)
        else:
            samples_other = np.array([], dtype=int)

        # --- Combine and shuffle ---
        self.sampling_indices = np.concatenate((samples_both, samples_attr1_only, samples_other))
        self.rng.shuffle(self.sampling_indices)
        
        # Ensure the final array has the correct size after rounding
        if len(self.sampling_indices) != self.num_iters:
            self.sampling_indices = np.resize(self.sampling_indices, self.num_iters)

    def _sample_with_correlation(self):
        """Sample an example according to correlation constraints."""
        if not self.correlation_pairs:
            # No correlation constraints, sample uniformly
            return self.rng.choice(self.examples)
        
        # For simplicity, focus on the first correlation pair
        # In a more complex scenario, you might want to handle multiple pairs
        pair_key = f"{self.correlation_pairs[0]['attr1_val']}->{self.correlation_pairs[0]['attr2_val']}"
        corr_data = self.correlation_data[pair_key]
        
        correlation = corr_data['correlation']
        both_indices = corr_data['both_indices']
        attr1_only_indices = corr_data['attr1_only_indices']
        
        # If we have examples that satisfy the correlation constraint
        if both_indices and attr1_only_indices:
            # Sample according to correlation: p(attr2_val | attr1_val) ≈ correlation
            if self.rng.random() < correlation:
                # Sample from examples with both attributes
                idx = self.rng.choice(both_indices)
            else:
                # Sample from examples with attr1 but not attr2
                idx = self.rng.choice(attr1_only_indices)
            return self.examples[idx]
        
        elif both_indices:
            # Only examples with both attributes exist
            if correlation > 0:
                idx = self.rng.choice(both_indices)
                return self.examples[idx]
            else:
                # Correlation is 0 but we only have positive examples
                # Fall back to uniform sampling
                return self.rng.choice(self.examples)
        
        elif attr1_only_indices:
            # Only examples with attr1 but not attr2 exist
            if correlation < 1:
                idx = self.rng.choice(attr1_only_indices)
                return self.examples[idx]
            else:
                # Correlation is 1 but we only have negative examples
                # Fall back to uniform sampling
                return self.rng.choice(self.examples)
        
        else:
            # No examples with attr1_val, fall back to uniform sampling
            return self.rng.choice(self.examples)
        
    def __len__(self):
        """
        Return the number of iterations made in the training loop per epoch.
        """
        return self.num_iters
    

    def __getitem__(self, index):
        """
        Get the next sequence from the example list, sampled according to correlation constraints.
        """
        
        while True:
            # Use pre-computed sampling index
            example_idx = self.sampling_indices[index % len(self.sampling_indices)]
            example = self.examples[example_idx]
            
            # Extract the sentence
            sentence = example['sentence']
            labels = {k: v for k, v in example.items() if k != "sentence"}
            sequence = torch.tensor(self.tokenizer(sentence, return_tensors="pt").input_ids)[0]
            seq_length = float(sequence.size(0))

            # Truncate the sequence if it is longer than the max sequence length
            if sequence.size(0) > self.max_sample_length - 3:
                continue
            # Pad the sequence to the max sequence length with <pad>
            else:
                pad_token_id = 0  # You'll need to define this properly
                sequence = torch.cat((
                    sequence, 
                    torch.tensor([pad_token_id] * (self.max_sample_length - len(sequence)))
                ))
                break

        return sequence, seq_length, labels


class PCFGDataset():
    def __init__(self,
        language: str = 'english', # in ['english', 'expr', 'dyck']
        config: dict = {'n_nouns': 10,
                        'n_verbs': 10,
                        'n_adjectives': 10,
                        'n_pronouns': 10,
                        'n_adverbs': 10,
                        'n_conjunctions': 2,
                        'p_conjunctions': 0.15,
                        'n_prepositions': 0,
                        'relative_clauses': False,
                        'transitive_verbs': False}, # config for PCFG. see below for other languages.
        alpha: float = 1e5,
        prior_type: str = 'dirichlet',
        num_iters: int=1e6,
        max_sample_length: int=128,
        seed: int = 42,
        ):
        """Define the PCFG dataset.

        Args:
            language: The language of the PCFG. One of ['english', 'expr', 'dyck1', 'dyck2'].
            config: The configuration of the PCFG. The keys depend on the language.
            * For 'english':
                n_nouns: The number of nouns in the vocabulary.
                n_verbs: The number of verbs in the vocabulary.
                n_adjectives: The number of adjectives in the vocabulary.
                n_pronouns: The number of pronouns in the vocabulary.
                n_adverbs: The number of adverbs in the vocabulary.
                n_conjunctions: The number of conjunctions in the vocabulary.
                p_conjunctions: The probability of generating a conjunction.
                n_prepositions: The number of prepositions in the vocabulary.
                relative_clauses: Whether to generate relative clauses (as both adjectives and adverbs).
                transitivity: Whether to distinguish transitive and intransitive verbs.
            * For 'expr':
                n_digits: The number of digits in the vocabulary.
                n_ops: The number of operations in the vocabulary.
                bracket: Whether to include brackets in the vocabulary.
            * For 'dyck':
                n_brackets: The number of types brackets in the vocabulary.
                p_nest: The probability of nesting sequences. Should be ≤ 0.5.
            alpha (float, optional): The concentration parameter for the Dirichlet distribution. Defaults to 1e5.
            prior_type (str, optional): The type of prior distribution. Defaults to 'dirichlet'.
            num_iters (int, optional): The number of iterations to make in the training loop per epoch. Defaults to 1e6.
            max_sample_length (int, optional): The maximum length of a sequence. Defaults to 128.
            seed (int, optional): The random seed. Defaults to 42.

        Returns:
            PCFGDataset: A PCFG dataset.
        """

        # Some setup details
        self.num_iters = int(num_iters)
        self.max_sample_length = max_sample_length

        # Instructions / tasks (NOTE: this is where the code will be changed to add tasks)
        self.tasks_dict = {}
        for n_task, task in enumerate(['freegen']):
            self.tasks_dict[f'Task{n_task}'] = task
        self.task_tokens = list(self.tasks_dict.keys())
        self.prior_over_tasks = [1.0]

        # Define the PCFG
        self.PCFG = PCFG(
            language=language,
            config=config,
            alpha=alpha,
            prior_type=prior_type,
            seed=seed,
            tasks=self.tasks_dict,
        )

        # Instruction decorator
        self.instruction_decorator = 'Task: {task_token} \n Ops: {ops} \n Out:' 
        self.decorator_length = len(self.PCFG.tokenize_sentence(self.instruction_decorator.format(task_token='Task0', ops='<null>')))

        ## Special tokens
        # Pad token
        self.pad_token = '<pad>'
        self.pad_token_id = self.PCFG.vocab['<pad>']

        # Tokenize the task tokens
        self.task_token_idx = {t: self.PCFG.tokenize_sentence(t)[0] for t in self.task_tokens}

        # Define the PCFG generator
        self.generator = self.PCFG.sentence_generator(num_of_samples=self.num_iters)

        # Generation input template
        self.template = torch.tensor(self.PCFG.tokenize_sentence('Task: Task0 \n Ops: <null> \n Out:'))


    def save_grammar(self, path_to_results: str):
        """
        Save the grammar underlying the dataset
        """
        base_dir = os.path.join(path_to_results, 'grammar')
        os.makedirs(base_dir, exist_ok=True)
        with open(os.path.join(base_dir, 'PCFG.pkl'), 'wb') as f:
            pkl.dump(self.PCFG, f)


    def load_grammar(self, path_to_results: str):
        """
        Load and override grammar of the dataset
        """
        base_dir = os.path.join(path_to_results, 'grammar')
        with open(os.path.join(base_dir, 'PCFG.pkl'), 'rb') as f:
            self.PCFG = pkl.load(f)


    def __len__(self):
        """
        Return the number of iterations made in the training loop per epoch.
        """
        return self.num_iters


    def __getitem__(self, index):
        """
        Get the next sequence from the PCFG generator.
        """
        
        while True:

            # Generate a sequence from the PCFG
            sequence = self.generator.__next__()

            # Tokenize the sequence
            sequence = torch.tensor(self.PCFG.tokenize_sentence(sequence))
            seq_length = float(sequence.size(0))

            # Define instruction (NOTE: kept null for now, but the code can be changed here to implement tasks)
            task_token = np.random.choice(self.task_tokens, p=self.prior_over_tasks)
            if task_token == 'Task0':
                ops = '<null>'
            else:
                raise ValueError(f"Invalid task token: {task_token}")
            instr = torch.tensor(self.PCFG.tokenize_sentence(
                self.instruction_decorator.format(task_token=task_token, ops=ops)
                ))

            # Concatenate the instruction to the sequence
            sequence = torch.cat((
                instr,
                sequence,
                torch.tensor([self.PCFG.vocab['<eos>']])
                ))

            # Truncate the sequence if it is longer than the max sequence length
            if sequence.size(0) > self.max_sample_length - 10:
                pass

            # Pad the sequence to the max sequence length with <pad>
            else:
                sequence = torch.cat((
                    sequence, 
                    torch.tensor([self.pad_token_id] * (self.max_sample_length - len(sequence)))
                    ))
                break

        return sequence, seq_length