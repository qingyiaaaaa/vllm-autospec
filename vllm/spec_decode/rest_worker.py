import weakref
from typing import List, Optional, Set, Tuple

import torch

from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest
from vllm.spec_decode.interfaces import SpeculativeProposals
from vllm.spec_decode.proposer_worker_base import NonLLMProposerWorkerBase
from vllm.spec_decode.top1_proposer import Top1Proposer

import torch
import torch.nn.functional as F
import draftretriever

def reorder_token_list(tokens_list: List[List]) -> List[List]:
    """
    Reorder the given list of tokens such that tokens are not the prefixes 
    of other tokens are placed at the begin.
    """
    prefixes = set()
    for tokens in tokens_list:
        for i in range(1, len(tokens)):
            prefix = tuple(tokens[:i])
            prefixes.add(prefix)

    non_prefix = []
    has_prefix = []
    for tokens in tokens_list:
        if tuple(tokens) in prefixes:
            has_prefix.append(tokens)
        else:
            non_prefix.append(tokens)
    for prefix in prefixes:
        if list(prefix) not in has_prefix:
            has_prefix.append(list(prefix))

    return non_prefix + has_prefix

def choose_unprefixed_token_list(tokens_list: List[List]) -> List[List]:
    """
    Reorder the given list of tokens such that tokens are not the prefixes 
    of other tokens are placed at the begin.
    """
    prefixes = set()
    for tokens in tokens_list:
        for i in range(1, len(tokens)):
            prefix = tuple(tokens[:i])
            prefixes.add(prefix)

    non_prefix = []
    has_prefix = []
    for tokens in tokens_list:
        if tuple(tokens) in prefixes:
            has_prefix.append(tokens)
        else:
            non_prefix.append(tokens)
    for prefix in prefixes:
        if list(prefix) not in has_prefix:
            has_prefix.append(list(prefix))

    return non_prefix


def pad_path(path, length, pad_value=-2):
    """
    Pad the given path list with a specific value up to a specified length.
    
    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-2): The value to use for padding.
    
    Returns:
    - list: A new list based on the original path but padded to the desired length.
    
    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]
    
    Note:
    If the given path is already longer than the specified length, 
    then no padding occurs, and the original path is returned.
    """
    
    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))

def top_p_filtering(logits, top_p=0.0, filter_value=float('-inf')):
    # from https://github.com/huggingface/transformers/blob/18a879f47576822aa1a5c49aecb27d89bfa5fa69/examples/run_generation.py#L79


    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
    logits[indices_to_remove] = filter_value
    return logits


class RestWorker(NonLLMProposerWorkerBase):
    """NGramWorker provides a light drafter without need for model.

    Current NGramWorker only implements prompt lookup decoding,
    and in future we may also do RAG type drafter and other scenarios
    which don't rely on LLM model to give proposals.
    """

    def __init__(self, *args, **kwargs):
        # Get local_rank/vocab_size from kwargs attribute
        self.local_rank = kwargs["local_rank"]
        self.vocab_size = kwargs["vllm_config"].model_config.get_vocab_size()
        self.device_type = kwargs.get("device_type", "cuda")

        # Lazy initialization list.
        self._proposer: Top1Proposer

        
    def set_datastore_tokenspans(self, datastore_path: str, max_token_span: int):
        self.datastore = draftretriever.Reader(
            index_file_path=datastore_path
        )
        self.token_spans = list(range(2, max_token_span + 1))[::-1]
        

    def init_device(self):
        self.device = torch.device(f"{self.device_type}:{self.local_rank}")
        self.load_model = lambda *args, **kwargs: None

        # Current NGramWorker only supports Top1Proposer
        self._proposer = Top1Proposer(
            weakref.proxy(self),  # type: ignore[arg-type]
            device=self.device,
            vocab_size=self.vocab_size,
        )

    def generate_proposals_and_prefix_idx(self, input_ids : List[int], max_num_draft=64):

        retrieved_token_list = []
                
        for token_span in self.token_spans:
            this_token = input_ids[-token_span:]
            # Retrieve draft tokens from the datastore, and get draft buffer
            retrieved_token_list, _, _ , _ , _ = self.datastore.search(this_token, choices=max_num_draft)
        
            # No retrieved sequences
            if len(retrieved_token_list) == 0:
                continue
            # Break because this span has hitted
            else:
                break

        if len(retrieved_token_list) == 0:
            return [], {}
        
        # generate a prefix dict
        
        prefix_dict = {(-1, ) : 0}
        
        for idx, retrieved_token in enumerate(retrieved_token_list):
            while retrieved_token and retrieved_token[-1] == -2:
                retrieved_token.pop()
        
        retrieved_token_list = reorder_token_list(retrieved_token_list)
        
        for idx, retrieved_token in enumerate(retrieved_token_list):
            prefix_dict[tuple(retrieved_token)] = idx + 1
        
        return retrieved_token_list, prefix_dict

    def generate_unprefixed_proposals(self, input_ids : List[int], max_num_draft = 64):
        retrieved_token_list = []
                
        for token_span in self.token_spans:
            this_token = input_ids[-token_span:]
            # Retrieve draft tokens from the datastore, and get draft buffer
            retrieved_token_list, _, _ , _ , _ = self.datastore.search(this_token, choices=max_num_draft)
        
            # No retrieved sequences
            if len(retrieved_token_list) == 0:
                continue
            # Break because this span has hitted
            else:
                break

        if len(retrieved_token_list) == 0:
            return []
        
        for retrieved_token in retrieved_token_list:
            while retrieved_token and retrieved_token[-1] == -2:
                retrieved_token.pop()
        
        retrieved_token_list = choose_unprefixed_token_list(retrieved_token_list) 
        
        return retrieved_token_list

    def sampler_output(
        self,
        execute_model_req: ExecuteModelRequest,
        sample_len: int,
        # Unused parameter. NGramWorker does not use the KV Cache and
        # therefore does not need this parameter.
        seq_ids_with_bonus_token_in_last_step: Set[int],
    ) -> Tuple[Optional[List[Optional[SamplerOutput]]], bool]:
        """NGram match algo to pick proposal candidate. Returns the list of
        sampler output, one per SequenceGroupMetadata.

        For ngram worker, we already done needed transposed internal, so the
        indicator pass to sampler_output_to_torch shall be False.
        """
        self._raise_if_unsupported(execute_model_req)

        has_spec_out = False
        token_id_list: List[Optional[torch.Tensor]] = []
        token_prob_list: List[Optional[torch.Tensor]] = []
        for idx, seq_group_metadata in enumerate(
                execute_model_req.seq_group_metadata_list):
            seq_data = next(iter(seq_group_metadata.seq_data.values()))

            seq_len = seq_data.get_len()
            # When seq_len is less than 3072 (3K), we use CPU to perform
            # the ngram match. Otherwise, we use the device specified in
            # the model config (normally GPU). 3072 is a rough threshold
            # based on profiling on H100, and it can be adjusted based
            # on the actual performance on different hardware.
            cur_device = "cpu" if seq_len < 3072 else self.device
            input_ids = torch.as_tensor(seq_data.get_token_ids(),
                                        dtype=torch.long,
                                        device=cur_device)
            input_length = seq_data.get_len()

            for ngram_size in range(
                    min(self.ngram_prompt_lookup_max, input_length - 1),
                    self.ngram_prompt_lookup_min - 1,
                    -1,
            ):
                ngram_tensor = input_ids[-ngram_size:]
                if ngram_size == 1:
                    # Do not match itself and do not use unfold and all
                    matches = (input_ids[:-1] == ngram_tensor)
                else:
                    windows = input_ids.unfold(dimension=0,
                                               size=ngram_size,
                                               step=1)
                    # Do not match itself
                    matches = (windows[:-1] == ngram_tensor).all(dim=-1)

                # first_match includes "values" (bool), indicating whether
                # the match is found, and "indices", indicating the index
                # of the first match.
                first_match = matches.max(dim=-1)
                if first_match.values.item():
                    proposal_start_idx = first_match.indices.add_(ngram_size)
                    spec_indices = (
                        proposal_start_idx).repeat(sample_len) + torch.arange(
                            sample_len, device=cur_device)
                    spec_indices.clamp_(max=input_ids.shape[-1] - 1)
                    res = input_ids.gather(dim=-1,
                                           index=spec_indices).to(self.device)
                    token_id_list.append(res)
                    token_prob_list.append(
                        torch.nn.functional.one_hot(
                            res,
                            num_classes=self.vocab_size).to(torch.float32))
                    has_spec_out = True
                    break
            else:
                token_id_list.append(None)
                token_prob_list.append(None)

        if not has_spec_out:
            return None, False

        outputs: List[Optional[SamplerOutput]] = []
        for idx in range(len(execute_model_req.seq_group_metadata_list)):
            if token_id_list[idx] is None:
                outputs.append(None)
            else:
                outputs.append(
                    SamplerOutput(
                        outputs=None,
                        sampled_token_probs=token_prob_list[idx],
                        logprobs=torch.zeros((sample_len, self.vocab_size),
                                             dtype=torch.float32,
                                             device=self.device),
                        sampled_token_ids=token_id_list[idx],
                    ))

        return outputs, False

    def get_spec_proposals(
        self,
        execute_model_req: ExecuteModelRequest,
        # Unused parameter. NGramWorker does not use the KV Cache and
        # therefore does not need this parameter.
        seq_ids_with_bonus_token_in_last_step: Set[int],
    ) -> SpeculativeProposals:
        """Produce speculations given an input batch of sequences. The number of
        speculative tokens per sequence is determined by max_proposal_len.
        """
        group_retrieved_token_list = {}
        group_prefix_dict = {}
        for seq_group_metadata in execute_model_req.seq_group_metadata_list:
            seq_id = seq_group_metadata.get_first_seq_id()
            assert len(seq_group_metadata.seq_data) == 1
            seq_data = next(iter(seq_group_metadata.seq_data.values()))
            input_ids = seq_data.get_token_ids()
            #retrieved_token_list, prefix_dict = self.generate_proposals_and_prefix_idx(input_ids)
            #group_retrieved_token_list[req_id] = retrieved_token_list
            #group_prefix_dict[req_id] = prefix_dict
            retrieved_token_list = self.generate_unprefixed_proposals(input_ids)
            group_retrieved_token_list[seq_id] = retrieved_token_list
        
        return SpeculativeProposals(
            proposal_token_ids=torch.tensor([1]),
            proposal_probs=torch.tensor([1]),
            proposal_lens=torch.tensor([1]),
            no_proposals=False,
            is_tree_decoding=True,
            tree_proposals=group_retrieved_token_list,
            prefix_dict=group_prefix_dict
        )
        
    def _raise_if_unsupported(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> None:
        """NGramWorker does not yet implement support for cache swap
        operations or beam search.
        """
        if any([
                execute_model_req.blocks_to_swap_in,
                execute_model_req.blocks_to_swap_out,
                execute_model_req.blocks_to_copy
        ]):
            raise NotImplementedError(
                "NGramWorker does not support cache operations")

        if any(
                len(seq_group_metadata.seq_data.keys()) != 1
                for seq_group_metadata in
                execute_model_req.seq_group_metadata_list):
            raise NotImplementedError(
                "NGramWorker does not support beam search.")
