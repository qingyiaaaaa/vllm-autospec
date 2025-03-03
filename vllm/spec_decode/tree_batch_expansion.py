from array import array
from itertools import chain, count
from typing import Iterator, List, Optional, Tuple
from vllm.utils import Device
import torch
from vllm.core.block_manager import SelfAttnBlockSpaceManager
from vllm.core.block.block_table import BlockList, BlockTable
from vllm import SamplingParams
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import (VLLM_INVALID_TOKEN_ID, VLLM_TOKEN_ID_ARRAY_TYPE,
                           ExecuteModelRequest, SequenceData,
                           SequenceGroupMetadata, get_all_seq_ids)
from vllm.spec_decode.interfaces import (SpeculativeProposals,
                                         SpeculativeScorer, SpeculativeScores)
from vllm.spec_decode.util import nvtx_range, split_batch_by_proposal_len, Timer

SeqId = int
TargetSeqId = int
TokenId = int

DEFAULT_SIMPLE_SAMPLING_PARAMS = SamplingParams()

class BatchExpansionTreeScorer(SpeculativeScorer):
    
    def __init__(self, scorer_worker, device, vocab_size):
        super().__init__(scorer_worker, device, vocab_size)
        self.block_manager = None
        self.max_expand_size = None
        self.min_execute_time = None
    
    def set_block_manager(self, block_manager : SelfAttnBlockSpaceManager):
        self.block_manager = block_manager
        
    def set_max_expand_size(self, max_expand_size : int):
        self.max_expand_size = max_expand_size
    
    
    def select_proposal(self, tree_proposals, max_expand_size):
        seq_ids = list(tree_proposals.keys())
        num_seqs = len(seq_ids)
        if num_seqs == 0 or max_expand_size <= 0:
            # No sequences to allocate or budget is zero, return empty results
            return {seq_id: [] for seq_id in seq_ids}
        
        # Step 1: Distribute batch_size evenly, calculate the initial target allocation for each sequence
        base_target = max_expand_size // num_seqs  # Basic batch_size allocated to each seq_id
        remainder = max_expand_size % num_seqs     # Remaining batch_size after even distribution
        
        selected_proposals = {seq_id: [] for seq_id in seq_ids}  # Output results, proposals selected for each seq_id
        leftover_capacity = remainder  # Remaining unused batch_size initialized with the leftover from division
        leftover_proposals = {seq_id: [] for seq_id in seq_ids}  # Store unused proposals after initial selection
        
        # Helper function: Select the best combination from a list of lengths such that the total is <= capacity (using dynamic programming)
        def select_best_combination(lengths, capacity):
            """
            Using subset-sum dynamic programming, select several items from the length array such that the total sum does not exceed capacity
            and is as large as possible.
            Returns the selected indices and the total length.
            """
            n = len(lengths)
            if n == 0 or capacity <= 0:
                return [], 0
            achievable = [False] * (capacity + 1)
            achievable[0] = True
            parent_index = [-1] * (capacity + 1)  # Record the index of items used to reach sum s
            parent_sum = [-1] * (capacity + 1)    # Record the previous sum before using the item
            
            for idx, length in enumerate(lengths):
                if length > capacity:
                    # Skip proposals that exceed the capacity
                    continue
                # Traverse in reverse order to ensure each proposal is only used once
                for s in range(capacity, length - 1, -1):
                    if achievable[s - length] and not achievable[s]:
                        achievable[s] = True
                        parent_index[s] = idx
                        parent_sum[s] = s - length
            # Find the maximum achievable sum that does not exceed the capacity
            best_sum = 0
            for s in range(capacity, -1, -1):
                if achievable[s]:
                    best_sum = s
                    break
            # Backtrack to get the indices of the selected proposals
            selected_idx = []
            curr_sum = best_sum
            while curr_sum > 0 and parent_index[curr_sum] is not None and parent_index[curr_sum] != -1:
                idx = parent_index[curr_sum]
                selected_idx.append(idx)
                curr_sum = parent_sum[curr_sum]
            selected_idx.reverse()
            return selected_idx, best_sum
        
        # Step 2: For each seq_id, select the optimal combination of proposals such that the batch_size consumption is close to the base_target
        for seq_id in seq_ids:
            proposals = tree_proposals[seq_id]
            lengths = [len(p) + 1 for p in proposals]
            if base_target > 0 and lengths:
                # Use dynamic programming to select the best combination with total length <= base_target
                selected_idx, used_len = select_best_combination(lengths, base_target)
                # Save the selected proposals
                for idx in selected_idx:
                    selected_proposals[seq_id].append(proposals[idx])
                # Reclaim unused capacity for this seq_id
                if used_len < base_target:
                    leftover_capacity += (base_target - used_len)
                # Save unselected proposals to be allocated later
                if selected_idx:
                    selected_set = set(selected_idx)
                    for idx, prop in enumerate(proposals):
                        if idx not in selected_set:
                            leftover_proposals[seq_id].append(prop)
                else:
                    leftover_proposals[seq_id] = proposals.copy()
            else:
                # If base_target is 0 or no proposals, leave all to be allocated later
                leftover_proposals[seq_id] = proposals.copy()
                if base_target > 0:
                    leftover_capacity += base_target
        
        # Step 3: Reclaim the remaining batch_size and allocate it to seq_ids that can still expand (gather all remaining proposals and select)
        if leftover_capacity > 0:
            all_leftover_items = []
            for seq_id, props in leftover_proposals.items():
                for prop in props:
                    prop_length = len(prop)
                    if prop_length <= leftover_capacity:
                        all_leftover_items.append((prop_length + 1, seq_id, prop))
            # If there are proposals that can utilize the remaining capacity, select the optimal combination
            if all_leftover_items:
                lengths = [item[0] for item in all_leftover_items]
                selected_idx, used_len = select_best_combination(lengths, leftover_capacity)
                # Add the selected proposals to the corresponding seq_id's result list
                for idx in selected_idx:
                    _, seq_id, prop = all_leftover_items[idx]
                    selected_proposals[seq_id].append(prop)
        return selected_proposals
    
    def prepare_tree_decoding_input(self, proposals : SpeculativeProposals):
        #this function will generate the proposal prefix dict 
        # and control the expand size according the self.max_expand_size
        if self.max_expand_size is not None:
            
            tree_proposals = proposals.tree_proposals
            #Reserve some space for non spec batch
            max_expand_size = self.max_expand_size - len(tree_proposals)
                
            proposals.tree_proposals = self.select_proposal(tree_proposals, max_expand_size)
        
        for seq_id, seq_proposals in proposals.tree_proposals.items():
            #seq_proposals contains a list of proposal List[List[int]]
            #we need to convert it to proposals.prefix_dict
            seq_expand_list = []
            for proposal in seq_proposals:
                seq_expand_list.append(tuple(proposal))
            
            for proposal in seq_proposals:
                for i in range(1, len(proposal)):
                    if tuple(proposal[:i]) not in seq_expand_list:
                        seq_expand_list.append(tuple(proposal[:i]))
            
            prefix_dict = { -1 : 0}
            for i, proposal in enumerate(seq_expand_list):
                prefix_dict[proposal] = i + 1
            proposals.tree_proposals[seq_id] = seq_expand_list
            proposals.prefix_dict[seq_id] = prefix_dict
        
    @staticmethod
    def _create_target_seq_id_iterator(
            seq_ids: List[SeqId]) -> Iterator[TargetSeqId]:
        """Create an iterator for creating target sequence ids.
        Target sequence ids are distinct from sequence ids because we create a
        distinct target sequence id for each proposal token to be scored.

        This implementation increments a counter starting at 1 + max of all
        provided input sequence ids.
        """
        return count(start=max(seq_ids) + 1)

    @staticmethod
    def _create_single_tree_target_seq_group_metadata(
        seq_group_metadata: SequenceGroupMetadata,
        seq_id: SeqId,
        target_seq_id: TargetSeqId,
        token_ids: List[TokenId],
        sampling_params: SamplingParams,
        block_table: List[int],
    ) -> SequenceGroupMetadata:
        """Create a single target SequenceGroupMetadata.

        Args:
            seq_group_metadata: The metadata for the input sequence.
            seq_id: The input sequence ID.
            target_seq_id: The corresponding target sequence ID.
            token_ids: The list of token ids that are to be appended to the
                input sequence.
        """
        seq_data = seq_group_metadata.seq_data[seq_id]
        prompt_token_ids = seq_data.prompt_token_ids_array
        new_output_token_ids = [*seq_data.get_output_token_ids(), *token_ids]
        mrope_position_delta = seq_data.mrope_position_delta

        new_seq_data_dict = {
            target_seq_id:
            SequenceData(
                prompt_token_ids,
                _output_token_ids=array(VLLM_TOKEN_ID_ARRAY_TYPE,
                                        new_output_token_ids),
            ),
        }
        # expect the last block, the other physical blocks can be shared by different 
        # speculative sequences
        for data in new_seq_data_dict.values():
            data.update_num_computed_tokens(data.get_len() - 1)
            data.mrope_position_delta = mrope_position_delta

        return SequenceGroupMetadata(
            request_id=seq_group_metadata.request_id,
            is_prompt=seq_group_metadata.is_prompt,
            seq_data=new_seq_data_dict,
            sampling_params=sampling_params,
            block_tables={
                target_seq_id: block_table,
            },
            lora_request=None,
            token_chunk_size=1,
        )


    @nvtx_range("BatchExpansionTop1Scorer.score_proposals")
    def score_proposals(
        self,
        execute_model_req: ExecuteModelRequest,
        proposals: SpeculativeProposals
    ) -> SpeculativeScores:
        """Score the proposed tokens via the scorer model.

        This converts each input sequence to a set of k+1 target sequences. The
        target sequences have the unique continuations to be scored and a
        unique sequence ID that is different from all input sequence ids.

        If a speculative sequence length would exceed the max model length, then
        no speculation is produced for that sequence.

        Args:
            execute_model_req: The execution request.
            proposals: The speculative proposals to score.
        Returns:
            SpeculativeScores: The scores of each speculative token, along with
                which sequences were ignored during scoring.
        """
        assert self.block_manager is not None, "block_manager must be provided for tree decoding"
        
        self.prepare_tree_decoding_input(proposals)
        
        batch_expand_size = [ len(proposal) + 1 for proposal in proposals.tree_proposals.values()]
        batch_start_idx = [sum(batch_expand_size[:i]) for i in range(len(batch_expand_size))]

        
        expand_seq_group_metadata_list = [0 for _ in range(sum(batch_expand_size))]
        all_seq_ids = get_all_seq_ids(execute_model_req.seq_group_metadata_list)
        target_seq_id_iterator =  self._create_target_seq_id_iterator(seq_ids=get_all_seq_ids(execute_model_req.seq_group_metadata_list))
        has_been_expanded = [False for _ in range(sum(batch_expand_size))]
        #we need this list to write the kv cache for the new allocated block
        #this maybe optimized by change the slot mapping 
        # from int to list to write more than one kvcache at once
        kv_cache_write_req_list = []
        
        for i,seq_metadata in enumerate(execute_model_req.seq_group_metadata_list):
            seq_id = seq_metadata.get_first_seq_id()
            expand_seq_group_metadata_list[batch_start_idx[i]] = seq_metadata
            tree_proposals = proposals.tree_proposals[seq_id]   #List[List[TokenId]]
            has_been_expanded[batch_start_idx[i]] = True
            seq_len = len(seq_metadata.seq_data[all_seq_ids[i]].get_token_ids())
            
            block_table = seq_metadata.block_tables[all_seq_ids[i]]         #List[int]

            
            if seq_len // self.block_manager.block_size + 1 < len(block_table):
                block_table_type = self.block_manager.block_tables[all_seq_ids[i]]    #Dict[SeqId, BlockTable]
                self.block_manager.block_allocator.free(block_table_type.blocks[-1])
                self.block_manager.block_tables[all_seq_ids[i]] = BlockTable(
                    self.block_manager.block_size,
                    self.block_manager.block_allocator,
                    block_table_type._blocks._blocks[:-1]
                )

                block_table = block_table[:-1]
                
                print("have reallocated block")
                
            for proposal in tree_proposals:
                if has_been_expanded[batch_start_idx[i] + proposals.prefix_dict[seq_id][tuple(proposal)]]:
                    continue
                else:
                    new_allocated_block = self.block_manager.block_allocator.allocate_immutable_block(
                        prev_block=None,
                        token_ids=[],
                        device=Device.GPU,
                        extra_hash=None
                    )
                    proposal_block_table = block_table[:-1] + [new_allocated_block.block_id]
                    execute_model_req.blocks_to_copy.append((block_table[-1], new_allocated_block.block_id))
                    if (seq_len + len(proposal)) // self.block_manager.block_size + 1 > len(proposal_block_table):
                        new_allocated_block1  = self.block_manager.block_allocator.allocate_immutable_block(
                        prev_block=None,
                        token_ids=[],
                        device=Device.GPU,
                        extra_hash=None
                        )
                        proposal_block_table = proposal_block_table + [new_allocated_block1.block_id]
                    for idx in range(len(proposal) + 1):
                        if not idx:
                            batch_offset = 0
                        else:
                            batch_offset = proposals.prefix_dict[seq_id][tuple(proposal[:idx])]
                        if has_been_expanded[batch_start_idx[i] + batch_offset]:
                            kv_cache_write_req_list.append(
                                self._create_single_tree_target_seq_group_metadata(
                                    seq_metadata,
                                    all_seq_ids[i],
                                    next(target_seq_id_iterator),
                                    proposal[:idx],
                                    sampling_params=seq_metadata.sampling_params,
                                    block_table=proposal_block_table
                            ))
                        else:
                            expand_seq_group_metadata_list[batch_start_idx[i] + batch_offset] = self._create_single_tree_target_seq_group_metadata(
                                    seq_metadata,
                                    all_seq_ids[i],
                                    next(target_seq_id_iterator),
                                    proposal[:idx],
                                    sampling_params=seq_metadata.sampling_params,
                                    block_table=proposal_block_table
                            )
                            has_been_expanded[batch_start_idx[i] + batch_offset] = True
        for kvcache_write_req in kv_cache_write_req_list:
            expand_seq_group_metadata_list.append(kvcache_write_req)
        with Timer() as target_executed_time:
            target_sampler_output = self._scorer_worker.execute_model(
                execute_model_req=execute_model_req.clone(
                    seq_group_metadata_list=expand_seq_group_metadata_list))
            
        print(f"expand batch size {len(expand_seq_group_metadata_list)}")
        print(f"target model execute time {target_executed_time.elapsed_time_ms}")

        
        if self.min_execute_time is None:
            self.min_execute_time = target_executed_time.elapsed_time_ms
        else:
            self.min_execute_time = min(self.min_execute_time, target_executed_time.elapsed_time_ms)
        
        if target_executed_time.elapsed_time_ms > self.min_execute_time * 1.5 and \
        (self.max_expand_size is None or len(expand_seq_group_metadata_list) -1 < self.max_expand_size):
            self.max_expand_size = len(expand_seq_group_metadata_list) - 1
            print(f"set the max expand size to {self.max_expand_size}")
        
        assert len(target_sampler_output) == 1, "expected single-step output"
        target_sampler_output = target_sampler_output[0]
        return SpeculativeScores(
            probs=target_sampler_output.sampled_token_probs,
            token_ids=target_sampler_output.sampled_token_ids,
            logprobs=target_sampler_output.logprobs,
            hidden_states=target_sampler_output.hidden_states,
            expand_seq_group_metadata_list=expand_seq_group_metadata_list
        )

