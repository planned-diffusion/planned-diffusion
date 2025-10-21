import torch
import torch.nn.functional as F
from typing import Optional
from .control_tags import ASYNC_START, ASYNC_END, PROMISE_START, PROMISE_END, SYNC_TOKEN_ID, MASK_TOKEN_ID

def create_pd_inputs(
    input_ids: torch.Tensor,
    prev_attention_mask: Optional[torch.Tensor],
    device: torch.device,
    length_scale: Optional[float] = None,
    disable_block_sparsity: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, int, list]:
    
    # Hardcoded token mapping for Dream tokenizer numbers
    number_tokens = {15: 0, 16: 1, 17: 2, 18: 3, 19: 4, 20: 5, 21: 6, 22: 7, 23: 8, 24: 9}
    
    # cut off planning input ids at the most recent sync token
    sync_tokens = (input_ids[0] == SYNC_TOKEN_ID).nonzero()
    planning_start_idx = sync_tokens[-1].item() if len(sync_tokens) > 0 else 0
    
    if planning_start_idx > 0:
        planning_input_ids = input_ids[:, planning_start_idx:] # only parse from most recent planning ids
    else:
        planning_input_ids = input_ids
    
    # Parse promises from planning_input_ids and get block sizes
    block_sizes = []
    block_info = []  # Store (start_idx, end_idx, block_size) for each block
    current_pos = 0
    while current_pos < planning_input_ids.shape[1]:
        # Find next promise start
        promise_starts = (planning_input_ids[0, current_pos:] == PROMISE_START).nonzero()
        if len(promise_starts) == 0:
            break
        start_pos = current_pos + promise_starts[0].item()
        
        # Find promise end
        promise_ends = (planning_input_ids[0, start_pos:] == PROMISE_END).nonzero()
        if len(promise_ends) == 0:
            break
        end_pos = start_pos + promise_ends[0].item()
        
        # Extract promise content (excluding start and end tokens)
        promise = planning_input_ids[0, start_pos+1:end_pos]
        
        # Find the number in the promise to determine block size
        number = None
        for i, token in enumerate(promise):
            if token.item() in number_tokens:
                # Found first digit
                number = number_tokens[token.item()]
                
                # Check if next token is also a number (two-digit case)
                if i + 1 < len(promise) and promise[i + 1].item() in number_tokens:
                    second_digit = number_tokens[promise[i + 1].item()]
                    number = number * 10 + second_digit
                break
            
        if number is None:
            raise ValueError(f"No number found in promise: {promise}")
        
        if length_scale is not None:
            block_size = int(number * length_scale)
        else:
            block_size = number * 10
        block_sizes.append(block_size)
        current_pos = end_pos + 1
    
    num_promises = len(block_sizes)
    assert num_promises > 0, "No promises found in planning sequence"
    
    
    total_block_length = sum(block_size + 2 for block_size in block_sizes)  # +2 for async tags
    
    new_sequence = torch.full(
        (input_ids.shape[0], total_block_length),
        MASK_TOKEN_ID,
        device=device,
        dtype=torch.long
    )
    
    # Create attention mask
    block_attention_mask = torch.zeros(
        (planning_input_ids.shape[0], total_block_length, total_block_length),
        device=device,
        dtype=torch.bool
    )
    
    # For each block, create async block with start/end tokens
    current_pos = 0
    for block_size in block_sizes:
        # Add async start token
        new_sequence[0, current_pos] = ASYNC_START
        current_pos += 1
        
        # Add mask tokens for the block
        block_end = current_pos + block_size
        current_pos = block_end
        
        # Add async end token
        new_sequence[0, current_pos] = ASYNC_END
        current_pos += 1
        
        # Get block indices (including async tokens)
        block_start = current_pos - block_size - 2  # -2 for async tags
        block_end = current_pos
        
        # Store block info for _diff_sample
        block_info.append((block_start + input_ids.shape[1], block_end + input_ids.shape[1], block_size))
        
        # Allow bidirectional attention within the block
        block_attention_mask[0, block_start:block_end, block_start:block_end] = True
        
        # Prevent external tokens from attending to block internals
        block_attention_mask[0, :block_start, block_start+1:block_end-1] = False
        block_attention_mask[0, block_end:, block_start+1:block_end-1] = False

    # If sparsity is disabled, make block attention dense
    if disable_block_sparsity:
        block_attention_mask[:] = True
        
    full_sequence = torch.cat([input_ids, new_sequence], dim=1)
    full_attention_mask = update_attention_mask(prev_attention_mask, block_attention_mask)
    
    return full_sequence, full_attention_mask, num_promises, block_info

def update_attention_mask(prev_attention_mask, new_attention_mask):
    
    batch_size, prev_seq_len, _ = prev_attention_mask.shape
    _, new_seq_len, _ = new_attention_mask.shape
    
    seq_len = prev_seq_len + new_seq_len
    
    # Create expanded mask for the new sequence length
    new_mask = torch.zeros(
        (batch_size, seq_len, seq_len),
        device=prev_attention_mask.device,
        dtype=prev_attention_mask.dtype
    )
    
    new_mask[:, :prev_seq_len, :prev_seq_len] = prev_attention_mask
    
    new_mask[:, -new_seq_len:, -new_seq_len:] = new_attention_mask
    
    new_mask[:,  -new_seq_len:, :prev_seq_len] = True
    
    return new_mask

def invert_and_expand_attention_mask(attention_mask, dtype=torch.float32):
    """
    Invert the attention mask by setting True to 0 and False to torch.finfo(dtype).min.
    """
    # Invert
    attention_mask = torch.where(
        attention_mask,
        torch.tensor(0.0, device=attention_mask.device, dtype=dtype),
        torch.tensor(torch.finfo(dtype).min, device=attention_mask.device, dtype=dtype)
    ).unsqueeze(1)
    # expanded_inverted_attention_mask = inverted_attention_mask.expand(1, 28, -1, -1) # 28 attention heads
    return attention_mask 
    
def is_pd_token(token_id):
    return token_id == ASYNC_START or token_id == ASYNC_END or token_id == PROMISE_START or token_id == PROMISE_END

def create_replace_mask(completed_blocks, block_info, seq_len):
    
    if not block_info:
        return None
    
    # Start with all positions needing replacement
    replace_mask = torch.ones(seq_len, dtype=torch.bool)
    
    start_idx = block_info[0][0]
    
    replace_mask[:start_idx] = False
    
    # Mark completed blocks as NOT needing replacement (they should be cached)
    if completed_blocks:
        for block_idx in completed_blocks:
            if block_idx < len(block_info):
                start_idx, end_idx, _ = block_info[block_idx]
                # replace_mask[start_idx:end_idx+1] = False
                replace_mask[start_idx:end_idx] = False
    
    # Return None if no positions need replacement (all blocks complete)
    if not replace_mask.any():
        return None
    
    replace_mask = replace_mask.unsqueeze(0)
    
    return replace_mask

def pad_key_values(past_key_values, seq_len):
    
    if past_key_values is None:
        return None
    
    padded_layers = []
    
    for layer_idx, (key_states, value_states) in enumerate(past_key_values):
        # key_states and value_states shape: [batch_size, seq_len, hidden_size]
        current_seq_len = key_states.shape[1]  # seq_len is at index 1
        
        if current_seq_len < seq_len:
            # Calculate padding needed
            pad_len = seq_len - current_seq_len
            key_padding = (0, 0, 0, pad_len)
            value_padding = (0, 0, 0, pad_len)
            
            padded_key = torch.nn.functional.pad(key_states, key_padding, mode='constant', value=0)
            padded_value = torch.nn.functional.pad(value_states, value_padding, mode='constant', value=0)
            
            padded_layers.append((padded_key, padded_value))
        else:
            # No padding needed
            padded_layers.append((key_states, value_states))
    
    return tuple(padded_layers)




def block_unmask(logits, mask_index, x, block_info, confidence, x0, t, s, steps, i, alg_temp):
    
    if block_info is None:
        return x
    
    # Create temporary sequence with all sampled tokens
    x_ = torch.zeros_like(x, device=x.device, dtype=torch.long) + x[0, 0]  # Use first token as default
    x_[mask_index] = x0.clone()
    
    # Create full confidence tensor like other algorithms
    full_confidence = torch.full_like(x, -torch.inf, device=x.device, dtype=logits.dtype)
    full_confidence[mask_index] = confidence
    
    if i < steps - 1:
        unmask_ratio = 1 - s / t
    else:
        unmask_ratio = 1.0  # Unmask all remaining tokens on final step
    
    # Process each block independently
    for start_idx, end_idx, block_size in block_info:
        # Get mask indices for this block
        block_mask = mask_index[0, start_idx:end_idx]
        
        if block_mask.sum() == 0:
            continue  # No mask tokens in this block
        
        # Calculate number of tokens to transfer for this block using global ratio
        number_transfer_tokens_block = max(1, int(block_mask.sum() * unmask_ratio))
        
        # Ensure we don't transfer more tokens than available
        number_transfer_tokens_block = min(number_transfer_tokens_block,  block_mask.sum())
        
        if number_transfer_tokens_block > 0:
            # Get confidence for this block from the full confidence tensor
            block_confidence = full_confidence[0, start_idx:end_idx]
            
            # Select top-k tokens within this block
            if alg_temp is None or alg_temp == 0:
                _, transfer_index_block = torch.topk(block_confidence, number_transfer_tokens_block)
            else:
                block_confidence = block_confidence / alg_temp
                block_confidence = F.softmax(block_confidence, dim=-1)
                transfer_index_block = torch.multinomial(block_confidence, num_samples=number_transfer_tokens_block)
            
            # Convert block-local indices to global indices
            global_indices = start_idx + transfer_index_block
            
            # Update the sequence for this block
            x[0, global_indices] = x_[0, global_indices]
    
    return x

def block_unmask_confidence_threshold(logits, mask_index, x, block_info, confidence, x0, steps, i, alg_temp, threshold=0.9, left_tokens_last_step_per_block=None, number_transfer_tokens_per_block=None, og_num_steps=None):


    assert block_info is not None
    # Create temporary sequence with all sampled tokens
    x_ = torch.zeros_like(x, device=x.device, dtype=torch.long) + x[0, 0]
    x_[mask_index] = x0.clone()
    # Create full confidence tensor
    full_confidence = torch.full_like(x, -torch.inf, device=x.device, dtype=logits.dtype)
    full_confidence[mask_index] = confidence
    # Initialize left_tokens_last_step_per_block if not provided
    if left_tokens_last_step_per_block is None:
        left_tokens_last_step_per_block = [0] * len(block_info)
    # number_transfer_tokens_per_block must be provided
    assert number_transfer_tokens_per_block is not None
    # Process each block independently with confidence threshold
    for block_idx, (start_idx, end_idx, block_size) in enumerate(block_info):
        block_mask = mask_index[0, start_idx:end_idx]
        if block_mask.sum() == 0:
            continue  # No mask tokens in this block
        
        number_transfer_tokens_block = block_mask.sum().item() // (og_num_steps - i) if i < og_num_steps  else 0
        
        number_transfer_tokens_block = max(1, number_transfer_tokens_block)
        number_transfer_tokens_block = min(number_transfer_tokens_block, block_mask.sum().item())
        
        left_tokens_last_step_block = left_tokens_last_step_per_block[block_idx]
        current_transfer_tokens_block = number_transfer_tokens_block + left_tokens_last_step_block
        
        block_confidence = full_confidence[0, start_idx:end_idx]

        selected_confidence, select_index = torch.topk(block_confidence, current_transfer_tokens_block)

        transfer_index_block = torch.zeros_like(block_confidence, dtype=torch.bool)
        select_index = select_index.to(x.device)
        
        transfer_index_block[select_index] = True # Bug fix
        left_tokens_last_step_per_block[block_idx] = 0
        

        for k in range(1, current_transfer_tokens_block):
            if selected_confidence[k] < threshold:
                if i < steps - 1:
                    left_tokens_last_step_per_block[block_idx] += 1
                    transfer_index_block[select_index[k]] = False
                else:
                    number_transfer_tokens_per_block[block_idx] = 0
                    steps += 1
                    left_tokens_last_step_per_block[block_idx] += 1
                    transfer_index_block[select_index[k]] = False
                
        global_indices = start_idx + torch.where(transfer_index_block)[0]
        if len(global_indices) > 0:
            x[0, global_indices] = x_[0, global_indices]
            
            
    return x, steps, left_tokens_last_step_per_block, number_transfer_tokens_per_block 
