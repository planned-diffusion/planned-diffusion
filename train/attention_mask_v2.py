import torch
import typing
from .special_tokens import SpecialTokens, DEFAULT_SPECIAL_TOKENS, special_tokens_from_tokenizer

"""
PASTA Diffusion Attention Rules:

1. Default Causal Attention: All tokens use standard causal attention (can only see previous tokens)

2. Async Block Bidirectionality: Within <async>...</async> blocks, all tokens can attend to each other bidirectionally  

3. Async Block Isolation: External tokens cannot attend to any tokens inside async blocks
"""

def compute_pasta_diffusion_metadata_v2(tokens, special_tokens: SpecialTokens, position_multiple: int = 10):
    """
    Compute PASTA attention mask and hybrid autoregressive-diffusion labels.
    Works with both string tokens and integer token_ids.
    
    Args:
        tokens: List of tokens (strings) or token_ids (integers)
        special_tokens: SpecialTokens dataclass containing async_start, async_end, sync tokens
        position_multiple: Specifies the number to multiply the predicted position by.
    
    Returns:
        (mask, labels, ar_mask): Attention mask, hybrid labels, autoregressive mask
    """
    n = len(tokens)
    mask = torch.tril(torch.ones(n, n, dtype=torch.float32))
    labels = []
    ar_mask = torch.ones(n, dtype=torch.float32)  # Start with all autoregressive
    
    # Track async blocks
    current_async_start = None
    completed_blocks = []
    
    for i, token in enumerate(tokens):
        # === ATTENTION MASK & DIFFUSION LABELS ===
        if token == special_tokens.async_start:
            current_async_start = i
            ar_mask[i] = 0  # async start token is diffusion
        elif token == special_tokens.async_end and current_async_start is not None:
            ar_mask[i] = 0  # async end token is diffusion
            start, end = current_async_start, i
            block_slice = slice(start, end + 1)
            
            # Set all tokens in async block to diffusion (0)
            ar_mask[block_slice] = 0
            
            # Apply PASTA attention rules using slicing
            mask[block_slice, block_slice] = 1.0           # Bidirectional within block
            mask[:start, block_slice] = 0.0                # External can't see in
            mask[end + 1:, block_slice] = 0.0              # External can't see in
            
            # Block can see pre-async context
            first_async = min([s for s, _ in completed_blocks] + [start])
            mask[block_slice, :first_async] = 1.0
            
            completed_blocks.append((start, end))
            
            # Add diffusion labels (exclude async start token, include content tokens, duplicate async end token)
            content_tokens = tokens[start + 1:end + 1]  # Skip <async> start token
            labels.extend(content_tokens)  # Add content tokens
            labels.append(special_tokens.async_end)  # Add </async> end token again
            current_async_start = None
            
        elif current_async_start is not None:
            # Token inside async block
            ar_mask[i] = 0
            
        elif token == special_tokens.sync:
            # Make completed blocks visible to future tokens
            for start, end in completed_blocks:
                mask[i:, start:end + 1] = 1.0
        
        # === AUTOREGRESSIVE LABELS ===
        not_in_async_block = current_async_start is None
        has_next_token = i + 1 < n
        not_async_end_token = token != special_tokens.async_end
        
        if not_in_async_block and has_next_token and not_async_end_token:
            # Skip consecutive async blocks to find next autoregressive token
            next_pos = i + 1
            while next_pos < n and tokens[next_pos] == special_tokens.async_start:
                try:
                    next_pos = tokens[next_pos:].index(special_tokens.async_end) + next_pos + 1
                except ValueError:
                    break  # No matching end token found
            
            if next_pos < n:
                labels.append(tokens[next_pos])
    
    labels.append(special_tokens.eos)
    return mask, labels, ar_mask


def assert_attention_mask(actual_mask, expected_mask, tokens, test_name):
    """Helper function to compare attention masks"""
    if not torch.allclose(actual_mask, expected_mask, atol=1e-6):
        print(f"\nFAILED: {test_name}")
        print(f"Tokens: {tokens}")
        print("Expected mask:")
        print(expected_mask)
        print("Actual mask:")
        print(actual_mask)
        print("Difference:")
        print(actual_mask - expected_mask)
        raise AssertionError(f"Attention mask mismatch in {test_name}")
    else:
        print(f"PASSED: {test_name}")

def assert_labels(actual_labels, expected_labels, tokens, test_name):
    """Helper function to compare labels"""
    if actual_labels != expected_labels:
        print(f"\nFAILED: {test_name} - Labels mismatch")
        print(f"Tokens: {tokens}")
        print(f"Expected labels ({len(expected_labels)}): {expected_labels}")
        print(f"Actual labels ({len(actual_labels)}): {actual_labels}")
        raise AssertionError(f"Labels mismatch in {test_name}")
    else:
        print(f"PASSED: {test_name} - Labels match")

def assert_ar_mask(actual_ar_mask, expected_ar_mask, tokens, test_name):
    """Helper function to compare autoregressive masks"""
    if not torch.allclose(actual_ar_mask, expected_ar_mask, atol=1e-6):
        print(f"\nFAILED: {test_name} - AR mask mismatch")
        print(f"Tokens: {tokens}")
        print(f"Expected AR mask: {expected_ar_mask}")
        print(f"Actual AR mask: {actual_ar_mask}")
        print("Difference:")
        print(actual_ar_mask - expected_ar_mask)
        raise AssertionError(f"AR mask mismatch in {test_name}")
    else:
        print(f"PASSED: {test_name} - AR mask match")

def test_simple_async_blocks():
    # Single comprehensive vocabulary for all test cases
    VOCAB = {
        "<bos>": 1, "<promise>-<topic>": 2, "</topic>": 3, "<async>": 4, "</async>": 5, 
        "<sync/>": 6, "<|im_end|>": 7, "task1": 8, "task2": 9, "task3": 10,
        "1": 11, "2": 12, "3": 13, "A": 14, "B": 15, "C": 16, "D": 17, "E": 18, "[PAD]": 19
    }
    
    # Single special tokens instance for all token ID tests
    SPECIAL_TOKENS_IDS = SpecialTokens(
        async_start=VOCAB["<async>"], 
        async_end=VOCAB["</async>"], 
        sync=VOCAB["<sync/>"],
        eos=VOCAB["<|im_end|>"],
        pad=VOCAB["[PAD]"]
    )
    
    # Test Case 1: Single async block
    tokens1 = [
        "<bos>", 
        "<promise>-<topic>", "task1", "3", "</topic>",
        "<async>",
            "A", "B", "C",
        "</async>",
        "<|im_end|>",
    ]
    labels1 = [
        "<promise>-<topic>", # <- <bos>,
        "task1", # <- <promise>-<topic>,
        "3", # <- task1,
        "</topic>", # <- 3,
        "<|im_end|>", # <- </topic>,
        # diffusion labels
        "A", "B", "C", "</async>", "</async>",
        "<|im_end|>" # padding so len(labels) == len(tokens)
    ]
    expected_ar_mask1 = torch.tensor([
        1, # <bos>
        1, # <promise>-<topic>
        1, # task1
        1, # 3
        1, # </topic>
        0, # <async>
        0, # A
        0, # B
        0, # C
        0, # </async>
        1, # <|im_end|>
    ], dtype=torch.float32)

    # Expected attention mask for single async block (11x11 causal mask)
    expected_mask1 = torch.tensor([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # <bos>
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # <promise>-<topic>
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # task1
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # 3
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # </topic>
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # <async> - bidirectional within async block
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # A - bidirectional within async block
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # B - bidirectional within async block
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # C - bidirectional within async block
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # </async> - can see everything including async internals
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],  # <|im_end|> - cannot see async internals, only boundaries
    ], dtype=torch.float32)

    # Test Case 2: Sequential async blocks
    tokens2 = [
        "<bos>", 
        "<promise>-<topic>", "task1", "2", "</topic>",
        "<async>", "A", "B", "</async>", "<sync/>",
        "<promise>-<topic>", "task2", "1", "</topic>",
        "<async>", "C", "</async>",
        "<|im_end|>",
    ]
    labels2 = [
        "<promise>-<topic>", # <- <bos>,
        "task1", # <- <promise>-<topic>,
        "2", # <- task1,
        "</topic>", # <- 2,
        "<sync/>", # <- </topic>,
        "A", "B", "</async>", "</async>", # diffusion labels for first async block
        "<promise>-<topic>", # <- <sync/>,
        "task2", # <- <promise>-<topic>,
        "1", # <- task2,
        "</topic>", # <- 1,
        "<|im_end|>", # <- </topic>,
        "C", "</async>", "</async>", # diffusion labels for second async block
        "<|im_end|>" # padding so len(labels) == len(tokens)
    ]
    expected_ar_mask2 = torch.tensor([
        1, # <bos>
        1, # <promise>-<topic>
        1, # task1
        1, # 2
        1, # </topic>
        0, # <async>
        0, # A
        0, # B
        0, # </async>
        1, # <sync/>
        1, # <promise>-<topic>
        1, # task2
        1, # 1
        1, # </topic>
        0, # <async>
        0, # C
        0, # </async>
        1, # <|im_end|>
    ], dtype=torch.float32)

    # Expected attention mask for sequential async blocks (18x18)
    expected_mask2 = torch.tensor([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # <bos>
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # <promise>-<topic>
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # task1
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # </topic>
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # <async> - bidirectional within first async
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # A - bidirectional within first async
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # B - bidirectional within first async
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # </async> - bidirectional within first async
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # <sync/> - can see first async block after sync
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # <promise>-<topic> - can see first async block
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # task2 - can see first async block
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],  # 1 - can see first async block
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # </topic> - can see first async block
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # <async> - bidirectional within second async
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # C - bidirectional within second async
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # </async> - bidirectional within second async
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1],  # <|im_end|> - cannot see async internals
    ], dtype=torch.float32)

    # Test Case 3: Parallel async blocks
    tokens3 = [
        "<bos>", 
        "<promise>-<topic>", "task1", "2", "</topic>",
        "<promise>-<topic>", "task2", "2", "</topic>",
        "<async>", "A", "B", "</async>",
        "<async>", "C", "D", "</async>",
        "<|im_end|>",
    ]
    labels3 = [
        "<promise>-<topic>", # <- <bos>,
        "task1", # <- <promise>-<topic>,
        "2", # <- task1,
        "</topic>", # <- 2, 
        "<promise>-<topic>", # <- </topic>,
        "task2", # <- <promise>-<topic>,
        "2", # <- task2,
        "</topic>", # <- 2,
        "<|im_end|>", # <- </topic>,
        "A", "B", "</async>", "</async>", # <- diffusion labels
        "C", "D", "</async>", "</async>", # <- diffusion labels
        "<|im_end|>" # padding so len(labels) == len(tokens)
    ]
    expected_ar_mask3 = torch.tensor([
        1, # <bos>
        1, # <promise>-<topic>
        1, # task1
        1, # 2
        1, # </topic>
        1, # <promise>-<topic>
        1, # task2
        1, # 2
        1, # </topic>
        0, # <async>
        0, # A
        0, # B
        0, # </async>
        0, # <async>
        0, # C
        0, # D
        0, # </async>
        1, # <|im_end|>
    ], dtype=torch.float32)

    # Expected attention mask for parallel async blocks (18x18)
    expected_mask3 = torch.tensor([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # <bos>
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # <promise>-<topic>
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # task1
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # </topic>
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # <promise>-<topic>
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # task2
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # </topic>
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],  # <async> - bidirectional within first async
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],  # A - bidirectional within first async
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],  # B - bidirectional within first async
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],  # </async> - bidirectional within first async
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0],  # <async> - bidirectional within second async, cannot see first async internals
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0],  # C - bidirectional within second async, cannot see first async internals
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0],  # D - bidirectional within second async, cannot see first async internals
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0],  # </async> - bidirectional within second async, cannot see first async internals
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # <|im_end|> - cannot see async internals
    ], dtype=torch.float32)

    # Test Case 4: Complex sync and async pattern
    tokens4 = [
        "<bos>",
        "<promise>-<topic>", "task1", "3", "</topic>", 
        "<promise>-<topic>", "task2", "2", "</topic>",
        "<async>", "A", "B", "</async>", 
        "<async>", "C", "D", "</async>",
        "<sync/>",
        "<promise>-<topic>", "task3", "1", "</topic>",
        "<async>", "E", "</async>",
        "<|im_end|>",
    ]
    labels4 = [
        "<promise>-<topic>", # <- <bos>
        "task1", # <- <promise>-<topic>
        "3", # <- task1
        "</topic>", # <- 3
        "<promise>-<topic>", # <- </topic>
        "task2", # <- <promise>-<topic>
        "2", # <- task2
        "</topic>", # <- 2
        "<sync/>", # <- </topic>
        "A", "B", "</async>", "</async>", # <- first async block, diffusion labels
        "C", "D", "</async>", "</async>", # <- second async block, diffusion labels
        "<promise>-<topic>", # <- <sync/>
        "task3", # <- <promise>-<topic>
        "1", # <- task3
        "</topic>", # <- 1
        "<|im_end|>", # <- </topic>
        "E", "</async>", "</async>", # <- third async block, diffusion labels
        "<|im_end|>" # padding so len(labels) == len(tokens)
    ]
    expected_ar_mask4 = torch.tensor([
        1, # <bos>
        1, # <promise>-<topic>
        1, # task1
        1, # 3
        1, # </topic>
        1, # <promise>-<topic>
        1, # task2
        1, # 2
        1, # </topic>
        0, # <async>
        0, # A
        0, # B
        0, # </async>
        0, # <async>
        0, # C
        0, # D
        0, # </async>
        1, # <sync/>
        1, # <promise>-<topic>
        1, # task3
        1, # 1
        1, # </topic>
        0, # <async>
        0, # E
        0, # </async>
        1, # <|im_end|>
    ], dtype=torch.float32)
        

    # Expected attention mask for complex sync and async pattern (26x26)
    expected_mask4 = torch.tensor([
        #    0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0: <bos>
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1: <promise>-<topic>
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2: task1
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3: 3
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4: </topic>
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 5: <promise>-<topic>
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 6: task2
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 7: 2
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 8: </topic>
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 9: <async> - bidirectional within first async
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 10: A - bidirectional within first async
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 11: B - bidirectional within first async
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 12: </async> - bidirectional within first async
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 13: <async> - bidirectional within second async, cannot see first async internals
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 14: C - bidirectional within second async, cannot see first async internals
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 15: D - bidirectional within second async, cannot see first async internals
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 16: </async> - bidirectional within second async, cannot see first async internals
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 17: <sync/> - can see both async blocks after sync
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # 18: <promise>-<topic> - can see both async blocks
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # 19: task3 - can see both async blocks
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],  # 20: 1 - can see both async blocks
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # 21: </topic> - can see both async blocks
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # 22: <async> - bidirectional within third async
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # 23: E - bidirectional within third async
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # 24: </async> - bidirectional within third async
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1],  # 25: <|im_end|> - cannot see async internals
    ], dtype=torch.float32)
    
    # ========================================
    # COMPREHENSIVE DUAL TESTING
    # ========================================

    print("Testing single async block...")
    print(f"Tokens: {tokens1}")
    
    # Test with string tokens
    mask_str, labels_str, ar_mask_str = compute_pasta_diffusion_metadata_v2(tokens1, DEFAULT_SPECIAL_TOKENS)
    print(f"String tokens mask shape: {mask_str.shape}")
    print(f"String tokens labels: {labels_str}")
    assert_attention_mask(mask_str, expected_mask1, tokens1, "Single async block (String tokens)")
    assert_labels(labels_str, labels1, tokens1, "Single async block (String tokens)")
    assert_ar_mask(ar_mask_str, expected_ar_mask1, tokens1, "Single async block (String tokens)")
    
    # Test with integer token_ids
    token_ids1 = [VOCAB[token] for token in tokens1]
    expected_labels_ids1 = [VOCAB[token] for token in labels1]
    
    mask_ids, labels_ids, ar_mask_ids = compute_pasta_diffusion_metadata_v2(
        token_ids1, SPECIAL_TOKENS_IDS
    )
    print(f"Token IDs mask shape: {mask_ids.shape}")
    print(f"Token IDs: {token_ids1}")
    assert_attention_mask(mask_ids, expected_mask1, token_ids1, "Single async block (Token IDs)")
    assert_labels(labels_ids, expected_labels_ids1, token_ids1, "Single async block (Token IDs)")
    assert_ar_mask(ar_mask_ids, expected_ar_mask1, token_ids1, "Single async block (Token IDs)")

    # Test Case 2: Sequential async blocks
    print("Testing sequential async blocks...")
    print(f"Tokens: {tokens2}")
    
    # Test with string tokens
    mask_str, labels_str, ar_mask_str = compute_pasta_diffusion_metadata_v2(tokens2, DEFAULT_SPECIAL_TOKENS)
    print(f"String tokens mask shape: {mask_str.shape}")
    assert_attention_mask(mask_str, expected_mask2, tokens2, "Sequential async blocks (String tokens)")
    assert_labels(labels_str, labels2, tokens2, "Sequential async blocks (String tokens)")
    assert_ar_mask(ar_mask_str, expected_ar_mask2, tokens2, "Sequential async blocks (String tokens)")
    
    # Test with integer token_ids
    token_ids2 = [VOCAB[token] for token in tokens2]  
    expected_labels_ids2 = [VOCAB[token] for token in labels2]
    
    mask_ids, labels_ids, ar_mask_ids = compute_pasta_diffusion_metadata_v2(
        token_ids2, SPECIAL_TOKENS_IDS
    )
    print(f"Token IDs mask shape: {mask_ids.shape}")
    assert_attention_mask(mask_ids, expected_mask2, token_ids2, "Sequential async blocks (Token IDs)")
    assert_labels(labels_ids, expected_labels_ids2, token_ids2, "Sequential async blocks (Token IDs)")
    assert_ar_mask(ar_mask_ids, expected_ar_mask2, token_ids2, "Sequential async blocks (Token IDs)")

    # Test Case 3: Parallel async blocks
    print("Testing parallel async blocks...")
    print(f"Tokens: {tokens3}")
    
    # Test with string tokens
    mask_str, labels_str, ar_mask_str = compute_pasta_diffusion_metadata_v2(tokens3, DEFAULT_SPECIAL_TOKENS)
    print(f"String tokens mask shape: {mask_str.shape}")
    print(f"String tokens labels: {labels_str}")
    assert_attention_mask(mask_str, expected_mask3, tokens3, "Parallel async blocks (String tokens)")
    assert_labels(labels_str, labels3, tokens3, "Parallel async blocks (String tokens)")
    assert_ar_mask(ar_mask_str, expected_ar_mask3, tokens3, "Parallel async blocks (String tokens)")
    
    # Test with integer token_ids
    token_ids3 = [VOCAB[token] for token in tokens3]
    expected_labels_ids3 = [VOCAB[token] for token in labels3]
    
    mask_ids, labels_ids, ar_mask_ids = compute_pasta_diffusion_metadata_v2(
        token_ids3, SPECIAL_TOKENS_IDS
    )
    print(f"Token IDs mask shape: {mask_ids.shape}")
    assert_attention_mask(mask_ids, expected_mask3, token_ids3, "Parallel async blocks (Token IDs)")
    assert_labels(labels_ids, expected_labels_ids3, token_ids3, "Parallel async blocks (Token IDs)")
    assert_ar_mask(ar_mask_ids, expected_ar_mask3, token_ids3, "Parallel async blocks (Token IDs)")

    # Test Case 4: Complex sync and async pattern
    print("Testing complex sync and async pattern...")
    print(f"Tokens: {tokens4}")
    
    # Test with string tokens
    mask_str, labels_str, ar_mask_str = compute_pasta_diffusion_metadata_v2(tokens4, DEFAULT_SPECIAL_TOKENS)
    print(f"String tokens mask shape: {mask_str.shape}")
    print(f"String tokens labels: {labels_str}")
    assert_attention_mask(mask_str, expected_mask4, tokens4, "Complex sync and async pattern (String tokens)")
    assert_labels(labels_str, labels4, tokens4, "Complex sync and async pattern (String tokens)")
    assert_ar_mask(ar_mask_str, expected_ar_mask4, tokens4, "Complex sync and async pattern (String tokens)")
    
    # Test with integer token_ids
    token_ids4 = [VOCAB[token] for token in tokens4]  
    expected_labels_ids4 = [VOCAB[token] for token in labels4]
    
    mask_ids, labels_ids, ar_mask_ids = compute_pasta_diffusion_metadata_v2(
        token_ids4, SPECIAL_TOKENS_IDS
    )
    print(f"Token IDs mask shape: {mask_ids.shape}")
    assert_attention_mask(mask_ids, expected_mask4, token_ids4, "Complex sync and async pattern (Token IDs)")
    assert_labels(labels_ids, expected_labels_ids4, token_ids4, "Complex sync and async pattern (Token IDs)")
    assert_ar_mask(ar_mask_ids, expected_ar_mask4, token_ids4, "Complex sync and async pattern (Token IDs)")

def test_user_case():
    """Test case based on user-provided tokens."""
    print("\nTesting user-provided case...")
    
    tokens = ['<|im_start|>', 'system', 'Ċ', 'You', 'Ġare', 'Ġa', 'Ġhelpful', 'Ġassistant', '.', '<|im_end|>', 'Ċ', '<|im_start|>', 'user', 'Ċ', 'Write', 'Ġa', 'Ġtitle', ':Ċ', 'With', 'Ġmore', 'Ġpeople', 'Ġhaving', 'Ġhigh', '-speed', 'Ġconnections', 'Ġat', 'Ġhome', ',', 'Ġthere', 'Ġwill', 'Ġbe', 'Ġless', 'Ġholiday', 'Ġshopping', 'Ġat', 'Ġwork', '.Ċ', 'Title', ':', '<|im_end|>', 'Ċ', '<|im_start|>', 'assistant', 'Ċ', '<promise>-<topic>', 'title', 'Ġ', '1', '</topic>', '<async>', 'Title', ':', 'ĠHigh', '-Speed', 'ĠHome', 'ĠInternet', 'ĠRed', 'uces', 'ĠWorkplace', 'ĠHoliday', 'ĠShopping', '[PAD]', '[PAD]', '</async>', '<|im_end|>', 'Ċ']

    _, labels, _ = compute_pasta_diffusion_metadata_v2(tokens, DEFAULT_SPECIAL_TOKENS)

    print(f"Input Tokens ({len(tokens)}):")
    print(tokens)
    print(f"Computed Labels ({len(labels)}):")
    print(labels)
    print("PASSED: User-provided case processed.")

def test_special_tokens_helper():
    """Test the special_tokens_from_tokenizer helper function."""
    print("Testing special_tokens_from_tokenizer helper...")
    
    # Mock HuggingFace-style tokenizer for testing
    class MockTokenizer:
        def __init__(self):
            self.vocab = {
                "<async>": 100,
                "</async>": 101, 
                "<sync/>": 102,
                "<|im_end|>": 103,
                "[PAD]": 104
            }
        
        def encode(self, text, add_special_tokens=False):
            return [self.vocab[text]]
    
    # Test with mock tokenizer
    tokenizer = MockTokenizer()
    special_tokens = special_tokens_from_tokenizer(tokenizer)
    
    # Verify the token IDs
    assert special_tokens.async_start == 100, f"Expected 100, got {special_tokens.async_start}"
    assert special_tokens.async_end == 101, f"Expected 101, got {special_tokens.async_end}"
    assert special_tokens.sync == 102, f"Expected 102, got {special_tokens.sync}"
    assert special_tokens.eos == 103, f"Expected 103, got {special_tokens.eos}"
    assert special_tokens.pad == 104, f"Expected 104, got {special_tokens.pad}"
    
    print("PASSED: special_tokens_from_tokenizer with HuggingFace-style tokenizer")

if __name__ == "__main__":
    print("Running PASTA attention mask tests...")
    try:
        test_simple_async_blocks()
        test_special_tokens_helper()
        test_user_case()
        print("\nAll tests completed!")
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise