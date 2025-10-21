import random
import typing
from .special_tokens import SpecialTokens, DEFAULT_SPECIAL_TOKENS, get_special_tokens_for_type

def pad(tokens: typing.Union[typing.List[str], typing.List[int]], 
        mode: str, 
        special_tokens: typing.Optional[SpecialTokens] = None,
        padding: typing.Optional[int] = None, 
        padding_low: typing.Optional[int] = None, 
        padding_high: typing.Optional[int] = None) -> typing.Union[typing.List[str], typing.List[int]]:
    """
    Pad tokens inside async blocks.
    Works with both string tokens and integer token_ids.
    
    Args:
        tokens: List of tokens (strings) or token_ids (integers)
        mode: Padding mode ("const" for constant padding, "stochastic" for random padding)
        special_tokens: SpecialTokens instance (optional - will auto-detect based on token type if not provided)
        padding: Number of padding tokens to add (used for "const" mode)
        padding_low: Lower bound for stochastic padding (inclusive)
        padding_high: Upper bound for stochastic padding (exclusive)
    
    Returns:
        List of padded tokens (same type as input)
        
    Examples:
        # Simplest usage (automatic detection of everything)
        pad(["<async>", "A", "</async>"], "const", padding=2)
        pad([4, 14, 5], "const", padding=2)
        
        # Explicit special tokens (useful when token IDs don't match defaults)
        from train.special_tokens import special_tokens_from_tokenizer
        special_tokens = special_tokens_from_tokenizer(tokenizer)
        pad(token_ids, "const", special_tokens=special_tokens, padding=2)
    """
    if not tokens:
        return tokens
    
    # Auto-detect special tokens if not provided
    if special_tokens is None:
        special_tokens = get_special_tokens_for_type(tokens)
    
    # Always use pad token from special_tokens
    pad_token = special_tokens.pad
    
    # Validate padding parameters
    if mode == "const":
        if padding is None:
            raise ValueError("padding parameter is required for const mode")
        pad_amount = padding
    elif mode == "stochastic":
        if padding_low is None or padding_high is None:
            raise ValueError("padding_low and padding_high parameters are required for stochastic mode")
        if padding_low >= padding_high:
            raise ValueError("padding_low must be less than padding_high")
    else:
        raise ValueError(f"Unsupported padding mode: {mode}")
    
    result = []
    i = 0
    
    while i < len(tokens):
        token = tokens[i]
        result.append(token)
        
        # Check if we're at the start of an async block
        if token == special_tokens.async_start:
            # Find the matching async_end token
            j = i + 1
            async_content = []
            
            # Collect tokens inside the async block
            while j < len(tokens) and tokens[j] != special_tokens.async_end:
                async_content.append(tokens[j])
                j += 1
            
            # Add the async content to result
            result.extend(async_content)
            
            # Determine padding amount based on mode
            if mode == "const":
                current_pad_amount = pad_amount
            elif mode == "stochastic":
                current_pad_amount = random.randint(padding_low, padding_high - 1)
            
            # Add padding tokens before async_end
            result.extend([pad_token] * current_pad_amount)
            
            # Add the async_end token if we found it
            if j < len(tokens) and tokens[j] == special_tokens.async_end:
                result.append(tokens[j])
                i = j  # Skip to the async_end token
            else:
                # If no matching async_end found, just continue
                i = j - 1
        
        i += 1
    
    return result

# Test Case 1: Single async block
tokens1 = [
    "<bos>", 
    "<promise>-<topic>", "task1", "3", "</topic>",
    "<async>",
        "A", "B", "C",
    "</async>",
    "<|im_end|>",
]

expected_const_pad_5_tokens1 = [
    "<bos>", 
    "<promise>-<topic>", "task1", "3", "</topic>",
    "<async>",
        "A", "B", "C", "[PAD]", "[PAD]", "[PAD]", "[PAD]", "[PAD]",
    "</async>",
    "<|im_end|>",
]

# Test Case 2: Sequential async blocks
tokens2 = [
    "<bos>", 
    "<promise>-<topic>", "task1", "2", "</topic>",
    "<async>", "A", "B", "</async>", "<sync/>",
    "<promise>-<topic>", "task2", "1", "</topic>",
    "<async>", "C", "</async>",
    "<|im_end|>",
]

expected_const_pad_5_tokens2 = [
    "<bos>", 
    "<promise>-<topic>", "task1", "2", "</topic>",
    "<async>", "A", "B", "[PAD]", "[PAD]", "[PAD]", "[PAD]", "[PAD]", "</async>", "<sync/>",
    "<promise>-<topic>", "task2", "1", "</topic>",
    "<async>", "C", "[PAD]", "[PAD]", "[PAD]", "[PAD]", "[PAD]", "</async>",
    "<|im_end|>",
]

# Test Case 3: Parallel async blocks
tokens3 = [
    "<bos>", 
    "<promise>-<topic>", "task1", "2", "</topic>",
    "<promise>-<topic>", "task2", "2", "</topic>",
    "<async>", "A", "B", "</async>",
    "<async>", "C", "D", "</async>",
    "<|im_end|>",
]

expected_const_pad_5_tokens3 = [
    "<bos>", 
    "<promise>-<topic>", "task1", "2", "</topic>",
    "<promise>-<topic>", "task2", "2", "</topic>",
    "<async>", "A", "B", "[PAD]", "[PAD]", "[PAD]", "[PAD]", "[PAD]", "</async>",
    "<async>", "C", "D", "[PAD]", "[PAD]", "[PAD]", "[PAD]", "[PAD]", "</async>",
    "<|im_end|>",
]

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

expected_const_pad_5_tokens4 = [
    "<bos>",
    "<promise>-<topic>", "task1", "3", "</topic>", 
    "<promise>-<topic>", "task2", "2", "</topic>",
    "<async>", "A", "B", "[PAD]", "[PAD]", "[PAD]", "[PAD]", "[PAD]", "</async>", 
    "<async>", "C", "D", "[PAD]", "[PAD]", "[PAD]", "[PAD]", "[PAD]", "</async>",
    "<sync/>",
    "<promise>-<topic>", "task3", "1", "</topic>",
    "<async>", "E", "[PAD]", "[PAD]", "[PAD]", "[PAD]", "[PAD]", "</async>",
    "<|im_end|>",
]

# Test verification
if __name__ == "__main__":
    # Vocabulary mapping for token ID tests
    VOCAB = {
        "<bos>": 1, "<promise>-<topic>": 2, "</topic>": 3, "<async>": 4, "</async>": 5, 
        "<sync/>": 6, "<|im_end|>": 7, "task1": 8, "task2": 9, "task3": 10,
        "1": 11, "2": 12, "3": 13, "A": 14, "B": 15, "C": 16, "D": 17, "E": 18, "[PAD]": 19
    }
    
    print("=== TESTING STRING TOKENS ===")
    
    # Test Case 1
    result1 = pad(tokens1, "const", padding=5)
    assert result1 == expected_const_pad_5_tokens1, f"Test 1 failed:\nExpected: {expected_const_pad_5_tokens1}\nGot: {result1}"
    print("✓ Test Case 1 (String tokens) passed")
    
    # Test Case 2  
    result2 = pad(tokens2, "const", padding=5)
    assert result2 == expected_const_pad_5_tokens2, f"Test 2 failed:\nExpected: {expected_const_pad_5_tokens2}\nGot: {result2}"
    print("✓ Test Case 2 (String tokens) passed")
    
    # Test Case 3
    result3 = pad(tokens3, "const", padding=5)
    assert result3 == expected_const_pad_5_tokens3, f"Test 3 failed:\nExpected: {expected_const_pad_5_tokens3}\nGot: {result3}"
    print("✓ Test Case 3 (String tokens) passed")
    
    # Test Case 4
    result4 = pad(tokens4, "const", padding=5)
    assert result4 == expected_const_pad_5_tokens4, f"Test 4 failed:\nExpected: {expected_const_pad_5_tokens4}\nGot: {result4}"
    print("✓ Test Case 4 (String tokens) passed")
    
    print("All constant padding test cases (String tokens) passed! 🎉")
    
    print("\n=== TESTING TOKEN IDs ===")
    
    # Convert test cases to token IDs
    token_ids1 = [VOCAB[token] for token in tokens1]
    token_ids2 = [VOCAB[token] for token in tokens2]
    token_ids3 = [VOCAB[token] for token in tokens3]
    token_ids4 = [VOCAB[token] for token in tokens4]
    
    expected_token_ids1 = [VOCAB[token] for token in expected_const_pad_5_tokens1]
    expected_token_ids2 = [VOCAB[token] for token in expected_const_pad_5_tokens2]
    expected_token_ids3 = [VOCAB[token] for token in expected_const_pad_5_tokens3]
    expected_token_ids4 = [VOCAB[token] for token in expected_const_pad_5_tokens4]
    
    # Test Case 1 with token IDs
    result1_ids = pad(token_ids1, "const", padding=5)
    assert result1_ids == expected_token_ids1, f"Test 1 (Token IDs) failed:\nExpected: {expected_token_ids1}\nGot: {result1_ids}"
    print("✓ Test Case 1 (Token IDs) passed")
    
    # Test Case 2 with token IDs
    result2_ids = pad(token_ids2, "const", padding=5)
    assert result2_ids == expected_token_ids2, f"Test 2 (Token IDs) failed:\nExpected: {expected_token_ids2}\nGot: {result2_ids}"
    print("✓ Test Case 2 (Token IDs) passed")
    
    # Test Case 3 with token IDs
    result3_ids = pad(token_ids3, "const", padding=5)
    assert result3_ids == expected_token_ids3, f"Test 3 (Token IDs) failed:\nExpected: {expected_token_ids3}\nGot: {result3_ids}"
    print("✓ Test Case 3 (Token IDs) passed")
    
    # Test Case 4 with token IDs
    result4_ids = pad(token_ids4, "const", padding=5)
    assert result4_ids == expected_token_ids4, f"Test 4 (Token IDs) failed:\nExpected: {expected_token_ids4}\nGot: {result4_ids}"
    print("✓ Test Case 4 (Token IDs) passed")
    
    print("All constant padding test cases (Token IDs) passed! 🎉")
    
    print("\n=== TESTING STOCHASTIC PADDING ===")
    
    # Test stochastic padding with string tokens
    print("Testing stochastic padding (String tokens):")
    result_stochastic = pad(tokens1, "stochastic", padding_low=2, padding_high=8)
    
    # Count padding tokens in the async block
    async_start = result_stochastic.index("<async>")
    async_end = result_stochastic.index("</async>", async_start)
    padding_count = result_stochastic[async_start+1:async_end].count("[PAD]")
    print(f"String tokens - Padding tokens added: {padding_count} (should be between 2 and 7)")
    
    # Test stochastic padding with token IDs
    print("Testing stochastic padding (Token IDs):")
    result_stochastic_ids = pad(token_ids1, "stochastic", padding_low=2, padding_high=8)
    
    # Count padding tokens in the async block
    async_start_ids = result_stochastic_ids.index(VOCAB["<async>"])
    async_end_ids = result_stochastic_ids.index(VOCAB["</async>"], async_start_ids)
    padding_count_ids = result_stochastic_ids[async_start_ids+1:async_end_ids].count(VOCAB["[PAD]"])
    print(f"Token IDs - Padding tokens added: {padding_count_ids} (should be between 2 and 7)")
    
    # Test multiple runs to show randomness
    print("\nMultiple stochastic runs (String tokens):")
    for i in range(3):
        result = pad(tokens1, "stochastic", padding_low=1, padding_high=6)
        async_start = result.index("<async>")
        async_end = result.index("</async>", async_start)
        padding_count = result[async_start+1:async_end].count("[PAD]")
        print(f"Run {i+1}: {padding_count} padding tokens")
    
    print("Multiple stochastic runs (Token IDs):")
    for i in range(3):
        result_ids = pad(token_ids1, "stochastic", padding_low=1, padding_high=6)
        async_start = result_ids.index(VOCAB["<async>"])
        async_end = result_ids.index(VOCAB["</async>"], async_start)
        padding_count = result_ids[async_start+1:async_end].count(VOCAB["[PAD]"])
        print(f"Run {i+1}: {padding_count} padding tokens")
    
    print("✓ Stochastic padding test completed!")