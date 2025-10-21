import typing
from dataclasses import dataclass

@dataclass
class SpecialTokens:
    """
    Dataclass to hold special tokens for PASTA diffusion.
    Can handle both string tokens and integer token IDs.
    """
    async_start: typing.Union[str, int]
    async_end: typing.Union[str, int] 
    sync: typing.Union[str, int]
    eos: typing.Union[str, int]
    pad: typing.Union[str, int]

# Default string-based special tokens for easy reference
DEFAULT_SPECIAL_TOKENS = SpecialTokens(
    async_start="<async>",
    async_end="</async>",
    sync="<sync/>",
    eos="<|im_end|>",
    pad="[PAD]"
)

def special_tokens_from_tokenizer(tokenizer) -> SpecialTokens:
    """
    Helper function to convert string special tokens to token IDs using a HuggingFace tokenizer.
    
    Args:
        tokenizer: HuggingFace tokenizer with encode() method
    
    Returns:
        SpecialTokens: SpecialTokens instance with token IDs instead of strings
    
    Example:
        from train.special_tokens import special_tokens_from_tokenizer
        special_tokens = special_tokens_from_tokenizer(tokenizer)
        # Use with attention_mask.py
        mask, labels, ar_mask = compute_pasta_diffusion_metadata(token_ids, special_tokens)
        # Use with pad.py  
        padded_tokens = pad(tokens, "const", special_tokens=special_tokens, padding=5)
    """
    async_start_id = tokenizer.encode(DEFAULT_SPECIAL_TOKENS.async_start, add_special_tokens=False)[0]
    async_end_id = tokenizer.encode(DEFAULT_SPECIAL_TOKENS.async_end, add_special_tokens=False)[0]
    sync_id = tokenizer.encode(DEFAULT_SPECIAL_TOKENS.sync, add_special_tokens=False)[0]
    eos_id = tokenizer.encode(DEFAULT_SPECIAL_TOKENS.eos, add_special_tokens=False)[0]
    pad_id = tokenizer.encode(DEFAULT_SPECIAL_TOKENS.pad, add_special_tokens=False)[0]
    return SpecialTokens(
        async_start=async_start_id,
        async_end=async_end_id,
        sync=sync_id,
        eos=eos_id,
        pad=pad_id
    )

def get_special_tokens_for_type(tokens: typing.Union[typing.List[str], typing.List[int]]) -> SpecialTokens:
    """
    Automatically determine special tokens based on the type of input tokens.
    
    Args:
        tokens: List of tokens (strings) or token_ids (integers)
        
    Returns:
        SpecialTokens: Appropriate special tokens for the input type
        
    Raises:
        TypeError: If tokens are not str or int type
        ValueError: If tokens list is empty
        
    Example:
        # For string tokens
        special_tokens = get_special_tokens_for_type(["<bos>", "<async>", "A", "</async>"])
        # Returns DEFAULT_SPECIAL_TOKENS with string values
        
        # For token IDs  
        special_tokens = get_special_tokens_for_type([1, 4, 14, 5])
        # Returns SpecialTokens with hardcoded int values
    """
    if not tokens:
        raise ValueError("Cannot determine special token type from empty tokens list")
    
    first_token = tokens[0]
    
    if isinstance(first_token, str):
        return DEFAULT_SPECIAL_TOKENS
    elif isinstance(first_token, int):
        # Hardcoded token IDs - in practice these would come from tokenizer
        return SpecialTokens(
            async_start=4,   # <async>
            async_end=5,     # </async>
            sync=6,          # <sync/>
            eos=7,           # <|im_end|>
            pad=19           # [PAD]
        )
    else:
        raise TypeError(f"Unsupported token type: {type(first_token).__name__}. Only str and int are supported.")



def test_special_tokens():
    """Test all special token functionality."""
    print("=== TESTING SPECIAL TOKENS MODULE ===")
    
    # Test 1: DEFAULT_SPECIAL_TOKENS
    print("Testing DEFAULT_SPECIAL_TOKENS...")
    assert DEFAULT_SPECIAL_TOKENS.async_start == "<async>"
    assert DEFAULT_SPECIAL_TOKENS.async_end == "</async>"
    assert DEFAULT_SPECIAL_TOKENS.sync == "<sync/>"
    assert DEFAULT_SPECIAL_TOKENS.eos == "<|im_end|>"
    assert DEFAULT_SPECIAL_TOKENS.pad == "[PAD]"
    print("✓ DEFAULT_SPECIAL_TOKENS works correctly")
    
    # Test 2: get_special_tokens_for_type with strings
    print("Testing get_special_tokens_for_type with string tokens...")
    string_tokens = ["<bos>", "<async>", "A", "</async>"]
    string_special_tokens = get_special_tokens_for_type(string_tokens)
    assert string_special_tokens.async_start == "<async>"
    assert string_special_tokens.async_end == "</async>"
    print("✓ get_special_tokens_for_type works with string tokens")
    
    # Test 3: get_special_tokens_for_type with integers
    print("Testing get_special_tokens_for_type with integer tokens...")
    int_tokens = [1, 4, 14, 5]
    int_special_tokens = get_special_tokens_for_type(int_tokens)
    assert int_special_tokens.async_start == 4
    assert int_special_tokens.async_end == 5
    assert int_special_tokens.sync == 6
    assert int_special_tokens.eos == 7
    assert int_special_tokens.pad == 19
    print("✓ get_special_tokens_for_type works with integer tokens")
    
    
    # Test 4: special_tokens_from_tokenizer
    print("Testing special_tokens_from_tokenizer...")
    
    # Mock HuggingFace-style tokenizer
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
    
    tokenizer = MockTokenizer()
    tokenizer_special_tokens = special_tokens_from_tokenizer(tokenizer)
    assert tokenizer_special_tokens.async_start == 100
    assert tokenizer_special_tokens.async_end == 101
    assert tokenizer_special_tokens.sync == 102
    assert tokenizer_special_tokens.eos == 103
    assert tokenizer_special_tokens.pad == 104
    print("✓ special_tokens_from_tokenizer works correctly")
    
    # Test 5: Error cases for get_special_tokens_for_type
    print("Testing error cases for get_special_tokens_for_type...")
    
    # Empty list
    try:
        get_special_tokens_for_type([])
        assert False, "Should have failed with empty list"
    except ValueError:
        print("✓ Empty list correctly rejected")
    
    # Unsupported type
    try:
        get_special_tokens_for_type([1.5, 2.5])  # float tokens
        assert False, "Should have failed with unsupported type"
    except TypeError:
        print("✓ Unsupported type correctly rejected")
    
    print("\n🎉 All special tokens tests passed!")

if __name__ == "__main__":
    test_special_tokens() 