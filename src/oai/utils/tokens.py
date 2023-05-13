tokenizer = None


def get_tokenizer():
    global tokenizer
    if tokenizer is None:
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    return tokenizer


def get_token_count(text: str):
    return len(get_tokenizer().tokenize(text))


def truncate_by_n_tokens(text: str, n: int) -> str:
    tokenizer = get_tokenizer()
    return tokenizer.decode(tokenizer(text)["input_ids"][:n])
