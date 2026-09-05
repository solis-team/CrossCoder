def _get_encoding(encoding: str = "cl100k_base"):
    import tiktoken
    return tiktoken.get_encoding(encoding)


def count_tokens(text: str, encoding: str = "cl100k_base") -> int:
    """Return the number of tokens for the given text """
    enc = _get_encoding(encoding)
    return len(enc.encode(text))


def truncate_last_tokens(text: str, max_tokens: int = 4096, encoding: str = "cl100k_base") -> str:
    """
    If text has more than max_tokens tokens, keep the last max_tokens 
    """
    enc = _get_encoding(encoding)
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    tokens = tokens[-max_tokens:]
    return enc.decode(tokens)
