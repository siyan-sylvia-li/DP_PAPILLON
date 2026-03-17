def estimate_name_token_fraction(completions, name_sets, tokenizer):
    """
    For each completion, compute what fraction of tokens are name tokens.
    High fraction → name substitution back matters.
    Low fraction → name substitution back is unnecessary.
    """
    fractions = []
    for comp in completions:
        tokens = tokenizer.tokenize(comp)
        name_token_count = sum(
            1 for t in tokens 
            if any(name.lower() in t.lower() for name in name_sets)
        )
        fractions.append(name_token_count / len(tokens))
    return np.mean(fractions), np.std(fractions)