SEPARATOR = (" ", "\n", "\t", "'", ",", ".")

def matches_target(gen_text: str, tgt_text: str) -> bool:
    tgt = tgt_text.strip()
    for prefix in (" " + tgt, tgt):
        if gen_text.startswith(prefix):
            next_pos = len(prefix)
            if next_pos == len(gen_text):
                return True
            if gen_text[next_pos] in SEPARATOR:
                return True
    return False
