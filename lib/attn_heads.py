import torch

def previous_token_mask(input_ids, model):
    B, S = input_ids.shape
    device = input_ids.device

    causal = torch.tril(torch.zeros(S, S, device=device, dtype=model.dtype))
    causal = causal + torch.triu(
        torch.full((S, S), float("-inf"), device=device, dtype=model.dtype), diagonal=1
    )
    masks = causal.unsqueeze(0).expand(B, S, S).clone()

    row = torch.arange(1, S, device=device)
    col = row - 1
    masks[:, row, col] = float("-inf")

    return masks.unsqueeze(1)


def compute_verbatim_induction_scores(
    model,
    tokenizer,
    num_of_samples=100,
    seq_len=25,
    batch_size=16,
    device=None,
    precomputed_sequences=None,
):
    if device is None:
        device = next(model.parameters()).device

    original_attn_impl = model.config._attn_implementation
    model.config._attn_implementation = "eager"

    induction_scores = torch.zeros(
        model.config.num_hidden_layers, model.config.num_attention_heads
    ).to(device)

    if precomputed_sequences is None:
        vocab_size = tokenizer.vocab_size
        random_sequence = torch.randint(1, vocab_size, (num_of_samples, seq_len))
        random_repetitive_sequence = torch.cat(
            [random_sequence, random_sequence], dim=1
        )
    else:
        random_repetitive_sequence = precomputed_sequences

    model.eval()
    with torch.no_grad():
        for i in range(0, num_of_samples, batch_size):
            begin_index = i
            end_index = min(i + batch_size, num_of_samples)
            batch = random_repetitive_sequence[begin_index:end_index, :]
            input_data = {"input_ids": batch.to(device)}
            result = model(**input_data, output_attentions=True)

            for layer in range(model.config.num_hidden_layers):
                layer_values = result.attentions[layer]

                curr_ind_scores = (
                    layer_values.diagonal(offset=-seq_len + 1, dim1=-2, dim2=-1)[
                        ..., 1:
                    ]
                    .mean(dim=-1)
                    .sum(dim=0)
                )
                induction_scores[layer] += curr_ind_scores

    induction_scores /= num_of_samples

    model.config._attn_implementation = original_attn_impl

    return induction_scores

