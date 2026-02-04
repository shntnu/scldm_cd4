"""From https://github.com/epfml/schedules-and-scaling/blob/main/flops.ipynb"""


def embedding(seq_len, vocab_size, d_model):
    return 2 * seq_len * vocab_size * d_model


def attention(seq_len, d_model, key_size, num_heads):
    projections = 2 * 3 * seq_len * d_model * (key_size * num_heads)
    logits = 2 * seq_len * seq_len * (key_size * num_heads)
    softmax = 3 * num_heads * seq_len * seq_len
    softmax_query_reduction = 2 * seq_len * seq_len * (key_size * num_heads)
    final_layer = 2 * seq_len * (key_size * num_heads) * d_model
    return projections + logits + softmax + softmax_query_reduction + final_layer


def dense(seq_len, d_model, ffw_size, swiglu=False):
    if not swiglu:
        return 2 * seq_len * (2 * d_model * ffw_size)
    else:
        return 2 * seq_len * (3 * d_model * ffw_size)


def final_logits(seq_len, d_model, vocab_size):
    return 2 * seq_len * d_model * vocab_size


def get_flops(
    n_layers,
    seq_len,
    vocab_size,
    d_model,
    key_size,
    num_heads,
    ffw_size,
    swiglu=False,
    **kwargs,
):
    return (
        embedding(seq_len, vocab_size, d_model)
        + n_layers
        * (attention(seq_len, d_model, key_size, num_heads) + dense(seq_len, d_model, ffw_size, swiglu=swiglu))
        + final_logits(seq_len, d_model, vocab_size)
    )
