from typing import Optional

import torch

__all__ = ["encode_text"]


@torch.no_grad()
def encode_text(
    tokenizer,
    text_encoder,
    text: str,
    context_sentence: Optional[str] = None,
    remove_special_tokens: bool = True,
    padding=False,
) -> "torch.Tensor":
    """Encode a text into a sequence of tokens based on the stable diffusion
    text encoder and tokenizer.

    Arguments
    ---------
    text : str
        The text to encode.
    context_sentence : str
        The context sentence to encode. If None, the text is used as context.

    Returns
    -------
    torch.Tensor
        The encoded text. Shape: (tokens, embedding_size)
    """

    device = text_encoder.device

    if context_sentence is None:
        context_sentence = text

    tokens = tokenizer(context_sentence, padding=padding, return_tensors="pt")

    text_embeddings = text_encoder(
        tokens.input_ids.to(device), attention_mask=tokens.attention_mask.to(device)
    )
    text_embeddings = text_embeddings[0][0]  # Discard hidden states

    # Discard special tokens (<SOT>, <EOT>)
    if remove_special_tokens:
        text_embeddings = text_embeddings[1:-1, :]

    # Discard tokens that are not in the text
    if text != context_sentence:
        text_input_ids = (
            tokenizer(text, padding=False, return_tensors="pt")
            .input_ids.numpy()
            .flatten()
        )
        token_input_ids = tokens.input_ids.numpy().flatten()
        if remove_special_tokens:
            token_input_ids = token_input_ids[1:-1]
            text_input_ids = text_input_ids[1:-1]

        selected_ids = [
            token_input_id in text_input_ids for token_input_id in token_input_ids
        ]
        text_embeddings = text_embeddings[selected_ids, :]

    return text_embeddings
