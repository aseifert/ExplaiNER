from typing import Optional

import pandas as pd
import streamlit as st
from datasets import Dataset  # type: ignore

from src.data import encode_dataset, get_collator, get_data, get_split_df
from src.model import get_encoder, get_model, get_tokenizer
from src.subpages import Context
from src.utils import align_sample, device, explode_df

_TOKENIZER_NAME = (
    "xlm-roberta-base",
    "gagan3012/bert-tiny-finetuned-ner",
    "distilbert-base-german-cased",
)[0]


def _load_models_and_tokenizer(
    encoder_model_name: str,
    model_name: str,
    tokenizer_name: Optional[str],
    device: str = "cpu",
):
    sentence_encoder = get_encoder(encoder_model_name, device=device)
    tokenizer = get_tokenizer(tokenizer_name if tokenizer_name else model_name)
    labels = "O B-COMMA".split() if "comma" in model_name else None
    model = get_model(model_name, labels=labels)
    return sentence_encoder, model, tokenizer


@st.cache(allow_output_mutation=True)
def load_context(
    encoder_model_name: str,
    model_name: str,
    ds_name: str,
    ds_config_name: str,
    ds_split_name: str,
    split_sample_size: int,
    **kw_args,
) -> Context:
    """Utility method loading (almost) everything we need for the application.
    This exists just because we want to cache the results of this function.

    Args:
        encoder_model_name (str): Name of the sentence encoder to load.
        model_name (str): Name of the NER model to load.
        ds_name (str): Dataset name or path.
        ds_config_name (str): Dataset config name.
        ds_split_name (str): Dataset split name.
        split_sample_size (int): Number of examples to load from the split.

    Returns:
        Context: An object containing everything we need for the application.
    """

    sentence_encoder, model, tokenizer = _load_models_and_tokenizer(
        encoder_model_name=encoder_model_name,
        model_name=model_name,
        tokenizer_name=_TOKENIZER_NAME if "comma" in model_name else None,
        device=str(device),
    )
    collator = get_collator(tokenizer)

    # load data related stuff
    split: Dataset = get_data(ds_name, ds_config_name, ds_split_name, split_sample_size)
    tags = split.features["ner_tags"].feature
    split_encoded, word_ids, ids = encode_dataset(split, tokenizer)

    # transform into dataframe
    df = get_split_df(split_encoded, model, tokenizer, collator, tags)
    df["word_ids"] = word_ids
    df["ids"] = ids

    # explode, clean, merge
    df_tokens = explode_df(df)
    df_tokens_cleaned = df_tokens.query("labels != 'IGN'")
    df_merged = pd.DataFrame(df.apply(align_sample, axis=1).tolist())
    df_tokens_merged = explode_df(df_merged)

    return Context(
        **{
            "model": model,
            "tokenizer": tokenizer,
            "sentence_encoder": sentence_encoder,
            "df": df,
            "df_tokens": df_tokens,
            "df_tokens_cleaned": df_tokens_cleaned,
            "df_tokens_merged": df_tokens_merged,
            "tags": tags,
            "labels": tags.names,
            "split_sample_size": split_sample_size,
            "ds_name": ds_name,
            "ds_config_name": ds_config_name,
            "ds_split_name": ds_split_name,
            "split": split,
        }
    )
