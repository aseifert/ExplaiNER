from functools import partial

import pandas as pd
import streamlit as st
import torch
from datasets import Dataset, DatasetDict, load_dataset  # type: ignore
from torch.nn.functional import cross_entropy
from transformers import DataCollatorForTokenClassification  # type: ignore

from src.utils import device, tokenizer_hash_funcs


@st.cache(allow_output_mutation=True)
def get_data(ds_name: str, config_name: str, split_name: str, split_sample_size: int) -> Dataset:
    """Loads a Dataset from the HuggingFace hub (if not already loaded).

    Uses `datasets.load_dataset` to load the dataset (see its documentation for additional details).

    Args:
        ds_name (str): Path or name of the dataset.
        config_name (str): Name of the dataset configuration.
        split_name (str): Which split of the data to load.
        split_sample_size (int): The number of examples to load from the split.

    Returns:
        Dataset: A Dataset object.
    """
    ds: DatasetDict = load_dataset(ds_name, name=config_name, use_auth_token=True).shuffle(seed=0)  # type: ignore
    split = ds[split_name].select(range(split_sample_size))
    return split


@st.cache(
    allow_output_mutation=True,
    hash_funcs=tokenizer_hash_funcs,
)
def get_collator(tokenizer) -> DataCollatorForTokenClassification:
    """Returns a DataCollator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([PreTrainedTokenizer] or [PreTrainedTokenizerFast]): The tokenizer used for encoding the data.

    Returns:
        DataCollatorForTokenClassification: The DataCollatorForTokenClassification object.
    """
    return DataCollatorForTokenClassification(tokenizer)


def create_word_ids_from_input_ids(tokenizer, input_ids: list[int]) -> list[int]:
    """Takes a list of input_ids and return corresponding word_ids

    Args:
        tokenizer: The tokenizer that was used to obtain the input ids.
        input_ids (list[int]): List of token ids.

    Returns:
        list[int]: Word ids corresponding to the input ids.
    """
    word_ids = []
    wid = -1
    tokens = [tokenizer.convert_ids_to_tokens(i) for i in input_ids]

    for i, tok in enumerate(tokens):
        if tok in tokenizer.all_special_tokens:
            word_ids.append(-1)
            continue

        if not tokens[i - 1].endswith("@@") and tokens[i - 1] != "<unk>":
            wid += 1

        word_ids.append(wid)

    assert len(word_ids) == len(input_ids)
    return word_ids


def tokenize(batch, tokenizer) -> dict:
    """Tokenizes a batch of examples.

    Args:
        batch: The examples to tokenize
        tokenizer: The tokenizer to use

    Returns:
        dict: The tokenized batch
    """
    tokenized_inputs = tokenizer(batch["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    wids = []

    for idx, label in enumerate(batch["ner_tags"]):
        try:
            word_ids = tokenized_inputs.word_ids(batch_index=idx)
        except ValueError:
            word_ids = create_word_ids_from_input_ids(
                tokenizer, tokenized_inputs["input_ids"][idx]
            )
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx == -1 or word_idx is None or word_idx == previous_word_idx:
                label_ids.append(-100)
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx
        wids.append(word_ids)
        labels.append(label_ids)
    tokenized_inputs["word_ids"] = wids
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def stringify_ner_tags(batch, tags):
    """Stringifies a dataset batch's NER tags.

    Args:
        batch (_type_): _description_
        tags (_type_): _description_

    Returns:
        _type_: _description_
    """
    return {"ner_tags_str": [tags.int2str(idx) for idx in batch["ner_tags"]]}


def encode_dataset(split: Dataset, tokenizer):
    """Encodes a dataset split.

    Args:
        split (Dataset): A Dataset object.
        tokenizer: A PreTrainedTokenizer object.

    Returns:
        Dataset: A Dataset object with the encoded inputs.
    """

    tags = split.features["ner_tags"].feature
    split = split.map(partial(stringify_ner_tags, tags=tags), batched=True)
    remove_columns = split.column_names
    ids = split["id"]
    split = split.map(
        partial(tokenize, tokenizer=tokenizer),
        batched=True,
        remove_columns=remove_columns,
    )
    word_ids = [[id if id is not None else -1 for id in wids] for wids in split["word_ids"]]
    return split.remove_columns(["word_ids"]), word_ids, ids


def forward_pass_with_label(batch, model, collator, num_classes: int) -> dict:
    """Runs the forward pass for a batch of examples.

    Args:
        batch: The batch to process
        model: The model to process the batch with
        collator: A data collator
        num_classes (int): Number of classes

    Returns:
        dict: a dictionary containing `losses`, `preds` and `hidden_states`
    """

    # Convert dict of lists to list of dicts suitable for data collator
    features = [dict(zip(batch, t)) for t in zip(*batch.values())]

    # Pad inputs and labels and put all tensors on device
    batch = collator(features)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    with torch.no_grad():
        # Pass data through model
        output = model(input_ids, attention_mask, output_hidden_states=True)
        # logit.size: [batch_size, sequence_length, classes]

        # Predict class with largest logit value on classes axis
        preds = torch.argmax(output.logits, axis=-1).cpu().numpy()  # type: ignore

        # Calculate loss per token after flattening batch dimension with view
        loss = cross_entropy(
            output.logits.view(-1, num_classes), labels.view(-1), reduction="none"
        )

        # Unflatten batch dimension and convert to numpy array
        loss = loss.view(len(input_ids), -1).cpu().numpy()
        hidden_states = output.hidden_states[-1].cpu().numpy()

        # logits = output.logits.view(len(input_ids), -1).cpu().numpy()

    return {"losses": loss, "preds": preds, "hidden_states": hidden_states}


def predict(split_encoded: Dataset, model, tokenizer, collator, tags) -> pd.DataFrame:
    """Generates predictions for a given dataset split and returns the results as a dataframe.

    Args:
        split_encoded (Dataset): The dataset to process
        model: The model to process the dataset with
        tokenizer: The tokenizer to process the dataset with
        collator: The data collator to use
        tags: The tags used in the dataset

    Returns:
        pd.DataFrame: A dataframe containing token-level predictions.
    """

    split_encoded = split_encoded.map(
        partial(
            forward_pass_with_label,
            model=model,
            collator=collator,
            num_classes=tags.num_classes,
        ),
        batched=True,
        batch_size=8,
    )
    df: pd.DataFrame = split_encoded.to_pandas()  # type: ignore

    df["tokens"] = df["input_ids"].apply(
        lambda x: tokenizer.convert_ids_to_tokens(x)  # type: ignore
    )
    df["labels"] = df["labels"].apply(
        lambda x: ["IGN" if i == -100 else tags.int2str(int(i)) for i in x]
    )
    df["preds"] = df["preds"].apply(lambda x: [model.config.id2label[i] for i in x])
    df["preds"] = df.apply(lambda x: x["preds"][: len(x["input_ids"])], axis=1)
    df["losses"] = df.apply(lambda x: x["losses"][: len(x["input_ids"])], axis=1)
    df["hidden_states"] = df.apply(lambda x: x["hidden_states"][: len(x["input_ids"])], axis=1)
    df["total_loss"] = df["losses"].apply(sum)

    return df
