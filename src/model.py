import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForTokenClassification  # type: ignore
from transformers import AutoTokenizer  # type: ignore


@st.experimental_singleton()
def get_model(model_name: str, labels=None):
    if labels is None:
        return AutoModelForTokenClassification.from_pretrained(
            model_name,
            output_attentions=True,
        )  # type: ignore
    else:
        id2label = {idx: tag for idx, tag in enumerate(labels)}
        label2id = {tag: idx for idx, tag in enumerate(labels)}
        return AutoModelForTokenClassification.from_pretrained(
            model_name,
            output_attentions=True,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
        )  # type: ignore


@st.experimental_singleton()
def get_encoder(model_name: str, device: str = "cpu"):
    return SentenceTransformer(model_name, device=device)


@st.experimental_singleton()
def get_tokenizer(tokenizer_name: str):
    return AutoTokenizer.from_pretrained(tokenizer_name)
