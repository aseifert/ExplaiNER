from dataclasses import dataclass
from typing import Any

import pandas as pd
from datasets import Dataset  # type: ignore
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification  # type: ignore
from transformers import AutoTokenizer  # type: ignore


@dataclass
class Context:
    model: AutoModelForSequenceClassification
    tokenizer: AutoTokenizer
    sentence_encoder: SentenceTransformer
    tags: Any
    df: pd.DataFrame
    df_tokens: pd.DataFrame
    df_tokens_cleaned: pd.DataFrame
    df_tokens_merged: pd.DataFrame
    split_sample_size: int
    ds_name: str
    ds_config_name: str
    ds_split_name: str
    split: Dataset
    labels: list[str]


class Page:
    name: str
    icon: str

    def get_widget_defaults(self):
        return {}

    def render(self, context):
        ...
