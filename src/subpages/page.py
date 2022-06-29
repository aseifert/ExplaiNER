from dataclasses import dataclass
from typing import Any

import pandas as pd
from datasets import Dataset  # type: ignore
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification  # type: ignore
from transformers import AutoTokenizer  # type: ignore


@dataclass
class Context:
    """This object facilitates passing around the application's state between different pages."""

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
    """This class encapsulates the logic for a single page of the application."""

    name: str
    """The page's name that will be used in the sidebar menu."""

    icon: str
    """The page's icon that will be used in the sidebar menu."""

    def _get_widget_defaults(self):
        """This function holds the default settings for all widgets contained on this page.

        Returns:
            dict: A dictionary of widget defaults, where the keys are the widget names and the values are the default.
        """
        return {}

    def render(self, context):
        """This function renders the page."""
        ...
