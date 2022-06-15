"""See the data as seen by your model."""
import pandas as pd
import streamlit as st

from src.subpages.page import Context, Page
from src.utils import aggrid_interactive_table


@st.cache
def convert_df(df):
    return df.to_csv().encode("utf-8")


class RawDataPage(Page):
    name = "Raw data"
    icon = "qr-code"

    def render(self, context: Context):
        st.title(self.name)
        with st.expander("💡", expanded=True):
            st.write("See the data as seen by your model.")

        st.subheader("Dataset")
        st.code(
            f"Dataset: {context.ds_name}\nConfig: {context.ds_config_name}\nSplit: {context.ds_split_name}"
        )

        st.write("**Data after processing and inference**")

        processed_df = (
            context.df_tokens.drop("hidden_states", axis=1).drop("attention_mask", axis=1).round(3)
        )
        cols = (
            "ids input_ids token_type_ids word_ids losses tokens labels preds total_loss".split()
        )
        if "token_type_ids" not in processed_df.columns:
            cols.remove("token_type_ids")
        processed_df = processed_df[cols]
        aggrid_interactive_table(processed_df)
        processed_df_csv = convert_df(processed_df)
        st.download_button(
            "Download csv",
            processed_df_csv,
            "processed_data.csv",
            "text/csv",
        )

        st.write("**Raw data (exploded by tokens)**")
        raw_data_df = context.split.to_pandas().apply(pd.Series.explode)  # type: ignore
        aggrid_interactive_table(raw_data_df)
        raw_data_df_csv = convert_df(raw_data_df)
        st.download_button(
            "Download csv",
            raw_data_df_csv,
            "raw_data.csv",
            "text/csv",
        )
