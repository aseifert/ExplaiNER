import pandas as pd
import streamlit as st

from subpages.page import Context, Page
from utils import htmlify_labeled_example


class RandomSamplesPage(Page):
    name = "Random Samples"
    icon = "shuffle"

    def get_widget_defaults(self):
        return {
            "random_sample_size_min": 128,
        }

    def render(self, context: Context):
        st.title("🎲 Random Samples")
        with st.expander("💡", expanded=True):
            st.write(
                "Show random samples. Simple idea, but often it turns up some interesting things."
            )

        random_sample_size = st.number_input(
            "Random sample size:",
            value=min(st.session_state.random_sample_size_min, context.split_sample_size),
            step=16,
            key="random_sample_size",
        )

        if st.button("🎲 Resample"):
            st.experimental_rerun()

        random_indices = context.df.sample(int(random_sample_size)).index
        samples = context.df_tokens_merged.loc[random_indices]
        return

        for i, idx in enumerate(random_indices):
            sample = samples.loc[idx]

            if isinstance(sample, pd.Series):
                continue

            col1, _, col2 = st.columns([0.08, 0.025, 0.8])

            counter = f"<span title='#sample | index' style='display: block; background-color: black; opacity: 1; color: wh^; padding: 0 5px'>[{i+1} | {idx}]</span>"
            loss = f"<span title='total loss' style='display: block; background-color: yellow; color: gray; padding: 0 5px;'>𝐿 {sample.losses.sum():.3f}</span>"
            col1.write(f"{counter}{loss}", unsafe_allow_html=True)
            col1.write("")
            st.write(sample.astype(str))
            col2.write(htmlify_labeled_example(sample), unsafe_allow_html=True)
