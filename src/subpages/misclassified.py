"""This page contains all misclassified examples and allows filtering by specific error types."""
from collections import defaultdict

import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix

from src.subpages.page import Context, Page
from src.utils import htmlify_labeled_example


class MisclassifiedPage(Page):
    name = "Misclassified"
    icon = "x-octagon"

    def render(self, context: Context):
        st.title(self.name)
        with st.expander("💡", expanded=True):
            st.write(
                "This page contains all misclassified examples and allows filtering by specific error types."
            )

        misclassified_indices = context.df_tokens_merged.query("labels != preds").index.unique()
        misclassified_samples = context.df_tokens_merged.loc[misclassified_indices]
        cm = confusion_matrix(
            misclassified_samples.labels,
            misclassified_samples.preds,
            labels=context.labels,
        )

        # st.pyplot(
        #     plot_confusion_matrix(
        #         y_preds=misclassified_samples["preds"],
        #         y_true=misclassified_samples["labels"],
        #         labels=labels,
        #         normalize=None,
        #         zero_diagonal=True,
        #     ),
        # )
        df = pd.DataFrame(cm, index=context.labels, columns=context.labels).astype(str)
        import numpy as np

        np.fill_diagonal(df.values, "")
        st.dataframe(df.applymap(lambda x: x if x != "0" else ""))
        # import matplotlib.pyplot as plt
        # st.pyplot(df.style.background_gradient(cmap='RdYlGn_r').to_html())
        # selection = aggrid_interactive_table(df)

        # st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)

        confusions = defaultdict(int)
        for i, row in enumerate(cm):
            for j, _ in enumerate(row):
                if i == j or cm[i][j] == 0:
                    continue
                confusions[(context.labels[i], context.labels[j])] += cm[i][j]

        def format_func(item):
            return (
                f"true: {item[0][0]} <> pred: {item[0][1]} ||| count: {item[1]}" if item else "All"
            )

        conf = st.radio(
            "Filter by Class Confusion",
            options=list(zip(confusions.keys(), confusions.values())),
            format_func=format_func,
        )

        # st.write(
        #     f"**Filtering Examples:** True class: `{conf[0][0]}`, Predicted class: `{conf[0][1]}`"
        # )

        filtered_indices = misclassified_samples.query(
            f"labels == '{conf[0][0]}' and preds == '{conf[0][1]}'"
        ).index
        for i, idx in enumerate(filtered_indices):
            sample = context.df_tokens_merged.loc[idx]
            st.write(
                htmlify_labeled_example(sample),
                unsafe_allow_html=True,
            )
            st.write("---")
