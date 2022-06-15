"""
The metrics page contains precision, recall and f-score metrics as well as a confusion matrix over all the classes. By default, the confusion matrix is normalized. There's an option to zero out the diagonal, leaving only prediction errors (here it makes sense to turn off normalization, so you get raw error counts).
"""
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from seqeval.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from src.subpages.page import Context, Page


def _get_evaluation(df):
    y_true = df.apply(lambda row: [lbl for lbl in row.labels if lbl != "IGN"], axis=1)
    y_pred = df.apply(
        lambda row: [pred for (pred, lbl) in zip(row.preds, row.labels) if lbl != "IGN"],
        axis=1,
    )
    report: str = classification_report(y_true, y_pred, scheme="IOB2", digits=3)  # type: ignore
    return report.replace(
        "precision    recall  f1-score   support",
        "=" * 12 + "  precision    recall  f1-score   support",
    )


def plot_confusion_matrix(y_true, y_preds, labels, normalize=None, zero_diagonal=True):
    cm = confusion_matrix(y_true, y_preds, normalize=normalize, labels=labels)
    if zero_diagonal:
        np.fill_diagonal(cm, 0)

    # st.write(plt.rcParams["font.size"])
    # plt.rcParams.update({'font.size': 10.0})
    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fmt = "d" if normalize is None else ".3f"
    disp.plot(
        cmap="Blues",
        include_values=True,
        xticks_rotation="vertical",
        values_format=fmt,
        ax=ax,
        colorbar=False,
    )
    return fig


class MetricsPage(Page):
    name = "Metrics"
    icon = "graph-up-arrow"

    def get_widget_defaults(self):
        return {
            "normalize": True,
            "zero_diagonal": False,
        }

    def render(self, context: Context):
        st.title(self.name)
        with st.expander("💡", expanded=True):
            st.write(
                "The metrics page contains precision, recall and f-score metrics as well as a confusion matrix over all the classes. By default, the confusion matrix is normalized. There's an option to zero out the diagonal, leaving only prediction errors (here it makes sense to turn off normalization, so you get raw error counts)."
            )
            st.write(
                "With the confusion matrix, you don't want any of the classes to end up in the bottom right quarter: those are frequent but error-prone."
            )

        eval_results = _get_evaluation(context.df)
        if len(eval_results.splitlines()) < 8:
            col1, _, col2 = st.columns([8, 1, 1])
        else:
            col1 = col2 = st

        col1.subheader("🎯 Evaluation Results")
        col1.code(eval_results)

        results = [re.split(r" +", l.lstrip()) for l in eval_results.splitlines()[2:-4]]
        data = [(r[0], int(r[-1]), float(r[-2])) for r in results]
        df = pd.DataFrame(data, columns="class support f1".split())
        fig = px.scatter(
            df,
            x="support",
            y="f1",
            range_y=(0, 1.05),
            color="class",
        )
        # fig.update_layout(title_text="asdf", title_yanchor="bottom")
        col1.plotly_chart(fig)

        col2.subheader("🔠 Confusion Matrix")
        normalize = None if not col2.checkbox("Normalize", key="normalize") else "true"
        zero_diagonal = col2.checkbox("Zero Diagonal", key="zero_diagonal")
        col2.pyplot(
            plot_confusion_matrix(
                y_true=context.df_tokens_cleaned["labels"],
                y_preds=context.df_tokens_cleaned["preds"],
                labels=context.labels,
                normalize=normalize,
                zero_diagonal=zero_diagonal,
            ),
        )
