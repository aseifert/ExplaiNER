"""Show every example sorted by loss (descending) for close inspection."""
import pandas as pd
import streamlit as st

from src.subpages.page import Context, Page
from src.utils import (
    colorize_classes,
    get_bg_color,
    get_fg_color,
    htmlify_labeled_example,
)


class LossySamplesPage(Page):
    name = "Samples by Loss"
    icon = "sort-numeric-down-alt"

    def get_widget_defaults(self):
        return {
            "skip_correct": True,
            "samples_by_loss_show_df": True,
        }

    def render(self, context: Context):
        st.title(self.name)
        with st.expander("💡", expanded=True):
            st.write("Show every example sorted by loss (descending) for close inspection.")
            st.write(
                "The **dataframe** is mostly self-explanatory. The cells are color-coded by label, a lighter color signifies a continuation label. Cells in the loss row are filled red from left to right relative to the top loss."
            )
            st.write(
                "The **numbers to the left**: Top (black background) are sample number (listed here) and sample index (from the dataset). Below on yellow background is the total loss for the given sample."
            )
            st.write(
                "The **annotated sample**: Every predicted entity (every token, really) gets a black border. The text color signifies the predicted label, with the first token of a sequence of token also showing the label's icon. If (and only if) the prediction is wrong, a small little box after the entity (token) contains the correct target class, with a background color corresponding to that class."
            )

        st.subheader("💥 Samples ⬇loss")
        skip_correct = st.checkbox("Skip correct examples", value=True, key="skip_correct")
        show_df = st.checkbox("Show dataframes", key="samples_by_loss_show_df")

        st.write(
            """<style>
thead {
    display: none;
}
td {
    white-space: nowrap;
    padding: 0 5px !important;
}
</style>""",
            unsafe_allow_html=True,
        )

        top_indices = (
            context.df.sort_values(by="total_loss", ascending=False)
            .query("total_loss > 0.5")
            .index
        )

        cnt = 0
        for idx in top_indices:
            sample = context.df_tokens_merged.loc[idx]

            if isinstance(sample, pd.Series):
                continue

            if skip_correct and sum(sample.labels != sample.preds) == 0:
                continue

            if show_df:

                def colorize_col(col):
                    if col.name == "labels" or col.name == "preds":
                        bgs = []
                        fgs = []
                        ops = []
                        for v in col.values:
                            bgs.append(get_bg_color(v.split("-")[1]) if "-" in v else "#ffffff")
                            fgs.append(get_fg_color(bgs[-1]))
                            ops.append("1" if v.split("-")[0] == "B" or v == "O" else "0.5")
                        return [
                            f"background-color: {bg}; color: {fg}; opacity: {op};"
                            for bg, fg, op in zip(bgs, fgs, ops)
                        ]
                    return [""] * len(col)

                df = sample.reset_index().drop(["index", "hidden_states", "ids"], axis=1).round(3)
                losses_slice = pd.IndexSlice["losses", :]
                # x = df.T.astype(str)
                # st.dataframe(x)
                # st.dataframe(x.loc[losses_slice])
                styler = (
                    df.T.style.apply(colorize_col, axis=1)
                    .bar(subset=losses_slice, axis=1)
                    .format(precision=3)
                )
                # styler.data = styler.data.astype(str)
                st.write(styler.to_html(), unsafe_allow_html=True)
                st.write("")
                # st.dataframe(colorize_classes(sample.drop("hidden_states", axis=1)))#.bar(subset='losses'))  # type: ignore
                # st.write(
                #     colorize_errors(sample.round(3).drop("hidden_states", axis=1).astype(str))
                # )

            col1, _, col2 = st.columns([3.5 / 32, 0.5 / 32, 28 / 32])

            cnt += 1
            counter = f"<span title='#sample | index' style='display: block; background-color: black; opacity: 1; color: white; padding: 0 5px'>[{cnt} | {idx}]</span>"
            loss = f"<span title='total loss' style='display: block; background-color: yellow; color: gray; padding: 0 5px;'>𝐿 {sample.losses.sum():.3f}</span>"
            col1.write(f"{counter}{loss}", unsafe_allow_html=True)
            col1.write("")

            col2.write(htmlify_labeled_example(sample), unsafe_allow_html=True)
            # st.write(f"[{i};{idx}] " + htmlify_corr_sample(sample), unsafe_allow_html=True)
