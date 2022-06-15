"""Show count, mean and median loss per token and label."""
import streamlit as st

from src.subpages.page import Context, Page
from src.utils import AgGrid, aggrid_interactive_table


@st.cache
def get_loss_by_token(df_tokens):
    return (
        df_tokens.groupby("tokens")[["losses"]]
        .agg(["count", "mean", "median", "sum"])
        .droplevel(level=0, axis=1)  # Get rid of multi-level columns
        .sort_values(by="sum", ascending=False)
        .reset_index()
    )


@st.cache
def get_loss_by_label(df_tokens):
    return (
        df_tokens.groupby("labels")[["losses"]]
        .agg(["count", "mean", "median", "sum"])
        .droplevel(level=0, axis=1)
        .sort_values(by="mean", ascending=False)
        .reset_index()
    )


class LossesPage(Page):
    name = "Loss by Token/Label"
    icon = "sort-alpha-down"

    def render(self, context: Context):
        st.title(self.name)
        with st.expander("💡", expanded=True):
            st.write("Show count, mean and median loss per token and label.")
            st.write(
                "Look out for tokens that have a big gap between mean and median, indicating systematic labeling issues."
            )

        col1, _, col2 = st.columns([8, 1, 6])

        with col1:
            st.subheader("💬 Loss by Token")

            st.session_state["_merge_tokens"] = st.checkbox(
                "Merge tokens", value=True, key="merge_tokens"
            )
            loss_by_token = (
                get_loss_by_token(context.df_tokens_merged)
                if st.session_state["merge_tokens"]
                else get_loss_by_token(context.df_tokens_cleaned)
            )
            aggrid_interactive_table(loss_by_token.round(3))
            # st.subheader("🏷️ Loss by Label")
            # loss_by_label = get_loss_by_label(df_tokens_cleaned)
            # st.dataframe(loss_by_label)

            st.write(
                "_Caveat: Even though tokens have contextual representations, we average them to get these summary statistics._"
            )

        with col2:
            st.subheader("🏷️ Loss by Label")
            loss_by_label = get_loss_by_label(context.df_tokens_cleaned)
            AgGrid(loss_by_label.round(3), height=200)
