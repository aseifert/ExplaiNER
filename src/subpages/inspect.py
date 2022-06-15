import streamlit as st

from src.subpages.page import Context, Page
from src.utils import aggrid_interactive_table, colorize_classes


class InspectPage(Page):
    name = "Inspect"
    icon = "search"

    def render(self, context: Context):
        st.title(self.name)
        with st.expander("💡", expanded=True):
            st.write("Inspect your whole dataset, either unfiltered or by id.")

        df = context.df_tokens
        cols = (
            "ids input_ids token_type_ids word_ids losses tokens labels preds total_loss".split()
        )
        if "token_type_ids" not in df.columns:
            cols.remove("token_type_ids")
        df = df.drop("hidden_states", axis=1).drop("attention_mask", axis=1)[cols]

        if st.checkbox("Filter by id", value=True):
            ids = list(sorted(map(int, df.ids.unique())))
            next_id = st.session_state.get("next_id", 0)

            example_id = st.selectbox("Select an example", ids, index=next_id)
            df = df[df.ids == str(example_id)][1:-1]
            # st.dataframe(colorize_classes(df).format(precision=3).bar(subset="losses"))  # type: ignore
            st.dataframe(colorize_classes(df.round(3).astype(str)))

            if st.button("Next example"):
                st.session_state.next_id = (ids.index(example_id) + 1) % len(ids)
            if st.button("Previous example"):
                st.session_state.next_id = (ids.index(example_id) - 1) % len(ids)
        else:
            aggrid_interactive_table(df.round(3))
