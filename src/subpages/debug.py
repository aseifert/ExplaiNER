import streamlit as st
from pip._internal.operations import freeze

from subpages.page import Context, Page


class DebugPage(Page):
    name = "Debug"
    icon = "bug"

    def render(self, context: Context):
        st.title(self.name)
        # with st.expander("💡", expanded=True):
        #     st.write("Some debug info.")

        st.subheader("Installed Packages")
        # get output of pip freeze from system
        with st.expander("pip freeze"):
            st.code("\n".join(freeze.freeze()))

        st.subheader("Streamlit Session State")
        st.json(st.session_state)
        st.subheader("Tokenizer")
        st.code(context.tokenizer)
        st.subheader("Model")
        st.code(context.model.config)
        st.code(context.model)
