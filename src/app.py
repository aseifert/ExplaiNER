"""The App module is the main entry point for the application.

    Run `streamlit run app.py` to start the app.
"""

import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

from src.load import load_context
from src.subpages import (
    DebugPage,
    FindDuplicatesPage,
    HomePage,
    LossesPage,
    LossySamplesPage,
    MetricsPage,
    MisclassifiedPage,
    Page,
    ProbingPage,
    RandomSamplesPage,
    RawDataPage,
)
from src.subpages.attention import AttentionPage
from src.subpages.hidden_states import HiddenStatesPage
from src.subpages.inspect import InspectPage
from src.utils import classmap

sts = st.sidebar
st.set_page_config(
    layout="wide",
    page_title="Error Analysis",
    page_icon="🏷️",
)


def _show_menu(pages: list[Page]) -> int:
    with st.sidebar:
        page_names = [p.name for p in pages]
        page_icons = [p.icon for p in pages]
        selected_menu_item = st.session_state.active_page = option_menu(
            menu_title="ExplaiNER",
            options=page_names,
            icons=page_icons,
            menu_icon="layout-wtf",
            default_index=0,
        )
        return page_names.index(selected_menu_item)
    assert False


def _initialize_session_state(pages: list[Page]):
    if "active_page" not in st.session_state:
        for page in pages:
            st.session_state.update(**page._get_widget_defaults())
    st.session_state.update(st.session_state)


def _write_color_legend(context):
    def style(x):
        return [f"background-color: {rgb}; opacity: 1;" for rgb in colors]

    labels = list(set([lbl.split("-")[1] if "-" in lbl else lbl for lbl in context.labels]))
    colors = [st.session_state.get(f"color_{lbl}", "#000000") for lbl in labels]

    color_legend_df = pd.DataFrame(
        [classmap[l] for l in labels], columns=["label"], index=labels
    ).T
    st.sidebar.write(
        color_legend_df.T.style.apply(style, axis=0).set_properties(
            **{"color": "white", "text-align": "center"}
        )
    )


def main():
    """The main entry point for the application."""
    pages: list[Page] = [
        HomePage(),
        AttentionPage(),
        HiddenStatesPage(),
        ProbingPage(),
        MetricsPage(),
        LossySamplesPage(),
        LossesPage(),
        MisclassifiedPage(),
        RandomSamplesPage(),
        FindDuplicatesPage(),
        InspectPage(),
        RawDataPage(),
        DebugPage(),
    ]

    _initialize_session_state(pages)

    selected_page_idx = _show_menu(pages)
    selected_page = pages[selected_page_idx]

    if isinstance(selected_page, HomePage):
        selected_page.render()
        return

    if "model_name" not in st.session_state:
        # this can happen if someone loads another page directly (without going through home)
        st.error("Setup not complete. Please click on 'Home / Setup in left menu bar'")
        return

    context = load_context(**st.session_state)
    _write_color_legend(context)
    selected_page.render(context)


if __name__ == "__main__":
    main()
