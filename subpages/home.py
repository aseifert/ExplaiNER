import json
import random
from typing import Optional

import streamlit as st

from data import get_data
from subpages.page import Context, Page
from utils import classmap, color_map_color

_SENTENCE_ENCODER_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)[0]
_MODEL_NAME = (
    "elastic/distilbert-base-uncased-finetuned-conll03-english",
    "gagan3012/bert-tiny-finetuned-ner",
    "socialmediaie/bertweet-base_wnut17_ner",
    "sberbank-ai/bert-base-NER-reptile-5-datasets",
    "aseifert/comma-xlm-roberta-base",
    "dslim/bert-base-NER",
    "aseifert/distilbert-base-german-cased-comma-derstandard",
)[0]
_DATASET_NAME = (
    "conll2003",
    "wnut_17",
    "aseifert/comma",
)[0]
_CONFIG_NAME = (
    "conll2003",
    "wnut_17",
    "seifertverlag",
)[0]


class HomePage(Page):
    name = "Home / Setup"
    icon = "house"

    def get_widget_defaults(self):
        return {
            "encoder_model_name": _SENTENCE_ENCODER_MODEL,
            "model_name": _MODEL_NAME,
            "ds_name": _DATASET_NAME,
            "ds_split_name": "validation",
            "ds_config_name": _CONFIG_NAME,
            "split_sample_size": 512,
        }

    def render(self, context: Optional[Context] = None):
        st.title("ExplaiNER")

        with st.expander("💡", expanded=True):
            st.write(
                "**Error Analysis is an important but often overlooked part of the data science project lifecycle**, for which there is still very little tooling available. Practitioners tend to write throwaway code or, worse, skip this crucial step of understanding their models' errors altogether. This project tries to provide an **extensive toolkit to probe any NER model/dataset combination**, find labeling errors and understand the models' and datasets' limitations, leading the user on her way to further **improving both model AND dataset**."
            )
            st.write(
                "**Note:** This Space requires a fair amount of computation, so please be patient with the loading animations. 🙏 I am caching as much as possible, so after the first wait most things should be precomputed."
            )
            st.write(
                "_Caveat: Even though everything is customizable here, I haven't tested this app much with different models/datasets._"
            )

        col1, _, col2a, col2b = st.columns([0.8, 0.05, 0.15, 0.15])

        with col1:
            random_form_key = f"settings-{random.randint(0, 100000)}"
            # FIXME: for some reason I'm getting the following error if I don't randomize the key:
            """
                2022-05-05 20:37:16.507 Traceback (most recent call last):
            File "/Users/zoro/mambaforge/lib/python3.9/site-packages/streamlit/scriptrunner/script_runner.py", line 443, in _run_script
                exec(code, module.__dict__)
            File "/Users/zoro/code/error-analysis/main.py", line 162, in <module>
                main()
            File "/Users/zoro/code/error-analysis/main.py", line 102, in main
                show_setup()
            File "/Users/zoro/code/error-analysis/section/setup.py", line 68, in show_setup
                st.form_submit_button("Load Model & Data")
            File "/Users/zoro/mambaforge/lib/python3.9/site-packages/streamlit/elements/form.py", line 240, in form_submit_button
                return self._form_submit_button(
            File "/Users/zoro/mambaforge/lib/python3.9/site-packages/streamlit/elements/form.py", line 260, in _form_submit_button
                return self.dg._button(
            File "/Users/zoro/mambaforge/lib/python3.9/site-packages/streamlit/elements/button.py", line 304, in _button
                check_session_state_rules(default_value=None, key=key, writes_allowed=False)
            File "/Users/zoro/mambaforge/lib/python3.9/site-packages/streamlit/elements/utils.py", line 74, in check_session_state_rules
                raise StreamlitAPIException(
            streamlit.errors.StreamlitAPIException: Values for st.button, st.download_button, st.file_uploader, and st.form cannot be set using st.session_state.
            """
            with st.form(key=random_form_key):
                st.subheader("Model & Data Selection")
                st.text_input(
                    label="NER Model:",
                    key="model_name",
                    help="Path or name of the model to use",
                )
                st.text_input(
                    label="Encoder Model:",
                    key="encoder_model_name",
                    help="Path or name of the encoder to use for duplicate detection",
                )
                ds_name = st.text_input(
                    label="Dataset:",
                    key="ds_name",
                    help="Path or name of the dataset to use",
                )
                ds_config_name = st.text_input(
                    label="Config (optional):",
                    key="ds_config_name",
                )
                ds_split_name = st.selectbox(
                    label="Split:",
                    options=["train", "validation", "test"],
                    key="ds_split_name",
                )
                split_sample_size = st.number_input(
                    "Sample size:",
                    step=16,
                    key="split_sample_size",
                    help="Sample size for the split, speeds up processing inside streamlit",
                )
                # breakpoint()
                # st.form_submit_button("Submit")
                st.form_submit_button("Load Model & Data")

        split = get_data(ds_name, ds_config_name, ds_split_name, split_sample_size)
        labels = list(
            set([n.split("-")[1] for n in split.features["ner_tags"].feature.names if n != "O"])
        )

        with col2a:
            st.subheader("Classes")
            st.write("**Color**")
            colors = {label: color_map_color(i / len(labels)) for i, label in enumerate(labels)}
            for label in labels:
                if f"color_{label}" not in st.session_state:
                    st.session_state[f"color_{label}"] = colors[label]
                st.color_picker(label, key=f"color_{label}")
        with col2b:
            st.subheader("—")
            st.write("**Icon**")
            emojis = list(json.load(open("subpages/emoji-en-US.json")).keys())
            for label in labels:
                if f"icon_{label}" not in st.session_state:
                    st.session_state[f"icon_{label}"] = classmap[label]
                st.selectbox(label, key=f"icon_{label}", options=emojis)
                classmap[label] = st.session_state[f"icon_{label}"]

        # if st.button("Reset to defaults"):
        #     st.session_state.update(**get_home_page_defaults())
        #     # time.sleep 2 secs
        #     import time
        #     time.sleep(1)

        #     # st.legacy_caching.clear_cache()
        #     st.experimental_rerun()
