import streamlit as st
from datasets import Dataset

from src.subpages.page import Context, Page  # type: ignore
from src.utils import device, explode_df, htmlify_labeled_example, tag_text


class FaissPage(Page):
    name = "Bla"
    icon = "x-octagon"

    def render(self, context: Context):
        dd = Dataset.from_pandas(context.df_tokens_merged, preserve_index=False)  # type: ignore

        dd.add_faiss_index(column="hidden_states", index_name="token_index")
        token_id, text = (
            6,
            "Die Wissenschaft ist eine wichtige Grundlage für die Entwicklung von neuen Technologien.",
        )
        token_id, text = (
            15,
            "Außer der unbewussten Beeinflussung eines Resultats gibt es auch noch andere Motive die das reine strahlende Licht der Wissenschaft etwas zu trüben vermögen.",
        )
        token_id, text = (
            3,
            "Mit mehr Instrumenten einer besseren präziseren Datenbasis ist auch ein viel besseres smarteres Risikomanagement möglich.",
        )
        token_id, text = (
            7,
            "Es gilt die akademische Viertelstunde das heißt Beginn ist fünfzehn Minuten später.",
        )
        token_id, text = (
            7,
            "Damit einher geht übrigens auch dass Marcella Collocinis Tochter keine wie auch immer geartete strafrechtliche Verfolgung zu befürchten hat.",
        )
        token_id, text = (
            16,
            "After Steve Jobs met with Bill Gates of Microsoft back in 1993, they went to Cupertino and made the deal.",
        )

        tagged = tag_text(text, context.tokenizer, context.model, device)
        hidden_states = tagged["hidden_states"]
        # tagged.drop("hidden_states", inplace=True, axis=1)
        # hidden_states_vec = svd.transform([hidden_states[token_id]])[0].astype(np.float32)
        hidden_states_vec = hidden_states[token_id]
        tagged = tagged.astype(str)
        tagged["probs"] = tagged["probs"].apply(lambda x: x[:-2])
        tagged["check"] = tagged["probs"].apply(
            lambda x: "✅ ✅" if int(x) < 100 else "✅" if int(x) < 1000 else ""
        )
        st.dataframe(tagged.drop("hidden_states", axis=1).T)
        results = dd.get_nearest_examples("token_index", hidden_states_vec, k=10)
        for i, (dist, idx, token) in enumerate(
            zip(results.scores, results.examples["ids"], results.examples["tokens"])
        ):
            st.code(f"{dist:.3f} {token}")
            sample = context.df_tokens_merged.query(f"ids == '{idx}'")
            st.write(f"[{i};{idx}] " + htmlify_labeled_example(sample), unsafe_allow_html=True)
