"""Find potential duplicates in the data using cosine similarity."""
import streamlit as st
from sentence_transformers.util import cos_sim

from src.subpages.page import Context, Page


@st.cache()
def get_sims(texts: list[str], sentence_encoder):
    embeddings = sentence_encoder.encode(texts, batch_size=8, convert_to_numpy=True)
    return cos_sim(embeddings, embeddings)


class FindDuplicatesPage(Page):
    name = "Find Duplicates"
    icon = "fingerprint"

    def _get_widget_defaults(self):
        return {
            "cutoff": 0.95,
        }

    def render(self, context: Context):
        st.title("Find Duplicates")
        with st.expander("💡", expanded=True):
            st.write("Find potential duplicates in the data using cosine similarity.")

        cutoff = st.slider("Similarity threshold", min_value=0.0, max_value=1.0, key="cutoff")
        # split.add_faiss_index(column="embeddings", index_name="sent_index")
        # st.write("Index is ready")
        # sentence_encoder.encode(["hello world"], batch_size=8)
        # st.write(split["tokens"][0])
        texts = [" ".join(ts) for ts in context.split["tokens"]]
        sims = get_sims(texts, context.sentence_encoder)

        candidates = []
        for i in range(len(sims)):
            for j in range(i + 1, len(sims)):
                if sims[i][j] >= cutoff:
                    candidates.append((sims[i][j], i, j))
        candidates.sort(reverse=False)

        for (sim, i, j) in candidates[:100]:
            st.markdown(f"**Possible duplicate ({i}, {j}, sim: {sim:.3f}):**")
            st.markdown("* " + " ".join(context.split["tokens"][i]))
            st.markdown("* " + " ".join(context.split["tokens"][j]))

        # st.write("queries")
        # results = split.get_nearest_examples("sent_index", np.array(split["embeddings"][0], dtype=np.float32), k=2)
        # results = split.get_nearest_examples_batch("sent_index", queries, k=2)
        # st.write(results.total_examples[0]["id"][1])
        # st.write(results.total_examples[0])
