"""
For every token in the dataset, we take its hidden state and project it onto a two-dimensional plane. Data points are colored by label/prediction, with disagreements marked by a small black border.
"""
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.subpages.page import Context, Page


@st.cache
def reduce_dim_svd(X, n_iter: int, random_state=42):
    """Dimensionality reduction using truncated SVD (aka LSA).

    This transformer performs linear dimensionality reduction by means of truncated singular value decomposition (SVD). Contrary to PCA, this estimator does not center the data before computing the singular value decomposition. This means it can work with sparse matrices efficiently.

        Args:
            X: Training data
            n_iter (int): Desired dimensionality of output data. Must be strictly less than the number of features.
            random_state (int, optional): Used during randomized svd. Pass an int for reproducible results across multiple function calls. Defaults to 42.

        Returns:
            ndarray: Reduced version of X, ndarray of shape (n_samples, 2).
    """
    from sklearn.decomposition import TruncatedSVD

    svd = TruncatedSVD(n_components=2, n_iter=n_iter, random_state=random_state)
    return svd.fit_transform(X)


@st.cache
def reduce_dim_pca(X, random_state=42):
    """Principal component analysis (PCA).

    Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space. The input data is centered but not scaled for each feature before applying the SVD.

        Args:
            X: Training data
            random_state (int, optional): Used when the 'arpack' or 'randomized' solvers are used. Pass an int for reproducible results across multiple function calls.

        Returns:
            ndarray: Reduced version of X, ndarray of shape (n_samples, 2).
    """
    from sklearn.decomposition import PCA

    return PCA(n_components=2, random_state=random_state).fit_transform(X)


@st.cache
def reduce_dim_umap(X, n_neighbors=5, min_dist=0.1, metric="euclidean"):
    """Uniform Manifold Approximation and Projection

    Finds a low dimensional embedding of the data that approximates an underlying manifold.

        Args:
            X: Training data
            n_neighbors (int, optional): The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation. Larger values result in more global views of the manifold, while smaller values result in more local data being preserved. In general values should be in the range 2 to 100. Defaults to 5.
            min_dist (float, optional): The effective minimum distance between embedded points. Smaller values will result in a more clustered/clumped embedding where nearby points on the manifold are drawn closer together, while larger values will result on a more even dispersal of points. The value should be set relative to the `spread` value, which determines the scale at which embedded points will be spread out. Defaults to 0.1.
            metric (str, optional): The metric to use to compute distances in high dimensional space (see UMAP docs for options). Defaults to "euclidean".

        Returns:
            ndarray: Reduced version of X, ndarray of shape (n_samples, 2).
    """
    from umap import UMAP

    return UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric).fit_transform(X)


class HiddenStatesPage(Page):
    name = "Hidden States"
    icon = "grid-3x3"

    def _get_widget_defaults(self):
        return {
            "n_tokens": 1_000,
            "svd_n_iter": 5,
            "svd_random_state": 42,
            "umap_n_neighbors": 15,
            "umap_metric": "euclidean",
            "umap_min_dist": 0.1,
        }

    def render(self, context: Context):
        st.title("Embeddings")

        with st.expander("💡", expanded=True):
            st.write(
                "For every token in the dataset, we take its hidden state and project it onto a two-dimensional plane. Data points are colored by label/prediction, with disagreements signified by a small black border."
            )

        col1, _, col2 = st.columns([9 / 32, 1 / 32, 22 / 32])
        df = context.df_tokens_merged.copy()
        dim_algo = "SVD"
        n_tokens = 100

        with col1:
            st.subheader("Settings")
            n_tokens = st.slider(
                "#tokens",
                key="n_tokens",
                min_value=100,
                max_value=len(df["tokens"].unique()),
                step=100,
            )

            dim_algo = st.selectbox("Dimensionality reduction algorithm", ["SVD", "PCA", "UMAP"])
            if dim_algo == "SVD":
                svd_n_iter = st.slider(
                    "#iterations",
                    key="svd_n_iter",
                    min_value=1,
                    max_value=10,
                    step=1,
                )
            elif dim_algo == "UMAP":
                umap_n_neighbors = st.slider(
                    "#neighbors",
                    key="umap_n_neighbors",
                    min_value=2,
                    max_value=100,
                    step=1,
                )
                umap_min_dist = st.number_input(
                    "Min distance", key="umap_min_dist", value=0.1, min_value=0.0, max_value=1.0
                )
                umap_metric = st.selectbox(
                    "Metric", ["euclidean", "manhattan", "chebyshev", "minkowski"]
                )
            else:
                pass

        with col2:
            sents = df.groupby("ids").apply(lambda x: " ".join(x["tokens"].tolist()))

            X = np.array(df["hidden_states"].tolist())
            transformed_hidden_states = None
            if dim_algo == "SVD":
                transformed_hidden_states = reduce_dim_svd(X, n_iter=svd_n_iter)  # type: ignore
            elif dim_algo == "PCA":
                transformed_hidden_states = reduce_dim_pca(X)
            elif dim_algo == "UMAP":
                transformed_hidden_states = reduce_dim_umap(
                    X, n_neighbors=umap_n_neighbors, min_dist=umap_min_dist, metric=umap_metric  # type: ignore
                )

            assert isinstance(transformed_hidden_states, np.ndarray)
            df["x"] = transformed_hidden_states[:, 0]
            df["y"] = transformed_hidden_states[:, 1]
            df["sent0"] = df["ids"].map(lambda x: " ".join(sents[x][0:50].split()))
            df["sent1"] = df["ids"].map(lambda x: " ".join(sents[x][50:100].split()))
            df["sent2"] = df["ids"].map(lambda x: " ".join(sents[x][100:150].split()))
            df["sent3"] = df["ids"].map(lambda x: " ".join(sents[x][150:200].split()))
            df["sent4"] = df["ids"].map(lambda x: " ".join(sents[x][200:250].split()))
            df["disagreements"] = df["labels"] != df["preds"]

            subset = df[:n_tokens]
            disagreements_trace = go.Scatter(
                x=subset[subset["disagreements"]]["x"],
                y=subset[subset["disagreements"]]["y"],
                mode="markers",
                marker=dict(
                    size=6,
                    color="rgba(0,0,0,0)",
                    line=dict(width=1),
                ),
                hoverinfo="skip",
            )

            st.subheader("Projection Results")

            fig = px.scatter(
                subset,
                x="x",
                y="y",
                color="labels",
                hover_data=["ids", "preds", "sent0", "sent1", "sent2", "sent3", "sent4"],
                hover_name="tokens",
                title="Colored by label",
            )
            fig.add_trace(disagreements_trace)
            st.plotly_chart(fig)

            fig = px.scatter(
                subset,
                x="x",
                y="y",
                color="preds",
                hover_data=["ids", "labels", "sent0", "sent1", "sent2", "sent3", "sent4"],
                hover_name="tokens",
                title="Colored by prediction",
            )
            fig.add_trace(disagreements_trace)
            st.plotly_chart(fig)
