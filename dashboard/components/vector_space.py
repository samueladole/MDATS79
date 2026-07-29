"""
RAGScope — 3D Vector Space Visualisation
===========================================
Projects sampled embeddings down to 3 dimensions via PCA and renders them as
an interactive Plotly 3D scatter — a coarse, at-a-glance view of how the
indexed corpus clusters in embedding space, for the Knowledge Base page.

PCA is implemented directly with numpy's SVD rather than scikit-learn or
UMAP, to avoid adding a heavy ML dependency for what is purely a dashboard
visualisation aid (see docs/architecture_simplification_analysis.md for the
project's broader stance on minimal dependencies — numpy is already a
project dependency, scikit-learn/umap-learn are not).
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from pipeline.vectorstore import RetrievedChunk


def pca_3d(embeddings: list[list[float]]) -> np.ndarray:
    """
    Project a list of equal-length embedding vectors onto their top-3
    principal components.

    Parameters
    ----------
    embeddings : List of embedding vectors (e.g. 384-dimensional).

    Returns
    -------
    np.ndarray
        Array of shape ``(len(embeddings), 3)`` — coordinates along the
        3 directions of greatest variance in the input.
    """
    x = np.asarray(embeddings, dtype=np.float64)
    x_centered = x - x.mean(axis=0, keepdims=True)
    # Economy SVD: x_centered = U @ diag(S) @ Vt. The data's coordinates
    # along its principal axes are U @ diag(S); take the first 3 columns.
    u, s, _vt = np.linalg.svd(x_centered, full_matrices=False)
    return u[:, :3] * s[:3]


def vector_space_scatter(
    chunks: list[RetrievedChunk],
    dataset_labels: dict[str, str],
    dataset_colors: dict[str, str],
) -> None:
    """
    Render a 3D PCA scatter of ``chunks``' embeddings, coloured by dataset.

    Parameters
    ----------
    chunks         : Chunks with ``.embedding`` populated, e.g. from
                     ``VectorStore.sample_embeddings()``.
    dataset_labels : Maps raw ``metadata["dataset"]`` values to display names.
    dataset_colors : Maps display names to fixed hex colours, in the order
                     the legend/trace order should follow.
    """
    chunks = [c for c in chunks if c.embedding]
    if len(chunks) < 4:
        st.info("Not enough sampled points with embeddings to compute a projection.")
        return

    coords = pca_3d([c.embedding for c in chunks])
    dataset_names = [
        dataset_labels.get(c.metadata.get("dataset"), c.metadata.get("dataset") or "Other / Unlabelled")
        for c in chunks
    ]
    previews = [
        (c.text[:120] + "…") if c.text and len(c.text) > 120 else (c.text or "") for c in chunks
    ]

    fig = go.Figure()
    for ds_name in dataset_colors:
        idx = [i for i, d in enumerate(dataset_names) if d == ds_name]
        if not idx:
            continue
        fig.add_trace(
            go.Scatter3d(
                x=coords[idx, 0],
                y=coords[idx, 1],
                z=coords[idx, 2],
                mode="markers",
                name=ds_name,
                marker=dict(size=3, color=dataset_colors[ds_name], opacity=0.75),
                text=[f"{chunks[i].chunk_id}<br>{previews[i]}" for i in idx],
                hovertemplate="%{text}<extra></extra>",
            )
        )

    fig.update_layout(
        height=560,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", y=1.05),
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"{len(chunks):,} sampled points across {len({d for d in dataset_names}):,} dataset(s), "
        f"projected from {len(chunks[0].embedding)} dimensions via PCA. "
        "PCA preserves directions of greatest overall variance, not local neighbourhood structure — "
        "tight visual clusters are meaningful, but proximity between individual points is only approximate."
    )
