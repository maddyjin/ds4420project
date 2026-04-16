"""
training.py

Trains both collaborative filtering models:
  1. User-user CF (memory-based, cosine similarity)
  2. SVD matrix factorization (model-based)

Also includes utilities for senator similarity analysis and
ideology score construction for NOMINATE comparison.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD


# ---------------------------------------------------------------------------
# Method 1: User-User Collaborative Filtering
# ---------------------------------------------------------------------------

def compute_similarity_matrix(
    vote_matrix_filled: np.ndarray,
    senator_ids: list,
) -> pd.DataFrame:
    """
    Compute a senator × senator cosine similarity matrix from the filled vote matrix.

    Returns a DataFrame indexed and columned by senator_id.
    """
    norms = np.linalg.norm(vote_matrix_filled, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid division by zero
    normalized = vote_matrix_filled / norms
    sim_matrix = normalized @ normalized.T

    sim_df = pd.DataFrame(sim_matrix, index=senator_ids, columns=senator_ids)
    print(f"Similarity matrix shape: {sim_df.shape}")
    print(f"Mean similarity between senators: {sim_matrix[sim_matrix < 1].mean():.4f}")
    return sim_df


def predict_vote_memory(
    senator_id: str,
    bill_id: str,
    vote_matrix: pd.DataFrame,
    sim_df: pd.DataFrame,
    k: int = 10,
) -> float | None:
    """
    Predict how senator_id would vote on bill_id using the K most similar
    senators who actually voted on that bill.

    Args:
        senator_id: Identifier for the senator whose vote is being predicted.
        bill_id:    Identifier for the bill to predict a vote on.
        vote_matrix: Senator × bill matrix (1 = Yea, -1 = Nay, 0 = Abstain,
                     NaN = did not vote).
        sim_df:     Square senator × senator cosine similarity matrix.
        k:          Number of most-similar neighbors to use.

    Returns:
        Similarity-weighted average of neighbor votes in [-1, 1], or None if
        the prediction cannot be made (bill absent, no neighbors voted, etc.).
    """
    if bill_id not in vote_matrix.columns:
        return None

    bill_votes = vote_matrix[bill_id].dropna()
    bill_votes = bill_votes[bill_votes != 0]
    other_senators = bill_votes.index.tolist()

    if senator_id not in sim_df.index or len(other_senators) == 0:
        return None

    similarities = sim_df.loc[senator_id, other_senators]
    similarities = similarities[similarities.index != senator_id]
    top_k = similarities.nlargest(k)

    if top_k.sum() == 0:
        return None

    weighted_votes = sum(top_k[s] * bill_votes[s] for s in top_k.index)
    return weighted_votes / top_k.abs().sum()


# ---------------------------------------------------------------------------
# Method 2: SVD Matrix Factorization
# ---------------------------------------------------------------------------

def fit_svd(
    vote_matrix_filled: np.ndarray,
    n_components: int = 20,
    random_state: int = 42,
) -> tuple[TruncatedSVD, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a TruncatedSVD model on the filled vote matrix.

    Returns:
        svd               -- fitted TruncatedSVD object
        senator_factors   -- (n_senators, n_components) latent senator embeddings
        bill_factors      -- (n_components, n_bills) latent bill embeddings
        reconstructed     -- (n_senators, n_bills) reconstructed vote matrix
    """
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    senator_factors = svd.fit_transform(vote_matrix_filled)
    bill_factors = svd.components_
    reconstructed = senator_factors @ bill_factors

    print(f"Senator factors shape: {senator_factors.shape}")
    print(f"Bill factors shape:    {bill_factors.shape}")
    print(f"Variance explained by each component:\n  {np.round(svd.explained_variance_ratio_, 3)}")
    print(f"Total variance explained: {svd.explained_variance_ratio_.sum() * 100:.1f}%")
    print(f"Reconstructed matrix shape: {reconstructed.shape}")
    print(f"  Value range: [{reconstructed.min():.4f}, {reconstructed.max():.4f}]")

    return svd, senator_factors, bill_factors, reconstructed


# ---------------------------------------------------------------------------
# Senator ideology (SVD dim1)
# ---------------------------------------------------------------------------

def build_ideology_df(
    senator_factors: np.ndarray,
    senator_ids: list,
    index_col: str = "senator_id",
) -> pd.DataFrame:
    """
    Build a DataFrame with SVD dim1 and dim2 ideology scores for each senator.

    Args:
        senator_factors: (n_senators, n_components) matrix from fit_svd.
        senator_ids:     Ordered list of senator identifiers matching rows.
        index_col:       Column name to use for senator IDs (e.g. 'senator_id'
                         or 'bioguide_id').

    Returns:
        DataFrame sorted by svd_dim1 with columns [index_col, svd_dim1, svd_dim2].
    """
    ideology_df = pd.DataFrame({
        index_col:  senator_ids,
        "svd_dim1": senator_factors[:, 0],
        "svd_dim2": senator_factors[:, 1],
    }).sort_values("svd_dim1")
    return ideology_df


# ---------------------------------------------------------------------------
# Senator similarity analysis helpers
# ---------------------------------------------------------------------------

def most_similar_pairs(sim_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return the n most similar (positive) senator pairs."""
    sim_upper = sim_df.where(np.triu(np.ones(sim_df.shape), k=1).astype(bool))
    return (
        sim_upper.stack()
        .reset_index()
        .rename(columns={"senator_id": "senator_a", "bill_id": "senator_b", 0: "similarity"})
        .sort_values("similarity", ascending=False)
        .head(n)
    )


def most_opposite_pairs(sim_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return the n most opposite (negative similarity) senator pairs."""
    sim_upper = sim_df.where(np.triu(np.ones(sim_df.shape), k=1).astype(bool))
    return (
        sim_upper.stack()
        .reset_index()
        .rename(columns={"senator_id": "senator_a", "bill_id": "senator_b", 0: "similarity"})
        .sort_values("similarity", ascending=True)
        .head(n)
    )