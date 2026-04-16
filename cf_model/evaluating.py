"""
evaluating.py

Evaluation utilities for both CF models, plus NOMINATE correlation
analysis and plotting helpers.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_absolute_error, accuracy_score

from training import predict_vote_memory, fit_svd


# ---------------------------------------------------------------------------
# Shared: build evaluation sample
# ---------------------------------------------------------------------------

def build_eval_sample(
    votes_df: pd.DataFrame,
    n: int = 500,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Sample n known (non-zero, non-NaN) votes to use as a held-out test set.
    """
    known_votes = votes_df[votes_df["vote_numeric"] != 0][
        ["senator_id", "bill_id", "vote_numeric"]
    ]
    return known_votes.sample(n=min(n, len(known_votes)), random_state=random_state)


# ---------------------------------------------------------------------------
# Method 1: User-User CF evaluation
# ---------------------------------------------------------------------------

def evaluate_memory_cf(
    sample: pd.DataFrame,
    vote_matrix: pd.DataFrame,
    sim_df: pd.DataFrame,
    k: int = 10,
) -> tuple[float, float]:
    """
    Evaluate the memory-based user-user CF model on a sample of known votes.

    Returns:
        accuracy -- fraction of binary predictions correct
        mae      -- mean absolute error of continuous predictions
    """
    actuals, predictions = [], []

    for _, row in sample.iterrows():
        pred = predict_vote_memory(
            row["senator_id"], row["bill_id"], vote_matrix, sim_df, k=k
        )
        if pred is not None:
            actuals.append(row["vote_numeric"])
            predictions.append(pred)

    pred_binary = [1 if p > 0 else -1 for p in predictions]
    mae = mean_absolute_error(actuals, predictions)
    acc = accuracy_score(actuals, pred_binary)

    print(f"Memory-based CF (k={k}) results (n={len(actuals)} predictions):")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  MAE:      {mae:.4f}")
    return acc, mae


# ---------------------------------------------------------------------------
# Method 2: SVD evaluation
# ---------------------------------------------------------------------------

def evaluate_svd(
    sample: pd.DataFrame,
    reconstructed: np.ndarray,
    senator_ids: list,
    bill_ids: list,
    n_components: int,
) -> tuple[float, float]:
    """
    Evaluate the SVD matrix factorization model on a sample of known votes.

    Returns:
        accuracy -- fraction of binary predictions correct
        mae      -- mean absolute error of continuous predictions
    """
    senator_idx = {s: i for i, s in enumerate(senator_ids)}
    bill_idx    = {b: i for i, b in enumerate(bill_ids)}
    actuals_svd, predictions_svd = [], []

    for _, row in sample.iterrows():
        sid, bid = row["senator_id"], row["bill_id"]
        if sid in senator_idx and bid in bill_idx:
            pred = reconstructed[senator_idx[sid], bill_idx[bid]]
            actuals_svd.append(row["vote_numeric"])
            predictions_svd.append(pred)

    pred_binary_svd = [1 if p > 0 else -1 for p in predictions_svd]
    mae_svd = mean_absolute_error(actuals_svd, predictions_svd)
    acc_svd = accuracy_score(actuals_svd, pred_binary_svd)

    print(f"SVD (n_components={n_components}) results (n={len(actuals_svd)} predictions):")
    print(f"  Accuracy: {acc_svd:.4f}")
    print(f"  MAE:      {mae_svd:.4f}")
    return acc_svd, mae_svd


def sweep_svd_components(
    sample: pd.DataFrame,
    vote_matrix_filled: np.ndarray,
    senator_ids: list,
    bill_ids: list,
    component_range: list[int] = None,
) -> tuple[list[int], list[float]]:
    """
    Sweep over a range of n_components values and record accuracy for each.

    Returns:
        component_range -- list of n values actually tested
        accuracies      -- corresponding accuracy for each n
    """
    if component_range is None:
        component_range = [2, 5, 10, 20, 50, 100]

    senator_idx = {s: i for i, s in enumerate(senator_ids)}
    bill_idx    = {b: i for i, b in enumerate(bill_ids)}
    accuracies  = []
    tested_ns   = []

    for n in component_range:
        if n >= min(vote_matrix_filled.shape):
            break
        svd_test = TruncatedSVD(n_components=n, random_state=42)
        sf = svd_test.fit_transform(vote_matrix_filled)
        recon = sf @ svd_test.components_

        preds, acts = [], []
        for _, row in sample.iterrows():
            sid, bid = row["senator_id"], row["bill_id"]
            if sid in senator_idx and bid in bill_idx:
                preds.append(1 if recon[senator_idx[sid], bill_idx[bid]] > 0 else -1)
                acts.append(row["vote_numeric"])

        accuracies.append(accuracy_score(acts, preds))
        tested_ns.append(n)

    best_n = tested_ns[int(np.argmax(accuracies))]
    print(f"Best n_components: {best_n}  (accuracy: {max(accuracies):.4f})")
    return tested_ns, accuracies


# ---------------------------------------------------------------------------
# Ideology: NOMINATE vs SVD correlation
# ---------------------------------------------------------------------------

def compute_nominate_correlation(
    svd_ideology_bio: pd.DataFrame,
    senators_df: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    """
    Merge SVD ideology scores with NOMINATE dim1 and compute the correlation.

    Args:
        svd_ideology_bio: DataFrame with columns ['bioguide_id', 'svd_dim1', 'svd_dim2'].
        senators_df:      Senator metadata with columns ['id', 'nominate_dim1_y'].

    Returns:
        ideology_merge -- merged DataFrame
        correlation    -- Pearson r between svd_dim1 and nominate_dim1_y
    """
    ideology_merge = svd_ideology_bio.merge(
        senators_df[["id", "nominate_dim1_y"]],
        left_on="bioguide_id",
        right_on="id",
        how="inner",
    ).dropna(subset=["nominate_dim1_y"])

    correlation = ideology_merge["svd_dim1"].corr(ideology_merge["nominate_dim1_y"])
    print(f"Correlation between SVD and NOMINATE: {correlation:.4f}")
    return ideology_merge, correlation


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_similarity_heatmap(sim_df: pd.DataFrame) -> None:
    """Heatmap of the senator-senator cosine similarity matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        sim_df.values,
        cmap="RdBu_r",
        center=0,
        xticklabels=False,
        yticklabels=False,
    )
    plt.title("Senator-Senator Cosine Similarity (red = similar, blue = opposite)")
    plt.tight_layout()
    plt.show()


def plot_svd_vs_nominate(ideology_merge: pd.DataFrame, correlation: float) -> None:
    """Scatter plot of SVD dim1 vs NOMINATE dim1."""
    plt.figure(figsize=(8, 6))
    plt.scatter(ideology_merge["nominate_dim1_y"], ideology_merge["svd_dim1"], alpha=0.6)
    plt.xlabel("NOMINATE dim1 (liberal ← → conservative)")
    plt.ylabel("SVD dim1 (learned from votes)")
    plt.title(f"SVD vs NOMINATE Ideology Score\nCorrelation: {correlation:.3f}")
    plt.tight_layout()
    plt.show()


def plot_senator_latent_space(svd_ideology: pd.DataFrame) -> None:
    """Scatter plot of senator positions in the SVD latent space (dim1 vs dim2)."""
    plt.figure(figsize=(9, 7))
    plt.scatter(svd_ideology["svd_dim1"], svd_ideology["svd_dim2"], alpha=0.7, s=40)
    plt.xlabel("SVD Dimension 1 (ideology)")
    plt.ylabel("SVD Dimension 2")
    plt.title("Senator Positions in SVD Latent Space")
    plt.tight_layout()
    plt.show()


def plot_svd_component_sweep(tested_ns: list[int], accuracies: list[float]) -> None:
    """Line plot of SVD accuracy vs. number of latent dimensions."""
    plt.figure(figsize=(7, 4))
    plt.plot(tested_ns, accuracies, marker="o")
    plt.xlabel("Number of SVD Components")
    plt.ylabel("Accuracy")
    plt.title("SVD Accuracy vs. Number of Latent Dimensions")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(acc: float, mae: float, acc_svd: float, mae_svd: float, n_components: int) -> None:
    print(f"User-user CF (k=10 most similar senators):")
    print(f"  Accuracy: {acc:.4f}  MAE: {mae:.4f}\n")
    print(f"SVD matrix factorization (n={n_components} components):")
    print(f"  Accuracy: {acc_svd:.4f}  MAE: {mae_svd:.4f}")