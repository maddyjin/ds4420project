"""
preprocessing.py

Loads raw data, builds the senator-bill vote matrix, and computes
PCA-reduced embedding scores for downstream model features.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(
    bills_path: str = "./data/bills/bills_with_topic.csv",
    votes_path: str = "./data/votes/votes.csv",
    senators_path: str = "./data/senators/all_senators_with_bios.csv",
    embedded_path: str = "./eda/misc/all_info_embedded_trimmed.parquet",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all four source files and return them as a tuple."""
    bills_df = pd.read_csv(bills_path)
    votes_df = pd.read_csv(votes_path)
    senators_df = pd.read_csv(senators_path)
    all_info_embedded_trimmed_df = pd.read_parquet(embedded_path)
    return bills_df, votes_df, senators_df, all_info_embedded_trimmed_df


# ---------------------------------------------------------------------------
# Vote encoding
# ---------------------------------------------------------------------------

VOTE_MAP = {"Yea": 1, "Nay": -1, "Not Voting": 0, "Present": 0}


def encode_votes(votes_df: pd.DataFrame) -> pd.DataFrame:
    """Add a numeric vote column to the votes dataframe."""
    votes_df = votes_df.copy()
    votes_df["vote_numeric"] = votes_df["vote"].map(VOTE_MAP)
    return votes_df


# ---------------------------------------------------------------------------
# Vote matrix construction
# ---------------------------------------------------------------------------

def build_vote_matrix(
    votes_df: pd.DataFrame,
    index_col: str = "senator_id",
    column_col: str = "bill_id",
    value_col: str = "vote_numeric",
) -> tuple[pd.DataFrame, np.ndarray, list, list]:
    """
    Pivot votes into a senator × bill matrix.

    Returns:
        vote_matrix       -- sparse DataFrame (NaN where no vote)
        vote_matrix_filled -- NaN-filled-with-0 numpy array
        senator_ids        -- ordered list of senator identifiers
        bill_ids           -- ordered list of bill identifiers
    """
    vote_matrix = votes_df.pivot_table(
        index=index_col,
        columns=column_col,
        values=value_col,
    )
    vote_matrix_filled = vote_matrix.fillna(0).values
    senator_ids = vote_matrix.index.tolist()
    bill_ids = vote_matrix.columns.tolist()

    sparsity = vote_matrix.isna().mean().mean() * 100
    print(f"Matrix shape: {vote_matrix.shape}")
    print(f"  Number of bills voted on: {len(bill_ids)}")
    print(f"  Number of senators: {len(senator_ids)}")
    print(f"  Sparsity: {sparsity:.1f}% missing")

    return vote_matrix, vote_matrix_filled, senator_ids, bill_ids


def build_vote_matrix_bio(
    all_info_embedded_trimmed_df: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, list, list]:
    """
    Build a bioguide_id × bill_id vote matrix from the embedded dataframe.
    Used for SVD runs that need to align with NOMINATE ideology scores.
    """
    vote_matrix_bio = all_info_embedded_trimmed_df.pivot_table(
        index="bioguide_id",
        columns="bill_id",
        values="vote",
    )
    vote_matrix_bio_filled = vote_matrix_bio.fillna(0).values
    bioguide_ids = vote_matrix_bio.index.tolist()
    bill_ids_bio = vote_matrix_bio.columns.tolist()
    return vote_matrix_bio, vote_matrix_bio_filled, bioguide_ids, bill_ids_bio


# ---------------------------------------------------------------------------
# Embedding features
# ---------------------------------------------------------------------------

def add_pca_embedding_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce bill description and bio embedding columns to scalar scores
    via single-component PCA, and encode party as a numeric code.

    Expects columns: 'bill_des_embed', 'bio_embed', 'parties', 'vote'.

    Returns a copy of the dataframe with three new columns:
        bill_des_embed_score, bio_embed_score, parties_nums
    and filtered to only rows where vote is 0 or 1.
    """
    df = df.copy()

    pca = PCA(n_components=1)
    X = np.vstack(df["bill_des_embed"].values)
    df["bill_des_embed_score"] = pca.fit_transform(X)

    pca = PCA(n_components=1)
    X = np.vstack(df["bio_embed"].values)
    df["bio_embed_score"] = pca.fit_transform(X)

    df["parties"] = df["parties"].astype("category")
    df["parties_nums"] = df["parties"].cat.codes

    df_model = df[df["vote"].isin([0, 1])].copy()
    return df_model