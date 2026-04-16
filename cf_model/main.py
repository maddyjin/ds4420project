# main.py

from preprocessing import load_data, encode_votes, build_vote_matrix, build_vote_matrix_bio
from training import compute_similarity_matrix, fit_svd, build_ideology_df
from evaluating import (
    build_eval_sample, evaluate_memory_cf, evaluate_svd,
    compute_nominate_correlation, print_summary
)

# 1. Load and preprocess
bills_df, votes_df, senators_df, embedded_df = load_data()
votes_df = encode_votes(votes_df)
vote_matrix, vote_matrix_filled, senator_ids, bill_ids = build_vote_matrix(votes_df)

# 2. Build eval sample (do this before training so it's consistent)
sample = build_eval_sample(votes_df)

# 3. Train user-user CF and evaluate
sim_df = compute_similarity_matrix(vote_matrix_filled, senator_ids)
acc, mae = evaluate_memory_cf(sample, vote_matrix, sim_df)

# 4. Train SVD and evaluate
N_COMPONENTS = 20
svd, senator_factors, bill_factors, reconstructed = fit_svd(vote_matrix_filled, n_components=N_COMPONENTS)
acc_svd, mae_svd = evaluate_svd(sample, reconstructed, senator_ids, bill_ids, N_COMPONENTS)

# 5. NOMINATE correlation (needs bioguide_id matrix)
_, vote_matrix_bio_filled, bioguide_ids, _ = build_vote_matrix_bio(embedded_df)
_, senator_factors_bio, _, _ = fit_svd(vote_matrix_bio_filled, n_components=N_COMPONENTS)
svd_ideology_bio = build_ideology_df(senator_factors_bio, bioguide_ids, index_col="bioguide_id")
ideology_merge, correlation = compute_nominate_correlation(svd_ideology_bio, senators_df)

# 6. Summary
print_summary(acc, mae, acc_svd, mae_svd, N_COMPONENTS)