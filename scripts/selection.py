import pandas as pd
from features import CANDIDATE_FEATURES

K = 5  # how many top features to keep per stock


def select_features(featured_df, k=K):
    """
    Given ONE stock's engineered DataFrame (output of engineer_features),
    returns:
      - selected: list of the top-k feature names by |correlation| with target
      - all_correlations: pandas Series of every candidate feature's correlation
        with target, sorted by absolute value (descending) - kept for transparency/reporting

    Drops rows with any NaN first (rolling-window warmup rows + the final
    NaN-target row) so the correlation isn't distorted by missing data.
    """
    clean = featured_df.dropna(subset=CANDIDATE_FEATURES + ['target'])

    correlations = clean[CANDIDATE_FEATURES].corrwith(clean['target'])
    all_correlations = correlations.reindex(
        correlations.abs().sort_values(ascending=False).index
    )

    selected = all_correlations.head(k).index.tolist()
    return selected, all_correlations
