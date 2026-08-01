"""
Picks the better of LR vs RF per stock, purely by whichever test accuracy is
higher - no tie-break threshold, no preference for one algorithm.

Usage:
    python scripts/select_winner.py

Requires data/training_summary.csv to already exist (run scripts/run_training.py first).

Output (not committed - derived data):
    data/final_model_selection.csv
"""
import pandas as pd


def main():
    summary = pd.read_csv('data/training_summary.csv')

    def pick_winner(row):
        if row['rf_test_accuracy'] > row['lr_test_accuracy']:
            return 'RF'
        elif row['lr_test_accuracy'] > row['rf_test_accuracy']:
            return 'LR'
        else:
            return 'Tie'  # exact tie on accuracy - genuinely rare, both equally valid

    summary['winner'] = summary.apply(pick_winner, axis=1)
    summary['winner_accuracy'] = summary[['lr_test_accuracy', 'rf_test_accuracy']].max(axis=1)
    summary['beats_baseline'] = summary['winner_accuracy'] > summary['test_majority_baseline']
    summary['margin_over_baseline'] = summary['winner_accuracy'] - summary['test_majority_baseline']

    summary.to_csv('data/final_model_selection.csv', index=False)

    print(f"RF wins: {(summary['winner']=='RF').sum()}, "
          f"LR wins: {(summary['winner']=='LR').sum()}, "
          f"Ties: {(summary['winner']=='Tie').sum()}")
    print(f"Best-of-two beats baseline: {summary['beats_baseline'].sum()}/{len(summary)} stocks")
    print(f"Wrote data/final_model_selection.csv")


if __name__ == "__main__":
    main()
