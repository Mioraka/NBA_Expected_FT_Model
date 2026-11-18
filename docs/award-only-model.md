# Award-only shooting free throw model

The primary specification is a corrected broad-zone season/type league-rate baseline multiplied by a fitted MVP, All-NBA and All-Star correction. Scorer, FG%, FGM and other free throws are excluded as predictors. Shooting FTA target includes resolved shooting-foul FTA and and-ones only.

Run `python scripts/fit_award_only.py`. Add `--award-timing previous` for immediately previous season awards. Regular season and playoffs have separate coefficients and baseline rates. Entire player histories are held out in five folds; every training offset also excludes the focal player's history.

## Required inputs

Restore the corrected `data/processed/player_shooting_zone_model_table.csv` from the previously saved model data. The repository's legacy one-foot Parquet is deliberately rejected. The corrected CSV must retain separate `current_MVP`, `current_ALL_NBA`, `current_ALL_STAR` and corresponding `previous_` fields. Those flags are overlapping binary indicators, not a highest-career-award label.

Add original event data as `data/raw/datanba`, then run `python scripts/fit_award_only.py --events data/raw/datanba`. The ablation adds recorded-shot type shares: dunk, driving non-dunk, other layup, hook, jumper and unknown (other is the reference). It does not claim to classify all drives or foul-only opportunities. Event FGA must exactly reconcile with the corrected table. No automatic reinterpretation of inconsistent event identities is performed.

## Other free throws: output only

The enrichment code in `scripts/build_enriched_tables.py` carries `FTA_Other` into player-season stats. `--reporting-totals` accepts that CSV/Parquet, an explicit RS/PO totals CSV, or the original player XLSX. Default discovery checks `data/processed/player_season_stats_table.csv` then `data/raw/NBA_FT_Dist_Player.xlsx`. Sources need `FTA_Other`, `games_played` or `GP`, player_key or player ID, and explicit season. `SEASON_YEAR`/`season` are interpreted as season-ending years; `Season Code` is the starting year. If Type is absent the source is regular-season only. Unmatched playoff rows remain missing.

Outputs retain other_FTA, games_played, other_FTA_per_game, actual shooting FTA/game, predicted shooting FTA/game, and actual/predicted ratios. Other FTA never enters the target, model matrices, offsets or calibration. Missing source totals are marked rather than inferred from total minus resolved shooting FTA.

## Results and limitations

Outputs: heldout_predictions.csv, model_comparison.csv, coefficients.csv and audit.json. Compare MAE, RMSE and signed bias for All, any award and overlapping award groups separately in RS/PO. Other FTA/game is a reporting join after fitting. Current-season awards are retrospective, and residuals are not causal evidence of officiating favoritism.

Actions runs eight unit tests and conditionally runs fits when corrected input is present. It explicitly reports missing corrected or event data. A passing unit-test job does not establish that the statistical experiment ran. Prior numerical results used scorer flags and must not be presented as new award-only results.

## Restored source inputs

The corrected model CSV is versioned losslessly as player_shooting_zone_model_table.csv.gz. Recorded-shot counts from all original event rows are stored as shot_type_counts.csv.gz. Actions unpacks both and runs the complete ablation with --shot-type-counts. The count loader checks nonnegative integer counts, unique keys, category sums and exact FGA reconciliation. These counts reproduce the features computed directly from the raw event Parquet. The full 149 MB original event source remains in ChatGPT Library; it is not duplicated in this repository. Code still accepts --events for re-extraction when that source is available locally.

Other FTA is populated for RS from the original workbook and excluded from fitting. PO other totals and official games played remain missing, so no playoff per-game MAE is inferred.
