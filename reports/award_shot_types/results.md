# Award-only and shot-type validation results

Five player-held-out folds; separate RS/PO season-zone baselines and coefficients. Features: current-season MVP, All-NBA, All-Star only; no scorer/FG%/other FTA features. Shot-type variant adds shares of recorded FGA by mutually exclusive description categories. Training offsets exclude the focal player’s history.

| Competition | Model | N | MAE (FTA/season) | MAE/game |
|---|---|---:|---:|---:|
| RS | shot_baseline | 4881 | 23.0902 | 0.4513 |
| RS | award_only | 4881 | 21.8292 | 0.4302 |
| RS | award_plus_shot_types | 4881 | 20.7495 | 0.4143 |
| PO | shot_baseline | 1906 | 5.4214 | Unavailable |
| PO | award_only | 1906 | 5.0200 | Unavailable |
| PO | award_plus_shot_types | 1906 | 4.7953 | Unavailable |

Adding shot type reduces seasonal MAE relative to award-only by 4.95% RS and 4.47% PO. This is exploratory feature-set selection without an independent final test set.

The heldout predictions retain other_FTA and other_FTA_per_game, joined after fitting. RS totals are sourced from the original player workbook; PO other totals and official games played remain missing. Season-level playoff results use the original complete target rows; known coordinate/game-attribution caveats remain. These results do not establish referee favoritism.

Event data are stored in nine per-season partitions, preserving all 5,456,646 rows and columns. Source hashes and row counts are recorded in data/raw/datanba_manifest.json. Eight unit tests pass, including equivalence of optimized and explicit player-excluded offsets.
