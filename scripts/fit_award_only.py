"""Award-only shooting FTA expectations and player-held-out shot-type ablations.

Requires corrected broad-zone input. Never falls back to the legacy distance table.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold

KEY = ["player_key", "Season Code", "Type"]
AWARDS = ["MVP", "ALL_NBA", "ALL_STAR"]
TWO = ["0–<1", "1–<4", "4–<7", "7–<10", "10–<13",
       "13–<16", "16–<19", "19–<22", "22+ 2PT"]
THREE = ["Corner 3", "Arc 3", "3PT below 22"]
TARGET = ["FTA Normal 2PT", "FTA Normal 3PT", "FTA Normal Unknown",
          "FTA +1 2PT", "FTA +1 3PT", "FTA +1 Unknown"]

def read_table(path):
    return pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)

def validate(d):
    required = KEY + ["zone", "FGA Total"] + TARGET
    missing = set(required) - set(d)
    if missing:
        raise ValueError("Missing corrected input fields: " + str(sorted(missing)))
    if d.duplicated(KEY + ["zone"]).any():
        raise ValueError("Duplicate player-season-type-zone")
    if not set(d.zone).issubset(set(TWO + THREE + ["Unknown"])):
        raise ValueError("Input is not the corrected broad-zone table")
    if d[["FGA Total"] + TARGET].isna().any().any() or (d[["FGA Total"] + TARGET] < 0).any().any():
        raise ValueError("Missing or negative counts")
    if not set(d.Type).issubset({"RS", "PO"}):
        raise ValueError("Unknown competition type")

def award_features(d, timing):
    cols = [timing + "_" + a for a in AWARDS]
    if not set(cols).issubset(d):
        raise ValueError("Missing separate season-specific award flags: " + str(cols))
    for c in cols:
        if not d[c].isin([0, 1]).all() or (d.groupby(KEY)[c].nunique() > 1).any():
            raise ValueError("Invalid/inconsistent award flag: " + c)
    x = d.groupby(KEY)[cols].first().astype(float)
    x.insert(0, "intercept", 1.)
    return x

def baseline(train, test, alpha=100):
    """Neighbor prior and unknown-location contribution; no held-out outcomes."""
    tables = {}
    for context, g in train.groupby(["Season Code", "Type"]):
        a = g.groupby("zone")[["FGA Total", "target"]].sum()
        exposure = a["FGA Total"].sum()
        if exposure <= 0:
            raise ValueError("No calibration exposure")
        unknown = a.loc[(a.index == "Unknown") | (a["FGA Total"] == 0), "target"].sum() / exposure
        known = a.loc[(a.index != "Unknown") & (a["FGA Total"] > 0)]
        global_known = known.target.sum() / known["FGA Total"].sum()
        rates = {}
        for z in TWO + THREE + ["Unknown"]:
            if z == "Unknown":
                rates[z] = unknown
                continue
            if z in TWO:
                j = TWO.index(z)
                pool = TWO[max(0, j-1):j] + TWO[j+1:j+2]
            else:
                pool = THREE
            ref = a.reindex(pool).fillna(0)
            prior = ref.target.sum()/ref["FGA Total"].sum() if ref["FGA Total"].sum() else global_known
            n = a["FGA Total"].get(z, 0)
            y = a.target.get(z, 0)
            rates[z] = ((y + alpha*prior)/(n + alpha) if n else prior) + unknown
        tables[context] = rates
    values = []
    for r in test.to_dict("records"):
        ctx = (r["Season Code"], r["Type"])
        if ctx not in tables:
            raise ValueError("Missing independent calibration for " + str(ctx))
        values.append(r["FGA Total"] * tables[ctx][r["zone"]])
    x = test[KEY].copy()
    x["expected"] = values
    return x.groupby(KEY).expected.sum()

def fit(x, y, offset):
    def objective(beta):
        eta = x @ beta
        if np.max(np.abs(eta)) > 100:
            return float("inf"), np.zeros_like(beta)
        mu = np.maximum(offset, 1e-9) * np.exp(eta)
        penalty = np.r_[0., beta[1:]]
        return float(np.sum(mu-y*eta) + .5*np.sum(penalty**2)), x.T@(mu-y)+penalty
    result = minimize(objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(result.message)
    return result.x

def classify(description):
    """Mutually exclusive description classes; missing is explicitly unknown."""
    s = str(description).lower() if pd.notna(description) else ""
    if "dunk" in s: return "dunk"
    if "driving" in s: return "driving_non_dunk"
    if "layup" in s or "lay up" in s: return "other_layup"
    if "hook" in s: return "hook"
    if "jump" in s or "fadeaway" in s or "step back" in s: return "jumper"
    if not s: return "unknown"
    return "other"

def shot_features(path, d):
    import pyarrow.parquet as pq
    ids = d[["Player ID", "player_key"]].drop_duplicates()
    if ids["Player ID"].duplicated().any():
        raise ValueError("Ambiguous source player IDs")
    mapping = ids.set_index("Player ID").player_key
    chunks = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=200000,
                  columns=["pid", "etype", "de", "_season", "_season_type"]):
        f = batch.to_pandas()
        f = f[f.etype.isin([1, 2]) & f.pid.isin(mapping.index)].copy()
        f["player_key"] = f.pid.map(mapping)
        f["Season Code"] = f._season
        f["Type"] = f._season_type.map({"rg": "RS", "po": "PO"})
        if f.Type.isna().any(): raise ValueError("Unknown event season type")
        f["shot_type"] = f.de.map(classify)
        chunks.append(f.groupby(KEY+["shot_type"]).size().rename("n"))
    counts = pd.concat(chunks).groupby(level=list(range(4))).sum().unstack(fill_value=0)
    expected = d.groupby(KEY)["FGA Total"].sum()
    counts = counts.reindex(expected.index, fill_value=0)
    if not counts.sum(axis=1).equals(expected.astype("int64")):
        if not np.array_equal(counts.sum(axis=1), expected):
            raise ValueError("Shot-type FGA do not reconcile with corrected table")
    shares = counts.div(expected.replace(0, np.nan), axis=0).fillna(0)
    # One reference category removed to avoid exact compositional redundancy.
    shares = shares.reindex(columns=["dunk", "driving_non_dunk", "other_layup",
              "hook", "jumper", "unknown", "other"], fill_value=0)
    return shares.drop(columns="other").add_prefix("shot_type_")


def reporting_totals(predictions, detail, path=None):
    """Join existing other-FTA totals for reporting only; never derive by subtraction."""
    out = predictions.copy()
    out["other_FTA"] = np.nan
    out["games_played"] = np.nan
    if path is None:
        for candidate in ["data/processed/player_season_stats_table.csv",
                          "data/raw/NBA_FT_Dist_Player.xlsx"]:
            if Path(candidate).is_file():
                path = candidate
                break
    if path:
        raw = pd.read_excel(path) if str(path).endswith(".xlsx") else read_table(path)
        other = next((c for c in ["FTA_Other", "other_FTA"] if c in raw), None)
        gp = next((c for c in ["games_played", "GP"] if c in raw), None)
        if not other or not gp:
            raise ValueError("Reporting source requires FTA_Other and games_played/GP")
        raw = raw.copy()
        if "Type" not in raw:
            raw["Type"] = "RS"  # Legacy season totals contain regular season only.
        if "Season Code" not in raw:
            year = next((c for c in ["SEASON_YEAR", "season"] if c in raw), None)
            if not year: raise ValueError("Reporting source needs explicit season year")
            raw["Season Code"] = pd.to_numeric(raw[year], errors="raise") - 1
        if "player_key" not in raw:
            idcol = next((c for c in ["Player ID", "player_id"] if c in raw), None)
            if not idcol: raise ValueError("Reporting source needs player ID or player_key")
            ids = detail[["Player ID", "player_key"]].drop_duplicates()
            if ids["Player ID"].duplicated().any(): raise ValueError("Ambiguous reporting ID")
            raw = raw.merge(ids, left_on=idcol, right_on="Player ID", how="left",
                            validate="many_to_one")
        raw = raw.rename(columns={other: "_other", gp: "_gp"})
        if raw.duplicated(KEY).any(): raise ValueError("Duplicate reporting totals")
        for c in ["_other", "_gp"]:
            raw[c] = pd.to_numeric(raw[c], errors="raise")
        if (raw._other.dropna() < 0).any() or (raw._gp.dropna() <= 0).any():
            raise ValueError("Invalid reporting count")
        out = out.drop(columns=["other_FTA", "games_played"]).merge(
            raw[KEY+["_other", "_gp"]], on=KEY, how="left", validate="one_to_one"
        ).rename(columns={"_other": "other_FTA", "_gp": "games_played"})
    out["other_FTA_per_game"] = out.other_FTA / out.games_played
    out["actual_shooting_FTA_per_game"] = out.target / out.games_played
    for model in ["shot_baseline", "award_only", "award_plus_shot_types"]:
        if model in out:
            out[model+"_per_game"] = out[model] / out.games_played
            out["actual_to_"+model+"_ratio"] = out.target / out[model].replace(0, np.nan)
    out["other_FTA_status"] = np.where(out.other_FTA.notna() & out.games_played.notna(),
                                     "source_joined", "missing_source_totals")
    return out

def run(args):
    d = read_table(args.input)
    validate(d)
    d["target"] = d[TARGET].sum(axis=1)
    exposure = d.groupby(KEY)["FGA Total"].sum()
    excluded = exposure[exposure == 0].index
    d = d.set_index(KEY).drop(index=excluded).reset_index()
    features = award_features(d, args.award_timing)
    shots = shot_features(args.events, d) if args.events else None
    meta = d.groupby(KEY).agg(target=("target", "sum"))
    if "Player" in d: meta["Player"] = d.groupby(KEY).Player.first()
    for a in AWARDS: meta[args.award_timing+"_"+a] = features[args.award_timing+"_"+a]
    models = {"award_only": features}
    if shots is not None: models["award_plus_shot_types"] = features.join(shots, validate="one_to_one")
    rows, coefficients = [], []
    splitter = GroupKFold(args.folds)
    for fold, (tr, te) in enumerate(splitter.split(meta, groups=meta.index.get_level_values("player_key")), 1):
        train, test = meta.iloc[tr], meta.iloc[te]
        players = set(train.index.get_level_values("player_key"))
        cal = d[d.player_key.isin(players)]
        held = d[~d.player_key.isin(players)]
        result = test.copy()
        result["fold"] = fold
        result["shot_baseline"] = baseline(cal, held, args.prior).reindex(test.index)
        # Every training offset excludes the focal player's entire history.
        offsets = []
        for player, g in cal.groupby("player_key"):
            offsets.append(baseline(cal[cal.player_key != player], g, args.prior))
        offset = pd.concat(offsets).reindex(train.index)
        for name, x in models.items():
            if x.isna().any().any(): raise ValueError("Missing feature values")
            for typ in ["RS", "PO"]:
                ti = train.index[train.index.get_level_values("Type") == typ]
                vi = test.index[test.index.get_level_values("Type") == typ]
                if not len(ti) or not len(vi): continue
                beta = fit(x.loc[ti].to_numpy(), train.loc[ti].target.to_numpy(), offset.loc[ti].to_numpy())
                result.loc[vi, name] = result.loc[vi, "shot_baseline"]*np.exp(x.loc[vi].to_numpy()@beta)
                coefficients.extend(dict(fold=fold, Type=typ, model=name, feature=c,
                    coefficient=float(b), multiplier=float(np.exp(b))) for c,b in zip(x.columns,beta))
        rows.append(result.reset_index())
    predictions = pd.concat(rows, ignore_index=True)
    scores = []
    for typ in ["RS", "PO"]:
        for group in ["All", "Any current/prior award"] + AWARDS:
            q = predictions[predictions.Type == typ]
            flags = [args.award_timing+"_"+a for a in AWARDS]
            if group == "Any current/prior award": q = q[q[flags].max(axis=1) == 1]
            elif group in AWARDS: q = q[q[args.award_timing+"_"+group] == 1]
            for name in ["shot_baseline"]+list(models):
                e = q[name]-q.target
                scores.append(dict(Type=typ,group=group,model=name,n=len(q),
                    MAE=e.abs().mean(),RMSE=np.sqrt((e**2).mean()),bias=e.mean()))
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    predictions = reporting_totals(predictions, d, args.reporting_totals)
    predictions.to_csv(out/"heldout_predictions.csv",index=False)
    pd.DataFrame(scores).to_csv(out/"model_comparison.csv",index=False)
    pd.DataFrame(coefficients).to_csv(out/"coefficients.csv",index=False)
    audit = dict(primary="award_only",award_timing=args.award_timing,award_flags=AWARDS,
      scorer_features=False,other_FTA_features=False,other_FTA_target=False,
      other_FTA_missing_rows=int(predictions.other_FTA_per_game.isna().sum()),FG_percent_features=False,share_location_correction=False,
      shot_type_test=bool(args.events),zero_exposure_player_seasons_excluded=len(excluded),
      folds=args.folds,target=TARGET,limitations=[
      "Current awards are retrospective; previous means immediately previous season",
      "Shot descriptions classify recorded FGA, not all drives or fouled attempts",
      "Source coordinate and game-attribution limitations remain",
      "Exploratory ablation; no independent final test set"])
    (out/"audit.json").write_text(json.dumps(audit,indent=2))
    print(pd.DataFrame(scores).to_string(index=False))

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input",default="data/processed/player_shooting_zone_model_table.csv")
    p.add_argument("--reporting-totals", help="CSV/Parquet or XLSX with existing FTA_Other and games played; reporting only")
    p.add_argument("--events",help="Original datanba.parquet; required for shot-type ablation")
    p.add_argument("--award-timing",choices=["current","previous"],default="current")
    p.add_argument("--prior",type=float,default=100)
    p.add_argument("--folds",type=int,default=5)
    p.add_argument("--output",default="reports/award_only")
    run(p.parse_args())
