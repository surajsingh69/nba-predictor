
import pandas as pd
import numpy as np
def load_games(path="nba_games.csv"):
    df = pd.read_csv(path, index_col=0)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["team", "date"]).reset_index(drop=True)
    return df

ROLLING_STATS = [
    "pts", "fg%", "3p%", "ft%", "orb", "drb", "trb", "ast", "stl", "blk",
    "tov", "pts_opp", "fg%_opp", "trb_opp", "ast_opp", "tov_opp",
]

def add_rolling_features(df, window=10):
    """
    For each team, compute the rolling average of each stat over their last
    `window` games — SHIFTED by 1 game so the row for game N only uses games
    before N, never game N itself.
    """
    df = df.sort_values(["team", "date"]).copy()

    rolled = (
        df.groupby("team")[ROLLING_STATS]
        .apply(lambda g: g.shift(1).rolling(window, min_periods=3).mean())
        .reset_index(level=0, drop=True)
    )
    rolled.columns = [f"{c}_roll{window}" for c in rolled.columns]

    df = pd.concat([df, rolled], axis=1)
    return df


def add_rest_days(df):
    df = df.sort_values(["team", "date"]).copy()
    df["prev_game_date"] = df.groupby("team")["date"].shift(1)
    df["rest_days"] = (df["date"] - df["prev_game_date"]).dt.days
    # first game of a team's history has no previous game — fill with a neutral value
    df["rest_days"] = df["rest_days"].fillna(7).clip(upper=10)
    df["back_to_back"] = (df["rest_days"] <= 1).astype(int)
    df = df.drop(columns=["prev_game_date"])
    return df

def add_recent_form(df, window=10):
    df = df.sort_values(["team", "date"]).copy()
    df["won_int"] = df["won"].astype(int)
    df[f"win_pct_last{window}"] = (
        df.groupby("team")["won_int"]
        .apply(lambda g: g.shift(1).rolling(window, min_periods=3).mean())
        .reset_index(level=0, drop=True)
    )
    df = df.drop(columns=["won_int"])
    return df

def add_head_to_head(df):
    df = df.sort_values(["team", "team_opp", "date"]).copy()
    df["won_int"] = df["won"].astype(int)
    df["h2h_win_pct"] = (
        df.groupby(["team", "team_opp"])["won_int"]
        .apply(lambda g: g.shift(1).expanding(min_periods=1).mean())
        .reset_index(level=[0, 1], drop=True)
    )
    
    df["h2h_win_pct"] = df["h2h_win_pct"].fillna(0.5)
    df = df.drop(columns=["won_int"])
    return df



def build_matchup_table(df):
    feature_cols = [c for c in df.columns if c.endswith(("_roll10",)) or
                    c in ["rest_days", "back_to_back", "win_pct_last10", "h2h_win_pct"]]

    keep = ["team", "team_opp", "date", "season", "home", "won"] + feature_cols
    base = df[keep].copy()

    opp = base[["team", "date"] + feature_cols].copy()
    opp = opp.rename(columns={"team": "team_opp", **{c: f"opp_{c}" for c in feature_cols}})

    merged = base.merge(opp, on=["team_opp", "date"], how="left")
    return merged



def main():
    df = load_games("nba_games.csv")
    df = add_rolling_features(df, window=10)
    df = add_rest_days(df)
    df = add_recent_form(df, window=10)
    df = add_head_to_head(df)

    matchups = build_matchup_table(df)

    
    feature_cols = [c for c in matchups.columns if c.endswith("_roll10") or "opp_" in c]
    before = len(matchups)
    matchups = matchups.dropna(subset=[c for c in feature_cols if not c.startswith("opp_")])
    print(f"Dropped {before - len(matchups)} early-season rows with insufficient history")

    matchups.to_csv("features.csv", index=False)
    print(f"Saved features.csv: {matchups.shape[0]} rows, {matchups.shape[1]} columns")
    print("\nSample columns:", [c for c in matchups.columns if "roll10" in c][:6])
    print("\nTarget balance (won):")
    print(matchups["won"].value_counts(normalize=True))


if __name__ == "__main__":
    main()