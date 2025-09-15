import argparse, json, math, os
from pathlib import Path
import numpy as np
import duckdb
import pandas as pd


def build_aggregates(ydir: Path) -> pd.DataFrame:
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={os.cpu_count() or 8}")

    # Paths
    listens = str((ydir / "listens.parquet").resolve())
    likes   = str((ydir / "likes.parquet").resolve())
    unlikes = str((ydir / "unlikes.parquet").resolve())
    dislikes   = str((ydir / "dislikes.parquet").resolve())
    undislikes = str((ydir / "undislikes.parquet").resolve())

    # helper to safely quote paths
    sq = lambda s: str(s).replace("'", "''")

    # Views (NOTE: literals instead of prepared params)
    con.execute(f"""
        CREATE TEMP VIEW listens AS
        SELECT * FROM read_parquet('{sq(listens)}')
    """)

    # Listen aggregates
    con.execute("""
        CREATE TEMP VIEW listen_stats AS
        SELECT
            uid::INTEGER AS uid,
            item_id::INTEGER AS item_id,
            COUNT(*)                           AS n_listens,
            AVG(played_ratio_pct)/100.0        AS mean_play_ratio,     -- 0..1
            AVG(track_length_seconds * (played_ratio_pct/100.0)) AS mean_secs_played,
            AVG(is_organic)                    AS organic_rate,         -- 0..1
            MAX(timestamp)                     AS ts_max
        FROM listens
        GROUP BY uid, item_id
    """)

    # Like / Unlike / Dislike / Undislike counts
    con.execute(f"""
        CREATE TEMP VIEW likes_v AS
        SELECT uid::INTEGER uid, item_id::INTEGER item_id, COUNT(*) c
        FROM read_parquet('{sq(likes)}') GROUP BY 1,2
    """)
    con.execute(f"""
        CREATE TEMP VIEW unlikes_v AS
        SELECT uid::INTEGER uid, item_id::INTEGER item_id, COUNT(*) c
        FROM read_parquet('{sq(unlikes)}') GROUP BY 1,2
    """)
    con.execute(f"""
        CREATE TEMP VIEW dislikes_v AS
        SELECT uid::INTEGER uid, item_id::INTEGER item_id, COUNT(*) c
        FROM read_parquet('{sq(dislikes)}') GROUP BY 1,2
    """)
    con.execute(f"""
        CREATE TEMP VIEW undislikes_v AS
        SELECT uid::INTEGER uid, item_id::INTEGER item_id, COUNT(*) c
        FROM read_parquet('{sq(undislikes)}') GROUP BY 1,2
    """)

    # Full outer join of all signals
    df = con.execute("""
        WITH base AS (
            SELECT uid, item_id FROM listen_stats
            UNION
            SELECT uid, item_id FROM likes_v
            UNION
            SELECT uid, item_id FROM unlikes_v
            UNION
            SELECT uid, item_id FROM dislikes_v
            UNION
            SELECT uid, item_id FROM undislikes_v
        )
        SELECT
            b.uid, b.item_id,
            ls.n_listens,
            ls.mean_play_ratio,
            ls.mean_secs_played,
            ls.organic_rate,
            ls.ts_max,
            COALESCE(l.c, 0)  AS like_cnt,
            COALESCE(ul.c, 0) AS unlike_cnt,
            COALESCE(d.c, 0)  AS dislike_cnt,
            COALESCE(ud.c, 0) AS undislike_cnt
        FROM base b
        LEFT JOIN listen_stats ls ON ls.uid=b.uid AND ls.item_id=b.item_id
        LEFT JOIN likes_v      l  ON l.uid=b.uid  AND l.item_id=b.item_id
        LEFT JOIN unlikes_v    ul ON ul.uid=b.uid AND ul.item_id=b.item_id
        LEFT JOIN dislikes_v   d  ON d.uid=b.uid  AND d.item_id=b.item_id
        LEFT JOIN undislikes_v ud ON ud.uid=b.uid AND ud.item_id=b.item_id
    """).df()

    return df

def compute_weights(df: pd.DataFrame,
                    a=1.0, b=0.8, c=0.4,
                    like_bonus=1.5, dislike_pen=2.0,
                    organic_boost=0.2,
                    half_life_days=30.0):
    # Fill NaNs with safe defaults
    df = df.copy()
    for col, val in [
        ("n_listens", 0), ("mean_play_ratio", 0.0), ("mean_secs_played", 0.0),
        ("organic_rate", 0.0), ("ts_max", np.nan),
        ("like_cnt", 0), ("unlike_cnt", 0), ("dislike_cnt", 0), ("undislike_cnt", 0),
    ]:
        if col in df.columns:
            df[col] = df[col].fillna(val)

    # derived counts (edits cancel)
    net_like     = (df["like_cnt"] - df["unlike_cnt"]).astype(float)
    net_dislike  = (df["dislike_cnt"] - df["undislike_cnt"]).astype(float)

    # terms
    term_listens = np.log1p(df["n_listens"].astype(float))
    term_ratio   = df["mean_play_ratio"].clip(0, 1)                 # 0..1
    term_dur     = (df["mean_secs_played"]/240.0).clip(0, 1)        # cap at 240s
    term_org     = (2.0*df["organic_rate"] - 1.0)                   # [-1,1], center at 0

    w = (
        a*term_listens
        + b*term_ratio
        + c*term_dur
        + like_bonus*net_like
        - dislike_pen*net_dislike
        + organic_boost*term_org
    )

    # time decay (timestamps are seconds within dataset; convert to days)
    if df["ts_max"].notna().any():
        ts = df["ts_max"].astype(float)
        now_ts = float(np.nanmax(ts.values))
        age_days = np.clip((now_ts - ts), 0, None) / 86400.0
        decay = np.exp(-math.log(2.0) * (age_days / float(half_life_days)))
        w = w * decay

    # sanitize
    w = pd.Series(w).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return w

def align_to_edge_index(df: pd.DataFrame, edge_index_path: Path, user_map_path: Path, item_map_path: Path, normalize_per_user: bool):
    edge_index = np.load(edge_index_path)  # [2, E]
    with open(user_map_path, "r") as f: u2i = json.load(f)
    with open(item_map_path, "r") as f: v2i = json.load(f)

    n_users = len(u2i)
    u = df["uid"].astype(str).map(u2i)
    i = df["item_id"].astype(str).map(v2i)
    mask = u.notna() & i.notna()
    df = df[mask].copy()
    df["u"] = u[mask].astype(np.int64)
    df["i"] = i[mask].astype(np.int64)

    # position map (u,i) -> column index
    u_e = edge_index[0].astype(np.int64)
    v_e = edge_index[1].astype(np.int64)
    i_e = v_e - n_users if v_e.min() >= n_users else v_e
    pos = {(int(uu), int(ii)): k for k,(uu,ii) in enumerate(zip(u_e, i_e))}

    edge_w = np.zeros(edge_index.shape[1], dtype=np.float32)
    hits = 0
    for uu,ii,ww in zip(df["u"].values, df["i"].values, df["weight"].values.astype(np.float32)):
        j = pos.get((int(uu), int(ii)))
        if j is not None:
            edge_w[j] = ww
            hits += 1

    if normalize_per_user:
        # row-normalize outgoing weights per user in edge_index order
        sums = np.zeros(n_users, dtype=np.float64)
        np.add.at(sums, u_e, edge_w)
        edge_w = edge_w / (sums[u_e] + 1e-12)

    return edge_w, hits

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yambda_dir", required=True, help="Folder with listens.parquet, likes/dislikes/unlikes/undislikes")
    ap.add_argument("--edge_index", help="Path to existing edge_index.npy")
    ap.add_argument("--user_map", help="user2idx.json")
    ap.add_argument("--item_map", help="item2idx.json")
    ap.add_argument("--out_path", help="Where to write edge_weight.npy (if aligning)")
    ap.add_argument("--half_life_days", type=float, default=30.0)
    ap.add_argument("--normalize_per_user", action="store_true")

    # weights hyperparams
    ap.add_argument("--a", type=float, default=1.0)       # log1p(listens)
    ap.add_argument("--b", type=float, default=0.8)       # mean play ratio
    ap.add_argument("--c", type=float, default=0.4)       # mean seconds played / 240
    ap.add_argument("--like_bonus", type=float, default=1.5)
    ap.add_argument("--dislike_pen", type=float, default=2.0)
    ap.add_argument("--organic_boost", type=float, default=0.2)
    args = ap.parse_args()

    # deps
    try:
        import pyarrow  # noqa: F401 (duckdb uses it)
    except:
        pass

    ydir = Path(args.yambda_dir)
    df = build_aggregates(ydir)

    df["weight"] = compute_weights(
        df,
        a=args.a, b=args.b, c=args.c,
        like_bonus=args.like_bonus, dislike_pen=args.dislike_pen,
        organic_boost=args.organic_boost,
        half_life_days=args.half_life_days
    )

    # If no edge_index given, just save a preview parquet for inspection
    if not args.edge_index:
        outp = ydir / "interactions_with_weights.parquet"
        df.to_parquet(outp, index=False)
        print(f"[ok] wrote {outp} with columns: {list(df.columns)}  rows={len(df)}")
        return

    # Align to existing edge_index
    if not (args.user_map and args.item_map and args.out_path):
        raise SystemExit("When --edge_index is set, you must also set --user_map, --item_map, and --out_path")

    edge_w, hits = align_to_edge_index(
        df,
        Path(args.edge_index),
        Path(args.user_map),
        Path(args.item_map),
        normalize_per_user=args.normalize_per_user
    )

    outp = Path(args.out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.save(outp, edge_w)
    print(f"[ok] wrote weights -> {outp}  (E={edge_w.shape[0]}, matched={hits})")

if __name__ == "__main__":
    main()
