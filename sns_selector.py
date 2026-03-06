import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os
import argparse
import random
import sys

try:
    import akshare as ak
except ModuleNotFoundError:
    exe = os.path.basename(sys.executable) or "python3"
    print("Missing dependency: akshare")
    print(f"Install with: {exe} -m pip install -U akshare pandas numpy")
    sys.exit(1)


MIN_HIST_SAMPLES = 90


# ---------- utils ----------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def safe_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def zscore(s):
    s = s.replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() < 5:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / (s.std(ddof=0) + 1e-9)


def with_retry(func, *args, retries=4, base_sleep=0.8, **kwargs):
    last_err = None
    for i in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_err = e
            if i == retries - 1:
                break
            time.sleep(base_sleep * (2 ** i))
    raise last_err


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def cache_path(cache_dir, kind, key):
    safe_key = str(key).replace("/", "_")
    return os.path.join(cache_dir, f"{kind}_{safe_key}.csv")


def normalize_code_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.extract(r"(\d{6})", expand=False)


def to_prefixed_code(code: str) -> str:
    code = str(code).zfill(6)
    if code.startswith(("5", "6", "9")):
        return f"sh{code}"
    return f"sz{code}"


def normalize_spot_df(spot: pd.DataFrame) -> pd.DataFrame:
    spot = spot.copy()
    if "code" in spot.columns:
        spot["code"] = normalize_code_series(spot["code"]).str.zfill(6)
    for c in ["amount", "mv", "turn", "pe", "pb"]:
        if c in spot.columns:
            spot[c] = pd.to_numeric(spot[c], errors="coerce")
        else:
            spot[c] = np.nan
    return spot


def get_index_constituents(index_symbols, cache_dir=None):
    frames = []
    for symbol in index_symbols:
        try:
            df = with_retry(ak.index_stock_cons_csindex, symbol=symbol, retries=4, base_sleep=0.8)
            code_col = safe_col(df, ["成分券代码", "成分股代码", "code"])
            name_col = safe_col(df, ["成分券名称", "成分股名称", "name"])
            idx_col = safe_col(df, ["指数名称", "index_name"])
            if code_col is None:
                continue
            part = pd.DataFrame(
                {
                    "code": normalize_code_series(df[code_col]).str.zfill(6),
                    "name": df[name_col] if name_col in df.columns else np.nan,
                    "index_name": df[idx_col] if idx_col in df.columns else symbol,
                    "index_code": symbol,
                }
            )
            frames.append(part)
        except Exception:
            continue

    if not frames:
        raise RuntimeError("index constituents fetch failed for all symbols")

    pool = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["code"]).reset_index(drop=True)
    if cache_dir:
        ensure_dir(cache_dir)
        pool.to_csv(cache_path(cache_dir, "pool", datetime.today().strftime("%Y%m%d")), index=False, encoding="utf-8-sig")
    return pool


# ---------- data fetch ----------
def get_spot(cache_dir=None):
    spot = None
    last_err = None
    for fetcher, retries in [(ak.stock_zh_a_spot_em, 4), (ak.stock_zh_a_spot, 3)]:
        try:
            spot = with_retry(fetcher, retries=retries, base_sleep=0.8)
            break
        except Exception as e:
            last_err = e
            continue
    if spot is None:
        raise last_err

    code_col = safe_col(spot, ["代码", "code", "股票代码"])
    name_col = safe_col(spot, ["名称", "name", "股票名称"])
    amt_col = safe_col(spot, ["成交额", "amount", "成交额(元)"])
    mv_col = safe_col(spot, ["总市值", "total_market_value", "总市值(元)"])
    turn_col = safe_col(spot, ["换手率", "turnover", "换手率%"])
    pe_col = safe_col(spot, ["市盈率-动态", "市盈率", "pe", "PE"])
    pb_col = safe_col(spot, ["市净率", "pb", "PB"])

    required = [code_col, name_col, amt_col]
    if any(c is None for c in required):
        raise ValueError(f"spot字段缺失, columns={list(spot.columns)}")

    spot = spot.rename(
        columns={
            code_col: "code",
            name_col: "name",
            amt_col: "amount",
            mv_col: "mv",
            turn_col: "turn",
            pe_col: "pe",
            pb_col: "pb",
        }
    )

    for c in ["amount", "mv", "turn", "pe", "pb"]:
        if c in spot.columns:
            spot[c] = pd.to_numeric(spot[c], errors="coerce")
        else:
            spot[c] = np.nan

    keep_cols = [c for c in ["code", "name", "amount", "mv", "turn", "pe", "pb"] if c in spot.columns]
    spot = spot[keep_cols].copy()
    spot = normalize_spot_df(spot)
    if cache_dir:
        ensure_dir(cache_dir)
        spot.to_csv(cache_path(cache_dir, "spot", datetime.today().strftime("%Y%m%d")), index=False, encoding="utf-8-sig")
    return spot


def get_hist(code, start_date, end_date, cache_dir=None):
    code = str(code).zfill(6)
    # adjust 可改：qfq/hfq/None
    hist_cache = cache_path(cache_dir, "hist", code) if cache_dir else None
    if hist_cache and os.path.exists(hist_cache):
        try:
            cached = pd.read_csv(
                hist_cache,
                usecols=["date", "close", "volume", "amount"],
                parse_dates=["date"],
                dtype={"close": "float64", "volume": "float64", "amount": "float64"},
            )
            cached = cached.dropna(subset=["date", "close", "amount"]).sort_values("date")
            if len(cached) >= 120:
                return cached
        except Exception:
            pass

    df = None
    last_err = None
    prefixed = to_prefixed_code(code)
    fetchers = [
        (
            "daily",
            lambda: with_retry(
                ak.stock_zh_a_daily,
                symbol=prefixed,
                start_date=start_date,
                end_date=end_date,
                adjust="qfq",
                retries=3,
                base_sleep=0.4,
            ),
        ),
        (
            "hist_tx",
            lambda: with_retry(
                ak.stock_zh_a_hist_tx,
                symbol=prefixed,
                start_date=start_date,
                end_date=end_date,
                adjust="qfq",
                retries=3,
                base_sleep=0.4,
            ),
        ),
        (
            "hist_em",
            lambda: with_retry(
                ak.stock_zh_a_hist,
                symbol=code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq",
                retries=3,
                base_sleep=0.4,
            ),
        ),
    ]
    for _, fetch in fetchers:
        try:
            df = fetch()
            break
        except Exception as e:
            last_err = e
            continue
    if df is None:
        raise last_err

    date_col = safe_col(df, ["日期", "date"])
    close_col = safe_col(df, ["收盘", "close"])
    vol_col = safe_col(df, ["成交量", "volume"])
    amt_col = safe_col(df, ["成交额", "amount"])
    if any(c is None for c in [date_col, close_col, amt_col]):
        raise ValueError(f"{code} hist字段缺失, columns={list(df.columns)}")

    df = df.rename(columns={date_col: "date", close_col: "close", vol_col: "volume", amt_col: "amount"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "volume" not in df.columns:
        df["volume"] = np.nan
    for c in ["close", "volume", "amount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values("date")
    out = df[["date", "close", "volume", "amount"]].dropna(subset=["date", "close", "amount"])
    if hist_cache and len(out) > 0:
        ensure_dir(cache_dir)
        out.to_csv(hist_cache, index=False, encoding="utf-8-sig")
    return out


# ---------- feature engineering ----------
def calc_features(hist: pd.DataFrame):
    if len(hist) < MIN_HIST_SAMPLES:
        return None
    close = hist["close"].astype(float).values

    # rolling returns
    def r(period):
        if len(close) < period + 1:
            return np.nan
        return close[-1] / close[-(period + 1)] - 1

    mom_20 = r(20)
    mom_60 = r(60)
    mom_120 = r(120)

    # max drawdown (120d)
    window = close[-120:]
    peak = np.maximum.accumulate(window)
    dd = (window / peak - 1).min()  # negative
    dd_120 = float(dd)

    # vol (60d)
    rets = np.diff(close[-61:]) / close[-61:-1]
    vol_60 = float(np.std(rets) * np.sqrt(252))

    # breakout proxy: close vs 120d high
    high_120 = float(np.max(window))
    breakout = float(close[-1] / high_120 - 1)

    # amount spike: last 5d avg / last 60d avg
    amt = hist["amount"].astype(float).values
    avg_amt_20 = amt[-20:].mean() if len(amt) >= 20 else np.nan
    if len(amt) >= 60:
        spike = (amt[-5:].mean() / (amt[-60:].mean() + 1e-9)) - 1
    else:
        spike = np.nan

    return {
        "ret_120": mom_120,
        "mom_20": mom_20,
        "mom_60": mom_60,
        "dd_120": dd_120,
        "vol_60": vol_60,
        "breakout_120": breakout,
        "amt_spike": spike,
        "avg_amt_20": avg_amt_20,
        "latest_date": hist["date"].max(),
    }


# ---------- scoring ----------
def build_scores(df: pd.DataFrame):
    # 标准化
    z_ret = zscore(df["ret_120"])
    z_rs = zscore(df["rs_120"])
    z_dd = zscore(df["dd_120"])  # dd 是负数，越接近 0 越好
    z_m20 = zscore(df["mom_20"])
    z_m60 = zscore(df["mom_60"])
    z_spk = zscore(df["amt_spike"])
    z_brk = zscore(df["breakout_120"])

    # 结构代理：收益/回撤 + 相对强弱 + 稳定性（dd）
    # dd 越接近 0 越好，所以用 -z_dd（因为 dd 更负更差）
    score_structure_proxy = 0.45 * z_ret + 0.35 * z_rs + 0.20 * (-z_dd)

    # 叙事代理：短中动量 + 成交放大 + 逼近/突破新高
    score_narrative_proxy = 0.35 * z_m20 + 0.25 * z_m60 + 0.25 * z_spk + 0.15 * z_brk

    r_value = sigmoid(1.2 * score_structure_proxy - 1.0 * score_narrative_proxy)

    # 映射到 0-100 分（便于排序展示）
    s_score = 50 + 15 * score_structure_proxy
    n_score = 50 + 15 * score_narrative_proxy

    df["R"] = r_value
    df["S_score"] = s_score.clip(0, 100)
    df["N_score"] = n_score.clip(0, 100)
    return df


def main():
    parser = argparse.ArgumentParser(description="SNS 选股脚本")
    parser.add_argument("--max-workers", type=int, default=8, help="并发抓取线程数")
    parser.add_argument("--batch-size", type=int, default=80, help="分批抓取大小")
    parser.add_argument("--sleep-min", type=float, default=0.3, help="批次间最小等待秒数")
    parser.add_argument("--sleep-max", type=float, default=1.0, help="批次间最大等待秒数")
    parser.add_argument("--cache-dir", type=str, default=".cache_sns", help="缓存目录")
    parser.add_argument("--limit-codes", type=int, default=0, help="仅测试前N只股票, 0表示全量")
    parser.add_argument(
        "--index-symbols",
        type=str,
        default="000510,000300,000922,000852",
        help="指数代码列表, 逗号分隔; 默认中证A500+沪深300+中证红利+中证1000",
    )
    args = parser.parse_args()

    end = datetime.today()
    start = end - timedelta(days=520)  # 覆盖 120 交易日+更充足缓冲
    start_date = start.strftime("%Y%m%d")
    end_date = end.strftime("%Y%m%d")

    index_symbols = [x.strip() for x in args.index_symbols.split(",") if x.strip()]
    pool = get_index_constituents(index_symbols=index_symbols, cache_dir=args.cache_dir)
    print(f"Universe from indexes={index_symbols}, unique codes={len(pool)}")

    spot = None
    try:
        spot = get_spot(cache_dir=args.cache_dir)
    except Exception as e:
        # 网络不可达时，自动使用最近一个现货缓存
        latest_spot = None
        if os.path.isdir(args.cache_dir):
            candidates = sorted(
                [os.path.join(args.cache_dir, f) for f in os.listdir(args.cache_dir) if f.startswith("spot_") and f.endswith(".csv")]
            )
            if candidates:
                latest_spot = candidates[-1]
        if latest_spot:
            print(f"Warning: get_spot failed, use cache: {latest_spot}; err={type(e).__name__}")
            spot = pd.read_csv(latest_spot)
            spot = normalize_spot_df(spot)
        else:
            print(f"Warning: get_spot failed and no cache; use index pool only. err={type(e).__name__}")
            spot = None

    # 基础过滤 + 指数股票池约束
    if spot is None:
        spot = pool[["code", "name", "index_name", "index_code"]].drop_duplicates(subset=["code"]).copy()
        for c in ["amount", "mv", "turn", "pe", "pb"]:
            spot[c] = np.nan
    else:
        spot = spot[~spot["name"].astype(str).str.contains("ST|\\*ST|退", na=False)]
        spot = spot.merge(pool[["code"]], on="code", how="inner")
        spot = spot.dropna(subset=["code"]).reset_index(drop=True)
        if "mv" in spot.columns and spot["mv"].notna().any():
            spot = spot[spot["mv"] > 1.5e10]  # 150亿市值阈值
        else:
            print("Warning: mv column unavailable, skip mv filter.")
        if "name" in pool.columns:
            spot = spot.merge(pool[["code", "name", "index_name", "index_code"]], on="code", how="left", suffixes=("", "_pool"))
            spot["name"] = spot["name"].fillna(spot.get("name_pool"))
            if "name_pool" in spot.columns:
                spot = spot.drop(columns=["name_pool"])

    codes = spot["code"].tolist()
    if args.limit_codes and args.limit_codes > 0:
        codes = codes[: args.limit_codes]

    rows = []
    errors = 0
    hist_success_count = 0
    hist_short_count = 0
    total_batches = (len(codes) + args.batch_size - 1) // args.batch_size
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        for batch_idx, i in enumerate(range(0, len(codes), args.batch_size)):
            batch = codes[i : i + args.batch_size]
            futs = {ex.submit(get_hist, c, start_date, end_date, args.cache_dir): c for c in batch}
            batch_errors = 0
            for fut in as_completed(futs):
                c = futs[fut]
                try:
                    hist = fut.result()
                    if len(hist) < MIN_HIST_SAMPLES:
                        hist_short_count += 1
                        continue
                    feats = calc_features(hist)
                    if feats is None:
                        continue
                    rows.append({"code": c, **feats})
                    hist_success_count += 1
                except Exception:
                    errors += 1
                    batch_errors += 1
            # 仅在中间批次且当前批次有错误时等待，减少全量运行固定等待开销
            if batch_idx < total_batches - 1 and batch_errors > 0:
                time.sleep(random.uniform(args.sleep_min, args.sleep_max))

    feat_df = pd.DataFrame(rows)
    if feat_df.empty:
        print(
            f"No features computed. codes={len(codes)}, "
            f"short_count={hist_short_count}, errors={errors}"
        )
        return

    df = spot.merge(feat_df, on="code", how="inner")

    # 相对强弱：用全样本 ret_120 的中位数做基准
    median_ret = df["ret_120"].median()
    df["rs_120"] = df["ret_120"] - median_ret

    # 日均成交额阈值：近20日平均成交额 > 5亿元
    df = df[df["avg_amt_20"] > 5e8]

    df = df.dropna(subset=["ret_120", "mom_20", "mom_60", "dd_120", "breakout_120", "amt_spike", "avg_amt_20", "rs_120"])
    if df.empty:
        print(f"No rows left after NaN filtering. errors={errors}")
        return

    df = build_scores(df)

    # 分流输出
    structure_pool = df[df["R"] >= 0.6].sort_values(["S_score", "R"], ascending=False).head(20)
    narrative_pool = df[df["R"] <= 0.4].sort_values(["N_score"], ascending=False).head(15)

    structure_pool.to_csv("sns_structure_top20.csv", index=False, encoding="utf-8-sig")
    narrative_pool.to_csv("sns_narrative_top15.csv", index=False, encoding="utf-8-sig")
    run_date_str = datetime.today().strftime("%Y%m%d")
    latest_data_date = pd.to_datetime(df["latest_date"], errors="coerce").max()
    latest_data_date_str = latest_data_date.strftime("%Y%m%d") if pd.notna(latest_data_date) else "unknown"
    excel_name = f"sns_result_{run_date_str}_data_{latest_data_date_str}.xlsx"
    with pd.ExcelWriter(excel_name, engine="openpyxl") as writer:
        structure_pool.to_excel(writer, sheet_name="structure", index=False)
        narrative_pool.to_excel(writer, sheet_name="narrative", index=False)

    print("Saved: sns_structure_top20.csv, sns_narrative_top15.csv")
    print(f"Saved: {excel_name}")
    print(
        f"codes={len(codes)}, features={len(feat_df)}, final={len(df)}, errors={errors}, "
        f"hist_success_count={hist_success_count}, hist_short_count={hist_short_count}"
    )


if __name__ == "__main__":
    main()
