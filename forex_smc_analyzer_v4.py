# ================================================================
# FOREX WEEKLY-SWING SMC ANALYZER  (v4)
# Twelve Data + Gemini | Google Colab | One-cell implementation
# ================================================================
# Alur: D1 bias -> H4 dealing range + POI (OB/FVG) -> tap POI
#       -> H1 CHoCH/MSS setelah tap -> entry WAJIB di OB H1 -> TP di likuiditas
# Hanya untuk edukasi. Bukan nasihat keuangan.
# API key tidak dicetak dan tidak disimpan di laporan.
#
# Perubahan v4 dibanding v3 (ringkas):
#  1. Urutan waktu: POI siap (close candle break H4) -> tap (episode tap
#     dihitung dari candle H1) -> candle H1 konfirmasi SETELAH tap -> order.
#  2. build_structure: displacement hanya dihitung dari candle searah;
#     setiap event menyimpan protected level baru.
#  3. dealing_range: satu leg (low terendah -> high setelahnya), wajib
#     memuat level break; menyimpan index leg.
#  4. detect_order_blocks: OB harus diikuti displacement yang keluar dari
#     OB sebelum/saat break; ukuran OB dibatasi (OB_MAX_ATR).
#  5. Candle konfirmasi H1: searah, close melewati level minimal margin ATR.
#  6. Jalur harga: scan_path (fill/SL/TP/candle ambigu) dipakai live & backtest.
#  7. RR dihitung SEBELUM biaya trading (spread, swap, slippage tidak
#     dihitung).
#  8. Syarat wajib dipisah dari skor kualitas (skor 0-10 tanpa poin gratis).
#  9. Target likuiditas: masih utuh di H1, di sisi benar, jarak masuk akal,
#     level berdekatan digabung.
# 10. Equal highs/lows: pemisahan antar anggota cluster, sisi harga benar,
#     jarak maksimum.
# 11. Backtest walk-forward tanpa look-ahead (opsional).
# ================================================================

import sys
import subprocess
import importlib.util

for _pkg in ("pandas", "requests", "numpy"):
    if importlib.util.find_spec(_pkg) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", _pkg])

import json
import time
import getpass
import requests
import numpy as np
import pandas as pd

from datetime import datetime, timezone

import html as _html

try:
    from IPython.display import display, Markdown, HTML
except Exception:  # di luar Colab
    def display(x):
        print(x)

    def Markdown(x):
        return x

    def HTML(x):
        return x

# -------------------- CONFIGURATION --------------------

TIMEFRAMES = {
    "D1": {"interval": "1day", "hours": 24},
    "H4": {"interval": "4h", "hours": 4},
    "H1": {"interval": "1h", "hours": 1},
}

OUTPUTSIZE = 500
SWING = {"D1": (3, 3), "H4": (3, 3), "H1": (3, 3)}   # (left, right) pivot
# swing    = CHoCH hanya jika close menembus level terlindungi (struktur utama)
# internal = CHoCH/MSS dari pivot terakhir (struktur internal, untuk H1)
STRUCTURE_MODE = {"D1": "swing", "H4": "swing", "H1": "internal"}
ATR_PERIOD = 14

# --- aturan risiko ---
MIN_RR = 2.0                 # RR minimum (sebelum biaya trading)
MAX_RISK_PERCENT = 1.0       # plafon risiko per trade (edukasi)
MIN_SCORE = 7                # aturan pribadi: skor kualitas >= 7/10

# --- kesegaran data (jam PASAR BUKA; akhir pekan tidak dihitung) ---
MAX_DATA_AGE_HOURS = {"D1": 48, "H4": 8, "H1": 3}

# --- parameter SMC ---
DISPLACEMENT_ATR = 0.80      # body minimum untuk displacement
FVG_MIN_ATR = 0.10           # FVG lebih kecil dari ini dianggap noise
EQ_TOLERANCE_ATR = 0.10      # toleransi equal high/low
EQ_LOOKBACK = 100            # equal high/low hanya dari pivot <= 100 candle terakhir
EQ_MIN_SEPARATION = 5        # jarak minimum (candle) antar pivot dalam satu cluster
EQ_MAX_DISTANCE_ATR = 12.0   # equal high/low terlalu jauh dari harga = tidak relevan
SWEEP_MIN_ATR = 0.05         # penetrasi minimum agar dihitung sweep (bukan noise)
OB_LOOKBACK = 5              # cari candle OB maks. 5 candle sebelum origin leg
OB_MAX_ATR = 2.5             # OB lebih lebar dari ini (dalam ATR) dibuang
MIN_RANGE_ATR = 2.0          # dealing range minimal 2 ATR agar bermakna
POI_MAX_AGE_H4 = 120         # umur maksimum POI (candle H4, dari candle OB/FVG)
TAP_MAX_AGE_H4 = 30          # tap POI harus cukup baru (setara candle H4)
TAP_MAX_AGE_H1 = TAP_MAX_AGE_H4 * 4
TAP_GAP_H1 = 4               # jeda > 4 candle H1 di luar zona = episode tap baru
WATCH_DISTANCE_ATR_H4 = 3.0  # POI belum tersentuh dianggap "dipantau" bila <= ini
H1_CONFIRM_MAX_AGE = 24      # konfirmasi H1 maksimal 24 candle lalu
CONFIRM_MIN_CLOSE_ATR = 0.05 # close konfirmasi harus melewati level minimal segini (ATR H1)
SL_BUFFER_ATR_H1 = 0.15      # buffer SL di luar extreme struktur
MIN_STOP_ATR_H1 = 0.50       # SL minimal 0.5 ATR H1 agar tidak kena noise
MAX_STOP_ATR_H4 = 2.00       # SL maksimal 2 ATR H4 (terlalu lebar = bukan setup rapi)
CHASE_LIMIT_R = 1.0          # harga sudah > 1R menjauh dari entry = jangan dikejar
TARGET_MERGE_ATR_H4 = 0.25   # level target berdekatan digabung
TARGET_MIN_ATR_H1 = 1.0      # target minimal 1 ATR H1 dari entry
TARGET_MAX_ATR_H4 = 15.0     # target maksimal 15 ATR H4 dari entry

# --- backtest ---
BT_DEFAULT_BARS = 1000       # jumlah candle H1 terakhir yang diuji
BT_MAX_BARS = 4000
BT_ORDER_EXPIRY_H1 = 24      # limit order kedaluwarsa bila tidak terisi dalam 24 candle H1

# --- tampilan ---
USE_GEMINI = False   # False: hanya kartu sinyal dari kode (tanpa Gemini, tanpa API key Gemini)

TD_URL = "https://api.twelvedata.com/time_series"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

TWELVE_DATA_KEY = ""
GEMINI_KEY = ""


def sanitize(text):
    """Hapus API key dari teks error sebelum dicetak/disimpan."""
    s = str(text)
    for k in (TWELVE_DATA_KEY, GEMINI_KEY):
        if k:
            s = s.replace(k, "***")
    return s


# -------------------- PAIR & BIAYA --------------------

def pip_size(pair):
    base, quote = pair.split("/")
    if base == "XAU":
        return 0.1
    if base == "XAG":
        return 0.01
    if quote == "JPY":
        return 0.01
    return 0.0001


# -------------------- JAM PASAR --------------------

def _friday_close_on_or_before(ts):
    """Penutupan pasar FX kira-kira Jumat 22:00 UTC (bergeser 1 jam saat DST)."""
    base = ts.normalize()
    days_back = (base.weekday() - 4) % 7
    fri = base - pd.Timedelta(days=days_back) + pd.Timedelta(hours=22)
    if fri > ts:
        fri -= pd.Timedelta(days=7)
    return fri


def market_closed_hours(start, end):
    """Jam penutupan akhir pekan (Jum 22:00 -> Min 22:00 UTC) di antara start..end."""
    if end <= start:
        return 0.0
    total = 0.0
    w = _friday_close_on_or_before(start)
    while w < end:
        w_end = w + pd.Timedelta(hours=48)
        overlap = (min(end, w_end) - max(start, w)).total_seconds() / 3600
        total += max(0.0, overlap)
        w += pd.Timedelta(days=7)
    return total


def market_is_closed_now(now):
    w = _friday_close_on_or_before(now)
    return w <= now < w + pd.Timedelta(hours=48)


def freshness(tf, close_est, as_of):
    raw_age = (as_of - close_est).total_seconds() / 3600
    closed_h = market_closed_hours(close_est, as_of)
    eff_age = max(0.0, raw_age - closed_h)
    return raw_age, eff_age, bool(eff_age <= MAX_DATA_AGE_HOURS[tf])


# -------------------- FETCH & BERSIHKAN CANDLE --------------------

def clean_candles(df, interval):
    """
    Validasi & normalisasi. Timestamp = waktu BUKA candle (UTC). Kolom
    close_time_est = waktu tutup perkiraan. TIDAK membuang candle yang belum
    tertutup; gunakan closed_slice(df, as_of) untuk itu.
    """
    required = {"datetime", "open", "high", "low", "close"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"Kolom candle tidak lengkap: {list(df.columns)}")

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=["datetime", "open", "high", "low", "close"])
          .sort_values("datetime")
          .drop_duplicates(subset=["datetime"], keep="last")
          .reset_index(drop=True)
    )
    if len(df) < 80:
        raise RuntimeError(f"Data {interval} terlalu sedikit: {len(df)} candle.")

    invalid = (
        (df["high"] < df["low"]) |
        (df["high"] < df[["open", "close"]].max(axis=1)) |
        (df["low"] > df[["open", "close"]].min(axis=1))
    )
    if invalid.any():
        raise RuntimeError(f"Data OHLC tidak konsisten pada interval {interval}.")

    hours = next(c["hours"] for c in TIMEFRAMES.values() if c["interval"] == interval)
    df["close_time_est"] = df["datetime"] + pd.Timedelta(hours=hours)
    return df[["datetime", "open", "high", "low", "close", "close_time_est"]]


def fetch_raw(pair, interval, outputsize):
    params = {
        "symbol": pair,
        "interval": interval,
        "outputsize": outputsize,
        "timezone": "UTC",
    }
    headers = {"Authorization": f"apikey {TWELVE_DATA_KEY}"}  # key tidak masuk URL

    payload, status = None, None
    for attempt in range(2):
        try:
            response = requests.get(TD_URL, params=params, headers=headers, timeout=60)
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Koneksi Twelve Data gagal ({type(exc).__name__})."
            ) from None

        status = response.status_code
        try:
            payload = response.json()
        except Exception:
            raise RuntimeError(
                f"Twelve Data mengembalikan respons non-JSON (HTTP {status})."
            )

        # Batas kredit per menit: tunggu sebentar lalu coba sekali lagi.
        if payload.get("code") == 429 and attempt == 0:
            time.sleep(20)
            continue
        break

    if status != 200 or "values" not in payload:
        message = sanitize(payload.get("message", "Respons data tidak lengkap."))
        raise RuntimeError(
            f"Twelve Data gagal untuk interval {interval}: HTTP {status}; {message}"
        )
    return clean_candles(pd.DataFrame(payload["values"]), interval)


def closed_slice(df, as_of, limit=OUTPUTSIZE):
    """Hanya candle yang SUDAH TUTUP pada as_of; ambil maksimal 'limit' terakhir."""
    cnt = int(df["close_time_est"].searchsorted(as_of, side="right"))
    return df.iloc[max(0, cnt - limit):cnt].reset_index(drop=True)


# -------------------- INDICATORS --------------------

def add_indicators(df):
    d = df.copy()
    prev_close = d["close"].shift(1)
    tr = pd.concat(
        [
            d["high"] - d["low"],
            (d["high"] - prev_close).abs(),
            (d["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    # ATR Wilder (standar), bukan rata-rata sederhana
    d["atr"] = tr.ewm(alpha=1 / ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean()
    return d


# -------------------- PIVOT / SWING --------------------

def find_pivots(H, L, left, right):
    """
    Pivot high: high[j] adalah yang tertinggi di window (left+right) dan
    kemunculan PERTAMA dari nilai itu. Pivot baru 'diketahui' pada candle
    j+right (tidak ada look-ahead; build_structure memakainya dengan jeda itu).
    """
    n = len(H)
    ph = np.zeros(n, dtype=bool)
    pl = np.zeros(n, dtype=bool)
    for j in range(left, n - right):
        wh = H[j - left:j + right + 1]
        wl = L[j - left:j + right + 1]
        if H[j] == wh.max() and int(np.argmax(wh)) == left:
            ph[j] = True
        if L[j] == wl.min() and int(np.argmin(wl)) == left:
            pl[j] = True
    return ph, pl


# -------------------- STRUKTUR: BOS / CHoCH --------------------

def build_structure(d, left, right, mode="swing"):
    """
    Model struktur (close-based), dua mode:

    mode="swing" (D1/H4):
      - Break searah tren = BOS: close menembus pivot searah terakhir yang
        belum dipecah.
      - Break berlawanan tren = CHoCH hanya jika close menembus LEVEL
        TERLINDUNGI (protected low pada tren naik, protected high pada tren
        turun). Pivot minor/internal tidak membalik bias.

    mode="internal" (H1):
      - Level yang dipecah = pivot high/low TERAKHIR yang belum dipecah,
        searah maupun berlawanan. Break berlawanan tren = CHoCH/MSS internal.

    Setiap event menyimpan 'origin' leg (low terendah untuk break naik, high
    tertinggi untuk break turun, dihitung dari level yang dipecah sampai
    candle break) sebagai dasar OB dan dealing range, serta level terlindungi
    BARU sesudah event ('protected_after'). Displacement = ada candle SEARAH
    dengan body >= DISPLACEMENT_ATR x ATR di dalam leg.
    """
    n = len(d)
    O = d["open"].to_numpy()
    H = d["high"].to_numpy()
    L = d["low"].to_numpy()
    C = d["close"].to_numpy()
    ATR = d["atr"].to_numpy()
    T = d["datetime"]

    ph, pl = find_pivots(H, L, left, right)

    events = []
    top = None
    bot = None
    trend = 0
    protected = None   # level terlindungi tren saat ini

    for i in range(n):
        j = i - right
        if j >= left:
            if ph[j]:
                top = {"price": float(H[j]), "index": j, "crossed": False}
            if pl[j]:
                bot = {"price": float(L[j]), "index": j, "crossed": False}

        direction = 0
        piv = None
        level_price = None
        level_index = None

        if trend == 0 or mode == "internal":
            if top is not None and not top["crossed"] and C[i] > top["price"]:
                direction, piv = 1, top
            elif bot is not None and not bot["crossed"] and C[i] < bot["price"]:
                direction, piv = -1, bot
        elif trend == 1:
            if top is not None and not top["crossed"] and C[i] > top["price"]:
                direction, piv = 1, top
            elif protected is not None and C[i] < protected["price"]:
                direction = -1
                level_price, level_index = protected["price"], protected["index"]
        else:  # trend == -1
            if bot is not None and not bot["crossed"] and C[i] < bot["price"]:
                direction, piv = -1, bot
            elif protected is not None and C[i] > protected["price"]:
                direction = 1
                level_price, level_index = protected["price"], protected["index"]

        if direction == 0:
            continue

        if piv is not None:
            level_price, level_index = piv["price"], piv["index"]
            piv["crossed"] = True

        if trend == 0:
            etype = "STRUCTURE_BREAK"
        elif direction == trend:
            etype = "BOS"
        else:
            etype = "CHoCH"

        s = int(level_index)
        if direction == 1:
            origin = s + int(np.argmin(L[s:i + 1]))
            bodies = C[origin:i + 1] - O[origin:i + 1]
        else:
            origin = s + int(np.argmax(H[s:i + 1]))
            bodies = O[origin:i + 1] - C[origin:i + 1]

        atr_i = ATR[i]
        body_max = float(np.max(bodies)) if len(bodies) else 0.0
        displacement = bool(
            np.isfinite(atr_i) and atr_i > 0 and body_max >= DISPLACEMENT_ATR * atr_i
        )

        # Level terlindungi baru = origin leg yang baru terbentuk.
        if direction == 1:
            protected = {"price": float(L[origin]), "index": int(origin)}
            if top is not None and C[i] > top["price"]:
                top["crossed"] = True
        else:
            protected = {"price": float(H[origin]), "index": int(origin)}
            if bot is not None and C[i] < bot["price"]:
                bot["crossed"] = True

        events.append({
            "index": int(i),
            "time": T.iloc[i].isoformat(),
            "direction": "bullish" if direction == 1 else "bearish",
            "type": etype,
            "level": float(level_price),
            "level_index": s,
            "level_time": T.iloc[s].isoformat(),
            "origin_index": int(origin),
            "origin_time": T.iloc[origin].isoformat(),
            "close": float(C[i]),
            "displacement": displacement,
            "protected_after": float(protected["price"]),
            "protected_after_time": T.iloc[protected["index"]].isoformat(),
        })
        trend = direction

    # Semua pivot + status likuiditas (intact = belum pernah ditembus wick)
    highs, lows = [], []
    for j in np.where(ph)[0]:
        j = int(j)
        price = float(H[j])
        highs.append({
            "price": price, "index": j, "confirmed_index": j + right,
            "time": T.iloc[j].isoformat(),
            "intact": bool(not np.any(H[j + 1:] > price)),
        })
    for j in np.where(pl)[0]:
        j = int(j)
        price = float(L[j])
        lows.append({
            "price": price, "index": j, "confirmed_index": j + right,
            "time": T.iloc[j].isoformat(),
            "intact": bool(not np.any(L[j + 1:] < price)),
        })

    return {"events": events, "trend": trend, "highs": highs, "lows": lows}


# -------------------- LIQUIDITY SWEEP --------------------

def detect_sweeps(d, highs, lows):
    """
    Sweep = wick menembus pivot yang belum pernah ditembus (penetrasi minimal
    SWEEP_MIN_ATR x ATR) lalu CLOSE kembali di sisi asal pada candle yang sama.
    Penembusan pertama selalu 'memakai' level: bila bukan sweep valid
    (close di luar level = break, atau penetrasi terlalu kecil), level tidak
    dihitung lagi.
    """
    H = d["high"].to_numpy()
    L = d["low"].to_numpy()
    C = d["close"].to_numpy()
    ATR = d["atr"].to_numpy()
    T = d["datetime"]
    n = len(d)
    sweeps = []

    for p in highs:
        for k in range(p["index"] + 1, n):
            if H[k] > p["price"]:
                pen = H[k] - p["price"]
                atr = ATR[k]
                if (C[k] < p["price"] and np.isfinite(atr) and atr > 0
                        and pen >= SWEEP_MIN_ATR * atr):
                    sweeps.append({
                        "index": k, "time": T.iloc[k].isoformat(),
                        "type": "BUY-SIDE LIQUIDITY SWEEP",
                        "implication": "potensi reaksi bearish",
                        "level": p["price"], "level_time": p["time"],
                        "sweep_extreme": float(H[k]),
                        "penetration_atr": round(float(pen / atr), 2),
                    })
                break

    for p in lows:
        for k in range(p["index"] + 1, n):
            if L[k] < p["price"]:
                pen = p["price"] - L[k]
                atr = ATR[k]
                if (C[k] > p["price"] and np.isfinite(atr) and atr > 0
                        and pen >= SWEEP_MIN_ATR * atr):
                    sweeps.append({
                        "index": k, "time": T.iloc[k].isoformat(),
                        "type": "SELL-SIDE LIQUIDITY SWEEP",
                        "implication": "potensi reaksi bullish",
                        "level": p["price"], "level_time": p["time"],
                        "sweep_extreme": float(L[k]),
                        "penetration_atr": round(float(pen / atr), 2),
                    })
                break

    sweeps.sort(key=lambda s: s["index"])
    return sweeps


# -------------------- EQUAL HIGHS / LOWS --------------------

def detect_equal_levels(d, highs, lows):
    """
    Equal highs/lows = >= 2 pivot intact yang berdekatan harga (<= tol dari
    anggota terendah cluster), dalam EQ_LOOKBACK candle terakhir. Anggota yang
    dihitung harus terpisah >= EQ_MIN_SEPARATION candle (anggota yang terlalu
    rapat dibuang, bukan seluruh cluster). Toleransi memakai median ATR 50
    candle terakhir. Level likuiditas = extreme cluster (tertinggi untuk highs,
    terendah untuk lows). Hanya yang relevan: berada di sisi harga yang benar
    dan tidak lebih jauh dari EQ_MAX_DISTANCE_ATR x ATR dari harga.
    """
    atr_ref = d["atr"].iloc[-50:].median()
    atr_now = d["atr"].iloc[-1]
    if pd.isna(atr_ref) or atr_ref <= 0 or pd.isna(atr_now) or atr_now <= 0:
        return {"equal_highs": [], "equal_lows": []}
    tol = float(atr_ref) * EQ_TOLERANCE_ATR
    price = float(d["close"].iloc[-1])
    n = len(d)

    def cluster(pivots, agg, is_high):
        ps = sorted(
            [p for p in pivots if p["intact"] and (n - 1 - p["index"]) <= EQ_LOOKBACK],
            key=lambda p: p["price"],
        )
        groups, group = [], []
        for p in ps:
            if group and p["price"] - group[0]["price"] <= tol:
                group.append(p)
            else:
                if len(group) >= 2:
                    groups.append(group)
                group = [p]
        if len(group) >= 2:
            groups.append(group)

        result = []
        for g in groups:
            g = sorted(g, key=lambda x: x["index"])
            kept = [g[0]]
            for x in g[1:]:
                if x["index"] - kept[-1]["index"] >= EQ_MIN_SEPARATION:
                    kept.append(x)
            if len(kept) < 2:
                continue
            level = float(agg(x["price"] for x in kept))
            if is_high and level <= price:
                continue
            if (not is_high) and level >= price:
                continue
            if abs(level - price) > EQ_MAX_DISTANCE_ATR * float(atr_now):
                continue
            result.append({
                "price": level,
                "count": len(kept),
                "times": [x["time"] for x in kept],
            })
        return result

    return {
        "equal_highs": cluster(highs, max, True),
        "equal_lows": cluster(lows, min, False),
    }


# -------------------- ZONA: TRACKING --------------------

def make_zone(kind, direction, lower, upper, formed_index, T, **extra):
    z = {
        "kind": kind,
        "direction": direction,
        "lower": float(lower),
        "upper": float(upper),
        "midpoint": float((lower + upper) / 2),
        "formed_index": int(formed_index),   # saat zona 'terkonfirmasi' (tracking mulai setelahnya)
        "candle_index": int(formed_index),   # candle asal zona (untuk umur); OB menimpa ini
        "formed_at": T.iloc[formed_index].isoformat(),
    }
    z.update(extra)
    return z


def track_zone(z, H, L, C, start):
    """
    Sejak 'start': hitung setiap episode masuk-zona (touch), kedalaman penetrasi
    maksimum (0 = sisi dekat, 1 = sisi jauh), dan invalidasi (CLOSE menembus
    sisi jauh zona). Touch pada candle yang juga menutup menembus zona tetap
    dihitung SEBELUM zona dinyatakan invalid.
    Status: UNTAPPED / TAPPED / INVALIDATED.
    """
    lo, hi = z["lower"], z["upper"]
    width = hi - lo
    bullish = z["direction"] == "bullish"
    touches = []
    in_zone = False
    invalid_idx = None
    max_depth = 0.0

    for m in range(start, len(H)):
        overlap = (L[m] <= hi) and (H[m] >= lo)
        if overlap:
            if not in_zone:
                touches.append(m)
            if width > 0:
                depth = (hi - L[m]) / width if bullish else (H[m] - lo) / width
                max_depth = max(max_depth, min(1.0, max(0.0, depth)))
            else:
                max_depth = 1.0
        in_zone = overlap

        broke = (C[m] < lo) if bullish else (C[m] > hi)
        if broke:
            invalid_idx = m
            break

    z["touch_indices"] = touches
    z["touch_count"] = len(touches)
    z["invalidated_index"] = invalid_idx
    z["max_depth"] = round(float(max_depth), 2)
    z["mean_threshold_hit"] = bool(max_depth >= 0.5)
    if invalid_idx is not None:
        z["status"] = "INVALIDATED"
    elif touches:
        z["status"] = "TAPPED"
    else:
        z["status"] = "UNTAPPED"
    return z


# -------------------- FVG --------------------

def detect_fvgs(d):
    H = d["high"].to_numpy()
    L = d["low"].to_numpy()
    O = d["open"].to_numpy()
    C = d["close"].to_numpy()
    ATR = d["atr"].to_numpy()
    T = d["datetime"]
    zones = []

    for k in range(2, len(d)):
        if L[k] > H[k - 2]:
            direction, lower, upper = "bullish", H[k - 2], L[k]
        elif H[k] < L[k - 2]:
            direction, lower, upper = "bearish", H[k], L[k - 2]
        else:
            continue

        atr = ATR[k - 1]
        if not np.isfinite(atr) or atr <= 0:
            continue
        if (upper - lower) < FVG_MIN_ATR * atr:
            continue

        disp = bool(abs(C[k - 1] - O[k - 1]) >= DISPLACEMENT_ATR * atr)
        z = make_zone("FVG", direction, lower, upper, k, T,
                      has_imbalance=disp, candle_index=k - 1)
        track_zone(z, H, L, C, start=k + 1)
        zones.append(z)

    return zones


# -------------------- ORDER BLOCKS --------------------

def _leg_has_fvg(H, L, start, end, direction):
    for k in range(max(2, start + 2), end + 1):
        if direction == "bullish" and L[k] > H[k - 2]:
            return True
        if direction == "bearish" and H[k] < L[k - 2]:
            return True
    return False


def detect_order_blocks(d, events):
    """
    OB = candle berlawanan TERAKHIR di (atau tepat sebelum) origin leg yang
    menghasilkan break struktur ber-displacement.
      - Break naik  -> OB bullish = candle bearish terakhir di origin leg.
      - Break turun -> OB bearish = candle bullish terakhir di origin leg.
    Validasi (semua wajib):
      a. Dari candle OB sampai candle break tidak ada CLOSE yang menembus sisi
         jauh OB (low OB untuk bullish, high OB untuk bearish).
      b. Setelah OB (sampai candle break) ada candle SEARAH dengan body >=
         DISPLACEMENT_ATR x ATR yang CLOSE-nya keluar dari OB (melewati high OB
         untuk bullish / low OB untuk bearish): displacement benar-benar
         berasal dari OB, bukan kebetulan ada candle besar di leg.
      c. Lebar OB <= OB_MAX_ATR x ATR (OB terlalu lebar = stop tidak rapi).
    Tanpa fallback: bila tak ada kandidat lolos, event tidak menghasilkan OB.
    Zona dilacak mulai SETELAH break: tap pertama = mitigasi.
    """
    O = d["open"].to_numpy()
    H = d["high"].to_numpy()
    L = d["low"].to_numpy()
    C = d["close"].to_numpy()
    ATR = d["atr"].to_numpy()
    T = d["datetime"]
    obs = {}

    for e in events:
        if not e["displacement"]:
            continue
        o = e["origin_index"]
        ev_i = e["index"]
        bullish = e["direction"] == "bullish"

        ob_idx = None
        disp_idx = None
        for j in range(o, max(-1, o - OB_LOOKBACK - 1), -1):
            opposite = (C[j] < O[j]) if bullish else (C[j] > O[j])
            if not opposite:
                continue
            atr_j = ATR[j]
            if not (np.isfinite(atr_j) and atr_j > 0):
                continue
            if (H[j] - L[j]) > OB_MAX_ATR * atr_j:
                continue
            seg = C[j + 1:ev_i + 1]
            if len(seg) == 0:
                continue
            valid = bool(np.all(seg >= L[j])) if bullish else bool(np.all(seg <= H[j]))
            if not valid:
                continue

            k_found = None
            for k in range(j + 1, ev_i + 1):
                atr_k = ATR[k]
                if not (np.isfinite(atr_k) and atr_k > 0):
                    continue
                body = (C[k] - O[k]) if bullish else (O[k] - C[k])
                beyond = (C[k] > H[j]) if bullish else (C[k] < L[j])
                if body >= DISPLACEMENT_ATR * atr_k and beyond:
                    k_found = k
                    break
            if k_found is None:
                continue
            ob_idx, disp_idx = j, k_found
            break

        if ob_idx is None:
            continue

        key = (e["direction"], ob_idx)
        if key in obs:
            continue

        z = make_zone(
            "ORDER BLOCK", e["direction"], L[ob_idx], H[ob_idx], ev_i, T,
            ob_time=T.iloc[ob_idx].isoformat(),
            candle_index=ob_idx,
            event_index=ev_i,
            displacement_index=int(disp_idx),
            break_type=e["type"],
            has_imbalance=_leg_has_fvg(H, L, ob_idx, ev_i, e["direction"]),
        )
        track_zone(z, H, L, C, start=ev_i + 1)
        obs[key] = z

    return list(obs.values())


# -------------------- DEALING RANGE / PREMIUM-DISCOUNT --------------------

def dealing_range(d, events, direction):
    """
    Satu leg searah 'direction' yang diturunkan dari event struktur terakhir:
      bullish: start = low TERENDAH sejak origin; end = high TERTINGGI SETELAH
               start (urutan low -> high dijamin oleh konstruksi).
      bearish: kebalikannya.
    Equilibrium = 50%. Bullish: discount di bawah EQ. Bearish: premium di atas
    EQ. OTE = 62%-79% retracement.

    Range dinyatakan TIDAK VALID (valid=False) bila:
      - sudah ada CLOSE yang menembus origin leg (leg gagal),
      - leg belum terbentuk (extreme akhir = extreme awal),
      - leg tidak memuat level break event (extreme akhir belum melewati level),
      - tingginya < MIN_RANGE_ATR x ATR.
    Wick yang menembus origin lalu close kembali = sweep; low/high range ikut
    extreme wick itu (swept=True).
    """
    evs = [e for e in events if e["direction"] == direction]
    if not evs:
        return None
    e = evs[-1]
    o = e["origin_index"]
    H = d["high"].to_numpy()
    L = d["low"].to_numpy()
    C = d["close"].to_numpy()
    n = len(d)
    atr = float(d["atr"].iloc[-1])
    price = float(C[-1])

    if direction == "bullish":
        start = o + int(np.argmin(L[o:]))
        end = start + int(np.argmax(H[start:]))
        low, high = float(L[start]), float(H[end])
        origin_price = float(L[o])
        closed_through = bool(np.any(C[o + 1:] < origin_price))
        swept = bool(low < origin_price)
        covers = bool(high >= e["level"])
    else:
        start = o + int(np.argmax(H[o:]))
        end = start + int(np.argmin(L[start:]))
        high, low = float(H[start]), float(L[end])
        origin_price = float(H[o])
        closed_through = bool(np.any(C[o + 1:] > origin_price))
        swept = bool(high > origin_price)
        covers = bool(low <= e["level"])

    r = max(high - low, 0.0)
    reason = None
    if r <= 0:
        reason = "Dealing range H4 tidak valid (high <= low)."
    elif closed_through:
        reason = "Origin leg H4 sudah ditembus close; leg searah D1 gagal."
    elif end == start:
        reason = "Leg H4 belum terbentuk setelah extreme awal."
    elif not covers:
        reason = "Leg H4 terakhir tidak memuat level break event struktur."
    elif np.isfinite(atr) and atr > 0 and r < MIN_RANGE_ATR * atr:
        reason = f"Dealing range H4 terlalu kecil (< {MIN_RANGE_ATR:.1f} ATR)."

    eq = low + r / 2
    if direction == "bullish":
        ote = (high - 0.79 * r, high - 0.62 * r)
    else:
        ote = (low + 0.62 * r, low + 0.79 * r)

    return {
        "direction": direction,
        "valid": reason is None,
        "invalid_reason": reason,
        "low": low,
        "high": high,
        "equilibrium": float(eq),
        "price_zone": "DISCOUNT" if price < eq else "PREMIUM",
        "ote_lower": float(ote[0]),
        "ote_upper": float(ote[1]),
        "start_index": int(start),
        "end_index": int(end),
        "leg_age_candles": int(n - 1 - end),
        "origin_time": e["origin_time"],
        "origin_swept_after_leg": swept,
        "based_on_event": f'{e["type"]} {e["time"]}',
    }


# -------------------- ANALISIS PER TIMEFRAME --------------------

def public_zone(z):
    return {
        "kind": z["kind"],
        "direction": z["direction"],
        "lower": z["lower"],
        "upper": z["upper"],
        "midpoint": z["midpoint"],
        "formed_at": z["formed_at"],
        "age_candles": z.get("age_candles"),
        "status": z["status"],
        "touch_count": z["touch_count"],
        "max_depth": z.get("max_depth"),
        "mean_threshold_hit": z.get("mean_threshold_hit"),
        "has_imbalance": z.get("has_imbalance", False),
    }


def public_event(e):
    return {
        k: e[k] for k in
        ("time", "direction", "type", "level", "level_time", "origin_time",
         "close", "displacement", "protected_after", "protected_after_time")
    }


def analyze_timeframe(df, timeframe, as_of=None):
    """df = candle TERTUTUP saja (closed_slice). as_of = waktu acuan (live: sekarang)."""
    left, right = SWING[timeframe]
    d = add_indicators(df)
    st = build_structure(d, left, right, STRUCTURE_MODE[timeframe])
    events = st["events"]
    sweeps = detect_sweeps(d, st["highs"], st["lows"])
    equal_levels = detect_equal_levels(d, st["highs"], st["lows"])
    fvgs = detect_fvgs(d)
    obs = detect_order_blocks(d, events)
    zones = obs + fvgs

    # Umur zona = jumlah candle sejak candle OB/FVG (bukan sejak break).
    n = len(d)
    for z in zones:
        z["age_candles"] = int(n - 1 - z["candle_index"])

    last = d.iloc[-1]
    ref_time = as_of if as_of is not None else pd.Timestamp.now(tz="UTC")
    raw_age, eff_age, fresh = freshness(timeframe, last["close_time_est"], ref_time)

    trend = st["trend"]
    bias = "BULLISH" if trend == 1 else "BEARISH" if trend == -1 else "NEUTRAL / UNCONFIRMED"

    bias_dir = "bullish" if trend == 1 else "bearish" if trend == -1 else None
    rng = dealing_range(d, events, bias_dir) if bias_dir else None

    active_zones = sorted(
        [z for z in zones if z["status"] != "INVALIDATED"],
        key=lambda z: z["formed_index"],
    )
    active_obs = [public_zone(z) for z in active_zones if z["kind"] == "ORDER BLOCK"][-8:]
    active_fvgs = [public_zone(z) for z in active_zones if z["kind"] == "FVG"][-8:]

    return {
        "timeframe": timeframe,
        "structure_mode": STRUCTURE_MODE[timeframe],
        "candle_count_closed": int(len(d)),
        "last_candle_open_utc": last["datetime"].isoformat(),
        "last_candle_close_est_utc": last["close_time_est"].isoformat(),
        "hours_since_estimated_close": round(raw_age, 2),
        "effective_age_hours_market_open": round(eff_age, 2),
        "data_fresh": fresh,
        "ohlc": {
            "open": float(last["open"]), "high": float(last["high"]),
            "low": float(last["low"]), "close": float(last["close"]),
        },
        "atr": None if pd.isna(last["atr"]) else float(last["atr"]),
        "bias": bias,
        "latest_structure_event": public_event(events[-1]) if events else None,
        "recent_structure_events": [public_event(e) for e in events[-8:]],
        "recent_liquidity_sweeps": [
            {k: v for k, v in s.items() if k != "index"} for s in sweeps[-8:]
        ],
        "intact_swing_highs": [p for p in st["highs"] if p["intact"]][-8:],
        "intact_swing_lows": [p for p in st["lows"] if p["intact"]][-8:],
        "equal_levels": equal_levels,
        "dealing_range_bias_direction": rng,
        "active_order_blocks": active_obs,
        "active_fvgs": active_fvgs,
        # --- internal (dibuang dari laporan publik) ---
        "_df": d,
        "_events": events,
        "_zones": zones,
        "_highs": st["highs"],
        "_lows": st["lows"],
        "_sweeps": sweeps,
        "_equal": equal_levels,
    }


def refresh_age(report, tf, as_of):
    """Salinan laporan dengan kesegaran dihitung ulang terhadap as_of (untuk backtest/cache)."""
    r = dict(report)
    close_est = pd.Timestamp(r["last_candle_close_est_utc"])
    raw_age, eff_age, fresh = freshness(tf, close_est, as_of)
    r["hours_since_estimated_close"] = round(raw_age, 2)
    r["effective_age_hours_market_open"] = round(eff_age, 2)
    r["data_fresh"] = fresh
    return r


# -------------------- HELPER GATE --------------------

def tap_episodes(h1d, z, start_time):
    """
    Indeks H1 awal setiap episode tap pada zona SETELAH start_time (zona siap).
    Candle H1 yang overlap zona dan berjarak <= TAP_GAP_H1 candle satu sama
    lain dianggap satu episode.
    """
    lo, hi = z["lower"], z["upper"]
    H = h1d["high"].to_numpy()
    L = h1d["low"].to_numpy()
    s = int(h1d["datetime"].searchsorted(start_time, side="left"))
    if s >= len(H):
        return []
    mask = (L[s:] <= hi) & (H[s:] >= lo)
    idx = np.where(mask)[0] + s
    episodes = []
    for m in idx:
        m = int(m)
        if not episodes or m - episodes[-1][-1] > TAP_GAP_H1:
            episodes.append([m])
        else:
            episodes[-1].append(m)
    return [ep[0] for ep in episodes]


def scan_path(H, L, start, is_long, entry, sl, tp):
    """
    Pindai candle dari 'start': kapan entry tersentuh, SL tersentuh, TP
    tersentuh, dan candle AMBIGU (SL dan TP tersentuh dalam satu candle, urutan
    tidak bisa diketahui dari OHLC).
    """
    res = {"fill_idx": None, "sl_idx": None, "tp_idx": None, "ambiguous_idx": None}
    for m in range(start, len(H)):
        if is_long:
            f, s, t = L[m] <= entry, L[m] <= sl, H[m] >= tp
        else:
            f, s, t = H[m] >= entry, H[m] >= sl, L[m] <= tp
        if f and res["fill_idx"] is None:
            res["fill_idx"] = m
        if s and t and res["ambiguous_idx"] is None:
            res["ambiguous_idx"] = m
        if s and res["sl_idx"] is None:
            res["sl_idx"] = m
        if t and res["tp_idx"] is None:
            res["tp_idx"] = m
    return res


def level_intact_h1(h1d, level, since_time, is_high):
    """Level likuiditas masih utuh menurut data H1 sejak since_time (wick belum menembus)."""
    s = int(h1d["datetime"].searchsorted(since_time, side="left"))
    s = min(s, len(h1d))
    if is_high:
        return not bool(np.any(h1d["high"].to_numpy()[s:] > level))
    return not bool(np.any(h1d["low"].to_numpy()[s:] < level))


# -------------------- GATE SETUP --------------------

def evaluate_setup(d1, h4, h1, pair):
    """
    Model entry SMC top-down. SYARAT WAJIB (gagal satu = bukan setup):
      1. Data D1/H4/H1 segar; bias D1 terkonfirmasi.
      2. H4: dealing range VALID (satu leg searah D1). POI (OB/FVG searah D1)
         di DISCOUNT (long) / PREMIUM (short), belum invalid.
      3. Urutan waktu: POI siap -> tap POI (episode H1) -> candle konfirmasi H1.
      4. Konfirmasi: CHoCH/MSS H1 ber-displacement searah D1, candle searah,
         close melewati level >= CONFIRM_MIN_CLOSE_ATR, origin SETELAH tap,
         tanpa close H1 menembus sisi jauh POI.
      5. Entry WAJIB di OB H1 dari event konfirmasi (mean threshold); SL di
         luar extreme tap->konfirmasi; lebar SL wajar.
      6. TP di likuiditas yang masih utuh di depan harga & entry; RR >= MIN_RR
         (dihitung sebelum biaya trading).
      7. Jalur harga sejak konfirmasi bersih: entry belum tersentuh, SL/TP
         belum tersentuh, tidak ada candle ambigu, harga belum menjauh > 1R.
    SKOR KUALITAS (0-10) hanya menilai faktor tambahan, tanpa poin gratis.
    Status: VALID UNTUK DIEVALUASI / TUNGGU / TIDAK ADA SETUP.
    """

    def out(status, reasons, **extra):
        r = {"status": status, "reasons": reasons}
        r.update(extra)
        return r

    stale = [k for k, r in (("D1", d1), ("H4", h4), ("H1", h1)) if not r["data_fresh"]]
    if stale:
        return out("TIDAK ADA SETUP", [f"Data {', '.join(stale)} tidak segar."])

    if d1["bias"] not in ("BULLISH", "BEARISH"):
        return out("TIDAK ADA SETUP", ["Bias struktur D1 belum terkonfirmasi."])

    direction = "bullish" if d1["bias"] == "BULLISH" else "bearish"
    is_long = direction == "bullish"

    h4d, h1d = h4["_df"], h1["_df"]
    n4, n1 = len(h4d), len(h1d)
    price = float(h1d["close"].iloc[-1])
    atr_h1 = float(h1d["atr"].iloc[-1])
    atr_h4 = float(h4d["atr"].iloc[-1])
    if not (np.isfinite(atr_h1) and np.isfinite(atr_h4) and atr_h1 > 0 and atr_h4 > 0):
        return out("TIDAK ADA SETUP", ["ATR tidak valid."])

    rng = dealing_range(h4d, h4["_events"], direction)
    if rng is None:
        return out("TIDAK ADA SETUP",
                   ["Belum ada leg H4 searah bias D1 untuk dijadikan dealing range."])
    if not rng["valid"]:
        return out("TIDAK ADA SETUP", [rng["invalid_reason"]], dealing_range_h4=rng)
    eq = rng["equilibrium"]

    zones = []
    for z in h4["_zones"]:
        if z["direction"] != direction or z["status"] == "INVALIDATED":
            continue
        if n4 - 1 - z["candle_index"] > POI_MAX_AGE_H4:
            continue
        mid = z["midpoint"]
        in_pd = (rng["low"] <= mid <= eq) if is_long else (eq <= mid <= rng["high"])
        if in_pd:
            zones.append(z)

    if not zones:
        return out(
            "TIDAK ADA SETUP",
            ["Tidak ada OB/FVG H4 aktif searah D1 di zona "
             f"{'discount' if is_long else 'premium'} dealing range H4."],
            dealing_range_h4=rng,
        )

    H1_O = h1d["open"].to_numpy()
    H1_H = h1d["high"].to_numpy()
    H1_L = h1d["low"].to_numpy()
    H1_C = h1d["close"].to_numpy()

    # ---------- zona yang sudah di-tap (episode H1 SETELAH zona siap) ----------
    tapped = []
    for z in zones:
        ready = h4d["close_time_est"].iloc[z["formed_index"]]
        eps = tap_episodes(h1d, z, ready)
        if not eps:
            continue
        last_tap = eps[-1]
        if n1 - 1 - last_tap <= TAP_MAX_AGE_H1:
            tapped.append((last_tap, len(eps), z))
    tapped.sort(key=lambda x: (x[0], x[2]["kind"] == "ORDER BLOCK"), reverse=True)

    first_failure = None

    for tap_h1, tap_count, z in tapped:
        tap_time = h1d["datetime"].iloc[tap_h1]

        # ----- konfirmasi H1: CHoCH/MSS ber-displacement SETELAH tap -----
        valid_evs = []
        for e in h1["_events"]:
            if e["direction"] != direction:
                continue
            if e["type"] not in ("CHoCH", "STRUCTURE_BREAK") or not e["displacement"]:
                continue
            i = e["index"]
            if i <= tap_h1 or e["origin_index"] < tap_h1:
                continue
            if (n1 - 1 - i) > H1_CONFIRM_MAX_AGE:
                continue
            # candle konfirmasi: searah + close melewati level dengan margin
            body_ok = (H1_C[i] > H1_O[i]) if is_long else (H1_C[i] < H1_O[i])
            atr_i = float(h1d["atr"].iloc[i])
            margin_ok = bool(
                np.isfinite(atr_i) and abs(H1_C[i] - e["level"]) >= CONFIRM_MIN_CLOSE_ATR * atr_i
            )
            if not (body_ok and margin_ok):
                continue
            seg = H1_C[tap_h1:i + 1]
            if is_long and np.any(seg < z["lower"]):
                continue   # POI sudah ditembus close H1 sebelum konfirmasi
            if (not is_long) and np.any(seg > z["upper"]):
                continue
            valid_evs.append(e)
        if not valid_evs:
            continue
        ev = valid_evs[-1]
        ev_i = ev["index"]

        # ----- entry: WAJIB OB H1 dari event konfirmasi -----
        h1_ob = next(
            (o for o in h1["_zones"]
             if o["kind"] == "ORDER BLOCK" and o.get("event_index") == ev_i),
            None,
        )
        if h1_ob is None:
            first_failure = first_failure or [
                "Event konfirmasi H1 ada, tetapi OB H1 valid tidak ditemukan; "
                "entry wajib di OB H1 (tidak ada fallback ke POI H4)."
            ]
            continue
        if h1_ob["status"] == "INVALIDATED":
            first_failure = first_failure or [
                "OB H1 sudah invalid (close menembus sisi jauh OB)."
            ]
            continue

        entry = float(h1_ob["midpoint"])
        in_half = (entry <= eq) if is_long else (entry >= eq)
        if not in_half:
            first_failure = first_failure or [
                "OB H1 berada di luar sisi discount/premium dealing range H4."
            ]
            continue
        overlap = bool(h1_ob["lower"] <= z["upper"] and h1_ob["upper"] >= z["lower"])

        # ----- SL: di luar extreme tap -> konfirmasi (+ OB H1) -----
        seg_lo = H1_L[tap_h1:ev_i + 1]
        seg_hi = H1_H[tap_h1:ev_i + 1]
        buffer = SL_BUFFER_ATR_H1 * atr_h1
        if is_long:
            extreme = min(float(seg_lo.min()), h1_ob["lower"])
            sl = extreme - buffer
            sl = min(sl, entry - MIN_STOP_ATR_H1 * atr_h1)
            risk = entry - sl
        else:
            extreme = max(float(seg_hi.max()), h1_ob["upper"])
            sl = extreme + buffer
            sl = max(sl, entry + MIN_STOP_ATR_H1 * atr_h1)
            risk = sl - entry

        if risk <= 0:
            first_failure = first_failure or ["Jarak entry-SL tidak valid."]
            continue
        if risk > MAX_STOP_ATR_H4 * atr_h4:
            first_failure = first_failure or [
                f"Jarak SL terlalu lebar (> {MAX_STOP_ATR_H4:.1f} ATR H4)."
            ]
            continue

        # ----- target likuiditas: utuh, di depan entry DAN harga, jarak wajar -----
        ref = max(entry, price) if is_long else min(entry, price)
        raw_liq = []
        for src, label in ((h4, "H4"), (d1, "D1")):
            pivots = src["_highs"] if is_long else src["_lows"]
            for p in pivots:
                if not p["intact"]:
                    continue
                if not ((p["price"] > ref) if is_long else (p["price"] < ref)):
                    continue
                if not level_intact_h1(h1d, p["price"], pd.Timestamp(p["time"]), is_long):
                    continue
                raw_liq.append({"price": p["price"],
                                "label": f"Swing {'high' if is_long else 'low'} {label}"})
            for q in src["_equal"]["equal_highs" if is_long else "equal_lows"]:
                if not ((q["price"] > ref) if is_long else (q["price"] < ref)):
                    continue
                if not level_intact_h1(h1d, q["price"], pd.Timestamp(min(q["times"])), is_long):
                    continue
                raw_liq.append({"price": q["price"],
                                "label": f"Equal {'highs' if is_long else 'lows'} {label}"})

        raw_liq = [
            x for x in raw_liq
            if TARGET_MIN_ATR_H1 * atr_h1 <= abs(x["price"] - entry) <= TARGET_MAX_ATR_H4 * atr_h4
        ]
        raw_liq.sort(key=lambda x: abs(x["price"] - entry))
        liq = []
        for x in raw_liq:   # gabungkan level yang berdekatan
            if liq and abs(x["price"] - liq[-1]["price"]) <= TARGET_MERGE_ATR_H4 * atr_h4:
                if x["label"] not in liq[-1]["label"]:
                    liq[-1]["label"] += f" + {x['label']}"
                continue
            liq.append(dict(x))

        tp = None
        for idx, cand in enumerate(liq):
            reward = abs(cand["price"] - entry)
            rr_net = reward / risk
            if rr_net >= MIN_RR:
                tp = {
                    "price": float(cand["price"]), "label": cand["label"],
                    "rr_gross": round(rr_net, 2),
                    "liquidity_in_path": [
                        {"price": float(x["price"]), "label": x["label"]} for x in liq[:idx]
                    ],
                }
                break

        if tp is None:
            first_failure = first_failure or [
                f"Tidak ada target likuiditas utuh di depan harga dengan RR >= {MIN_RR:.1f}:1."
            ]
            continue

        # ----- validasi jalur harga sejak konfirmasi -----
        path = scan_path(H1_H, H1_L, ev_i + 1, is_long, entry, sl, tp["price"])
        if path["ambiguous_idx"] is not None:
            first_failure = first_failure or [
                "Ada candle H1 yang menyentuh SL dan TP sekaligus sejak konfirmasi "
                "(urutan tidak bisa dipastikan); setup dianggap tidak bersih."
            ]
            continue
        if path["sl_idx"] is not None:
            first_failure = first_failure or [
                "Sejak konfirmasi H1, harga sudah menyentuh level SL; ide gugur."
            ]
            continue
        if path["tp_idx"] is not None:
            first_failure = first_failure or [
                "Sejak konfirmasi H1, harga sudah menyentuh target; setup kedaluwarsa."
            ]
            continue
        if path["fill_idx"] is not None:
            first_failure = first_failure or [
                "Level entry (mean threshold OB H1) sudah tersentuh sejak konfirmasi; "
                "ini bukan entri baru, jangan dikejar."
            ]
            continue
        moved = (price - entry) if is_long else (entry - price)
        if moved > CHASE_LIMIT_R * risk:
            first_failure = first_failure or [
                f"Harga sudah menjauh > {CHASE_LIMIT_R:.0f}R dari entry tanpa retest; jangan dikejar."
            ]
            continue

        # ----- skor kualitas (maks 10, tanpa poin gratis) -----
        sweep_type = "SELL-SIDE LIQUIDITY SWEEP" if is_long else "BUY-SIDE LIQUIDITY SWEEP"
        ev_time = pd.Timestamp(ev["time"])
        sweep_ok = any(
            s["type"] == sweep_type and tap_time <= pd.Timestamp(s["time"]) <= ev_time
            for s in (h1["_sweeps"] + h4["_sweeps"])
        )
        leg_ok = bool(
            rng["start_index"] - OB_LOOKBACK <= z["candle_index"] <= rng["end_index"]
        )
        ote_ok = bool(rng["ote_lower"] <= entry <= rng["ote_upper"])
        checks = [
            ("Bias struktur H4 searah D1", 2, h4["bias"] == d1["bias"]),
            ("POI berasal dari leg dealing range H4", 1, leg_ok),
            ("POI punya imbalance/displacement", 1, bool(z.get("has_imbalance"))),
            ("Tap pertama pada POI (1 episode tap)", 1, tap_count == 1),
            ("Liquidity sweep searah antara tap dan konfirmasi", 2, sweep_ok),
            ("OB H1 overlap dengan POI H4", 1, overlap),
            ("Entry di zona OTE (62-79%) range H4", 1, ote_ok),
            ("RR >= 3:1", 1, tp["rr_gross"] >= 3.0),
        ]
        score = sum(pts for _, pts, ok in checks if ok)
        breakdown = [
            {"kriteria": name, "poin": pts, "terpenuhi": bool(ok)} for name, pts, ok in checks
        ]
        mandatory = [
            "Data D1/H4/H1 segar dan bias D1 terkonfirmasi",
            "Dealing range H4 valid (satu leg) dan POI di discount/premium",
            "Urutan waktu: POI siap -> tap -> konfirmasi H1 setelah tap",
            "CHoCH/MSS H1 ber-displacement, candle searah, margin close cukup",
            "Entry di OB H1 dari event konfirmasi",
            "SL wajar dan RR >= minimum",
            "Jalur harga bersih (entry/SL/TP belum tersentuh, tidak ada candle ambigu)",
        ]

        entry_note = (
            "Limit di OB H1 (mean threshold); belum tersentuh sejak konfirmasi."
            if ((is_long and price > entry) or ((not is_long) and price < entry))
            else "Harga berada di/dekat level entry; evaluasi manual, jangan dikejar."
        )

        plan = {
            "direction": "BUY" if is_long else "SELL",
            "entry_reference": entry,
            "entry_basis": "Mean threshold OB H1",
            "stop_loss": float(sl),
            "take_profit": tp["price"],
            "take_profit_basis": tp["label"],
            "risk_distance": float(risk),
            "rr_gross": tp["rr_gross"],
            "liquidity_in_path": tp["liquidity_in_path"],
            "risk_percent_ceiling": MAX_RISK_PERCENT,
            "entry_note": entry_note,
            "invalidation": (
                f"Close H1 {'di bawah' if is_long else 'di atas'} {sl:.5f} "
                "meniadakan ide ini."
            ),
        }
        side = "bullish" if is_long else "bearish"
        bias_txt = f"Bias D1/H4 {side}" if h4["bias"] == d1["bias"] else f"Bias D1 {side} (H4 belum searah)"
        choch_txt = "MSS" if ev["type"] == "STRUCTURE_BREAK" else "CHoCH"
        signal_reason = (
            f"{bias_txt}, POI H4 tersentuh, lalu {choch_txt} disertai displacement H1. "
            f"Entry pada OB H1 dengan target likuiditas di {'atas' if is_long else 'bawah'} harga."
        )
        evidence = {
            "signal_reason": signal_reason,
            "poi_h4": public_zone(z),
            "poi_tap_count_h1": tap_count,
            "poi_last_tap_time": tap_time.isoformat(),
            "h1_confirmation": public_event(ev),
            "h1_order_block": public_zone(h1_ob),
            "dealing_range_h4": rng,
            "mandatory_checks_passed": mandatory,
            "score": score,
            "score_breakdown": breakdown,
        }

        if score >= MIN_SCORE:
            return out(
                "VALID UNTUK DIEVALUASI",
                [
                    "Data D1/H4/H1 segar.",
                    f"Bias D1 {d1['bias']}; POI H4 di {'discount' if is_long else 'premium'}.",
                    "POI sudah di-tap, lalu H1 membentuk CHoCH/MSS searah setelah tap.",
                    f"RR {tp['rr_gross']}:1 (sebelum biaya trading); "
                    f"skor kualitas {score}/10.",
                ],
                plan=plan, **evidence,
            )
        return out(
            "TUNGGU",
            [f"Skor kualitas {score}/10 di bawah ambang {MIN_SCORE}/10 (aturan pribadi)."],
            plan_reference_only=plan, **evidence,
        )

    if first_failure:
        return out("TIDAK ADA SETUP", first_failure, dealing_range_h4=rng)

    # ---------- POI sudah di-tap tapi H1 belum konfirmasi ----------
    if tapped:
        tap_h1, tap_count, z = tapped[0]
        return out(
            "TUNGGU",
            [
                "POI H4 sudah di-tap, tetapi belum ada CHoCH/MSS H1 ber-displacement "
                "searah yang origin-nya terbentuk setelah tap."
            ],
            poi_h4=public_zone(z),
            poi_last_tap_time=h1d["datetime"].iloc[tap_h1].isoformat(),
            syarat_konfirmasi=(
                f"Close H1 {'di atas' if is_long else 'di bawah'} pivot H1 terakhir "
                "setelah sweep/reaksi dari POI."
            ),
            dealing_range_h4=rng,
        )

    # ---------- POI belum di-tap: pantau ----------
    pending = []
    for z in zones:
        ready = h4d["close_time_est"].iloc[z["formed_index"]]
        if tap_episodes(h1d, z, ready):
            continue   # sudah di-tap tetapi tap-nya terlalu lama
        dist = max(0.0, price - z["upper"]) if is_long else max(0.0, z["lower"] - price)
        pending.append((dist, z))
    if pending:
        pending.sort(key=lambda x: (x[0], x[1]["kind"] != "ORDER BLOCK"))
        dist, z = pending[0]
        dist_atr = dist / atr_h4
        if dist_atr <= WATCH_DISTANCE_ATR_H4:
            return out(
                "TUNGGU",
                [
                    f"POI H4 terdekat belum di-tap (jarak {dist_atr:.2f} ATR H4). "
                    "Tunggu tap, lalu CHoCH/MSS H1 searah."
                ],
                poi_h4=public_zone(z),
                distance_atr_h4=round(dist_atr, 2),
                dealing_range_h4=rng,
            )
        return out(
            "TIDAK ADA SETUP",
            [f"POI H4 terdekat masih jauh ({dist_atr:.2f} ATR H4)."],
            nearest_poi_h4=public_zone(z),
            dealing_range_h4=rng,
        )

    return out("TIDAK ADA SETUP",
               ["Tidak ada POI H4 yang memenuhi syarat (tap lama / sudah invalid)."],
               dealing_range_h4=rng)


# -------------------- BACKTEST WALK-FORWARD --------------------

def simulate_trade(h1_all, t, plan, is_long):
    """
    Simulasi limit order dari keputusan di CLOSE candle t (data sampai t saja
    yang dipakai gate). Aturan konservatif untuk candle ambigu:
      - Sebelum fill: TP tersentuh tanpa entry tersentuh = order batal.
      - Candle fill: bila SL ikut tersentuh -> dihitung SL. TP di candle fill
        diabaikan (urutan tidak diketahui).
      - Setelah fill: SL dan TP di candle sama -> dihitung SL (ambiguous).
      - Order kedaluwarsa bila tidak terisi dalam BT_ORDER_EXPIRY_H1 candle.
    Hasil dalam R (sebelum biaya trading): menang = reward/risk, kalah = -1.
    """
    H = h1_all["high"].to_numpy()
    L = h1_all["low"].to_numpy()
    n = len(H)
    entry = plan["entry_reference"]
    sl = plan["stop_loss"]
    tp = plan["take_profit"]
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    last_pending = t + BT_ORDER_EXPIRY_H1

    fill, exit_idx, outcome, note, ambiguous = None, None, None, "", False
    end_idx = n - 1

    for m in range(t + 1, n):
        if is_long:
            touched, sl_hit, tp_hit = L[m] <= entry, L[m] <= sl, H[m] >= tp
        else:
            touched, sl_hit, tp_hit = H[m] >= entry, H[m] >= sl, L[m] <= tp
        end_idx = m
        if fill is None:
            if m > last_pending:
                outcome = "EXPIRED"
                break
            if touched:
                fill = m
                if sl_hit:
                    outcome, exit_idx = "SL", m
                    note = "SL tersentuh di candle fill (konservatif)"
                    break
                continue
            if tp_hit:
                outcome = "CANCELLED_TP_FIRST"
                break
        else:
            if sl_hit and tp_hit:
                outcome, exit_idx, ambiguous = "SL", m, True
                note = "SL dan TP di candle sama (konservatif: SL)"
                break
            if sl_hit:
                outcome, exit_idx = "SL", m
                break
            if tp_hit:
                outcome, exit_idx = "TP", m
                break

    if outcome is None:
        outcome = "OPEN" if fill is not None else "EXPIRED"
        end_idx = n - 1

    rec = {
        "signal_time": h1_all["close_time_est"].iloc[t].isoformat(),
        "direction": plan["direction"],
        "entry": entry, "sl": sl, "tp": tp,
        "outcome": outcome, "note": note, "ambiguous": ambiguous,
        "fill_time": None if fill is None else h1_all["datetime"].iloc[fill].isoformat(),
        "exit_time": None if exit_idx is None else h1_all["close_time_est"].iloc[exit_idx].isoformat(),
        "r_result": None,
        "rr_plan": plan["rr_gross"], "end_idx": int(end_idx),
    }

    if outcome in ("TP", "SL"):
        rec["r_result"] = round(reward / risk, 3) if outcome == "TP" else -1.0
    return rec


def backtest_stats(trades, bars):
    done = [t for t in trades if t["outcome"] in ("TP", "SL")]
    wins = [t for t in done if t["outcome"] == "TP"]
    losses = [t for t in done if t["outcome"] == "SL"]
    rs = [t["r_result"] for t in done]
    equity, peak, max_dd = 0.0, 0.0, 0.0
    streak, max_streak = 0, 0
    for t in done:
        equity += t["r_result"]
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
        if t["outcome"] == "SL":
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    gross_win = sum(t["r_result"] for t in wins)
    gross_loss = abs(sum(t["r_result"] for t in losses))
    return {
        "bars_tested_h1": int(bars),
        "signals": len(trades),
        "executed": len(done),
        "wins": len(wins),
        "losses": len(losses),
        "expired": sum(t["outcome"] == "EXPIRED" for t in trades),
        "cancelled_tp_first": sum(t["outcome"] == "CANCELLED_TP_FIRST" for t in trades),
        "still_open": sum(t["outcome"] == "OPEN" for t in trades),
        "ambiguous_candles": sum(bool(t["ambiguous"]) for t in trades),
        "win_rate_percent": round(100 * len(wins) / len(done), 1) if done else None,
        "avg_r": round(float(np.mean(rs)), 3) if rs else None,
        "expectancy_r": round(float(np.mean(rs)), 3) if rs else None,
        "total_r": round(float(sum(rs)), 2) if rs else 0.0,
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
        "max_drawdown_r": round(float(max_dd), 2),
        "max_consecutive_losses": int(max_streak),
        "sample_warning": (
            "Sampel < 30 trade: statistik tidak signifikan." if len(done) < 30 else None
        ),
    }


def run_backtest(raws, pair, bt_bars, progress=True):
    """
    Walk-forward tanpa look-ahead: pada setiap close H1, D1/H4/H1 dipotong
    hanya sampai candle yang sudah tertutup pada waktu itu, lalu gate yang
    SAMA dengan live dijalankan. Satu posisi/order pada satu waktu.
    raws = {"D1","H4","H1"}: DataFrame candle TERTUTUP (clean_candles + closed_slice).
    """
    d1_all, h4_all, h1_all = raws["D1"], raws["H4"], raws["H1"]
    n_all = len(h1_all)
    start_t = max(n_all - int(bt_bars), 150)
    cache_d1, cache_h4 = {}, {}
    trades = []
    blocked_until = -1
    total_steps = max(1, n_all - 1 - start_t)
    t0 = time.time()

    for t in range(start_t, n_all - 1):
        if progress and (t - start_t) % 250 == 0:
            done = t - start_t
            print(f"  backtest {done}/{total_steps} candle H1 | sinyal: {len(trades)} | "
                  f"{time.time() - t0:.0f} dtk")
        if t <= blocked_until:
            continue

        as_of = h1_all["close_time_est"].iloc[t]
        c1 = int(d1_all["close_time_est"].searchsorted(as_of, side="right"))
        c4 = int(h4_all["close_time_est"].searchsorted(as_of, side="right"))
        if c1 < 60 or c4 < 60:
            continue

        if c1 not in cache_d1:
            cache_d1[c1] = analyze_timeframe(
                d1_all.iloc[max(0, c1 - OUTPUTSIZE):c1].reset_index(drop=True), "D1", as_of)
        d1r = refresh_age(cache_d1[c1], "D1", as_of)
        if d1r["bias"] not in ("BULLISH", "BEARISH") or not d1r["data_fresh"]:
            continue

        if c4 not in cache_h4:
            cache_h4[c4] = analyze_timeframe(
                h4_all.iloc[max(0, c4 - OUTPUTSIZE):c4].reset_index(drop=True), "H4", as_of)
        h4r = refresh_age(cache_h4[c4], "H4", as_of)

        h1r = analyze_timeframe(
            h1_all.iloc[max(0, t + 1 - OUTPUTSIZE):t + 1].reset_index(drop=True), "H1", as_of)

        res = evaluate_setup(d1r, h4r, h1r, pair)
        if res["status"] != "VALID UNTUK DIEVALUASI":
            continue

        plan = res["plan"]
        rec = simulate_trade(h1_all, t, plan, plan["direction"] == "BUY")
        rec["score"] = res["score"]
        trades.append(rec)
        blocked_until = rec["end_idx"]

    stats = backtest_stats(trades, n_all - start_t)
    return stats, trades


# -------------------- GEMINI MODEL DISCOVERY --------------------

def discover_gemini_model():
    try:
        response = requests.get(
            f"{GEMINI_BASE}/models",
            headers={"x-goog-api-key": GEMINI_KEY},
            timeout=30,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Koneksi Gemini gagal ({type(exc).__name__}).") from None

    try:
        data = response.json()
    except Exception:
        raise RuntimeError(
            f"Gagal membaca daftar model Gemini (HTTP {response.status_code})."
        )

    if response.status_code != 200:
        raise RuntimeError(
            f"Daftar model Gemini gagal (HTTP {response.status_code}): "
            f"{sanitize(data.get('error', {}).get('message', 'Unknown error'))}"
        )

    blocked = ("image", "tts", "live", "audio", "embedding", "imagen",
               "aqa", "robotics", "computer-use", "native")
    candidates = []
    for model in data.get("models", []):
        name = model.get("name", "")
        low = name.lower()
        if "generateContent" not in model.get("supportedGenerationMethods", []):
            continue
        if "flash" not in low or any(b in low for b in blocked):
            continue
        candidates.append(name)

    if not candidates:
        raise RuntimeError(
            "Tidak ditemukan model Gemini Flash yang mendukung generateContent "
            "pada API key ini."
        )

    def rank(n):
        n = n.lower()
        return (
            0 if n.endswith("/gemini-2.5-flash") else 1,
            0 if "2.5-flash" in n else 1,
            1 if ("preview" in n or "exp" in n) else 0,
            1 if "lite" in n else 0,
            n,
        )

    candidates.sort(key=rank)
    return candidates[0]


# -------------------- GEMINI REPORT --------------------

def call_gemini(model_name, report_payload):
    url = f"{GEMINI_BASE}/{model_name}:generateContent"

    prompt = """
Anda menyusun ringkasan keputusan setup Forex untuk tujuan edukasi, bukan
nasihat keuangan. Gunakan HANYA data JSON yang disediakan. Jangan mengarang
harga, level, berita, atau probabilitas.

Keluarkan PERSIS enam baris berikut dalam Bahasa Indonesia, tanpa teks lain,
tanpa judul, tanpa tabel, tanpa daftar tambahan:

Status: BUY / SELL / NO SIGNAL
Entry: <angka atau ->
Stop Loss (SL): <angka atau ->
Take Profit (TP): <angka atau ->
RR minimal 1:2: <nilai RR 1:X (sebelum biaya trading), atau ->
Alasan: <satu kalimat singkat>

Aturan:
1. Status BUY atau SELL HANYA jika trade_gate.status = "VALID UNTUK DIEVALUASI"
   dan ada field "plan"; arah diambil dari plan.direction. Selain itu status
   = NO SIGNAL (termasuk bila status gate "TUNGGU" atau "TIDAK ADA SETUP").
2. Entry/SL/TP/RR diambil apa adanya dari "plan" (entry_reference,
   stop_loss, take_profit, rr_gross). Pada NO SIGNAL isi keempatnya dengan "-".
   Abaikan "plan_reference_only"; itu bukan rencana.
3. RR dihitung sebelum biaya trading (spread, swap, slippage tidak dihitung).
4. Alasan: satu kalimat singkat dari trade_gate.reasons. Contoh:
   "NO SIGNAL - struktur H4 tidak memenuhi syarat." Pada BUY/SELL, ringkas
   mengapa setup valid dan sebut ini kandidat evaluasi manual, bukan sinyal pasti.
5. NO SIGNAL adalah hasil normal. Jangan menyarankan mengubah aturan SMC.

Data JSON:
""" + json.dumps(report_payload, ensure_ascii=False, default=str)

    generation_config = {"temperature": 0.2, "maxOutputTokens": 8192}
    # Gemini 2.5 Flash menghitung token "thinking" ke dalam maxOutputTokens;
    # matikan agar laporan tidak terpotong/kosong.
    if "2.5-flash" in model_name.lower():
        generation_config["thinkingConfig"] = {"thinkingBudget": 0}

    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": generation_config,
    }

    last_error = None
    for attempt in range(4):
        try:
            response = requests.post(
                url,
                headers={"x-goog-api-key": GEMINI_KEY},
                json=body,
                timeout=90,
            )
            try:
                data = response.json()
            except Exception:
                data = {}

            if response.status_code == 200:
                parts = (
                    data.get("candidates", [{}])[0]
                        .get("content", {})
                        .get("parts", [])
                )
                text = "\n".join(p.get("text", "") for p in parts if p.get("text")).strip()
                if text:
                    return text
                last_error = "Gemini memberi respons tanpa teks."
            else:
                message = data.get("error", {}).get("message", response.text[:500])
                last_error = f"HTTP {response.status_code}: {sanitize(message)}"
                if response.status_code not in (429, 500, 502, 503, 504):
                    break

        except requests.RequestException as exc:
            last_error = f"Network error ({type(exc).__name__})"

        if attempt < 3:
            time.sleep(2 ** (attempt + 1))

    raise RuntimeError(
        "Gemini gagal setelah percobaan ulang. Analisis aturan tetap tersedia. "
        f"Detail: {last_error}"
    )


# -------------------- INPUT PENGGUNA --------------------

# -------------------- KARTU SINYAL --------------------

def price_decimals(pair):
    pip = pip_size(pair)
    return 5 if pip == 0.0001 else 3 if pip == 0.01 else 2


def _first_sentence(text):
    """Ambil kalimat pertama saja agar alasan NO SIGNAL tetap singkat."""
    text = str(text).strip()
    cut = text.find(". ")
    out = text if cut == -1 else text[:cut + 1]
    return out if out.endswith(".") else out + "."


def signal_summary(pair, gate):
    """
    Ringkasan sinyal dari gate (tanpa AI). Aturan SMC TIDAK diubah: hanya
    status VALID UNTUK DIEVALUASI yang menjadi BUY/SELL. "TUNGGU" dan
    "TIDAK ADA SETUP" ditampilkan sebagai NO SIGNAL (harga tidak dipaksakan).
    Semua harga berasal dari data market yang dianalisis.
    """
    plan = gate.get("plan") if gate.get("status") == "VALID UNTUK DIEVALUASI" else None
    if plan:
        dec = price_decimals(pair)
        return {
            "pair": pair, "signal": plan["direction"], "valid": True,
            "entry": f"{plan['entry_reference']:.{dec}f}",
            "stop_loss": f"{plan['stop_loss']:.{dec}f}",
            "take_profit": f"{plan['take_profit']:.{dec}f}",
            "rr": f"1:{plan['rr_gross']:g}",
            "reason": gate.get("signal_reason", ""),
        }
    reasons = gate.get("reasons") or ["Syarat setup tidak terpenuhi."]
    return {"pair": pair, "signal": "NO SIGNAL", "valid": False,
            "reason": _first_sentence(reasons[0])}


def render_signal_card(sig):
    """Teks ringkas (Markdown, satu baris per komponen)."""
    if sig["valid"]:
        lines = [
            f"### {sig['pair']} — {sig['signal']}",
            f"**Entry** {sig['entry']}",
            f"**Stop Loss (SL)** {sig['stop_loss']}",
            f"**Take Profit (TP)** {sig['take_profit']}",
            f"**Risk : Reward** {sig['rr']}",
            f"**Alasan:** {sig['reason']}",
        ]
    else:
        lines = [
            f"### {sig['pair']} — NO SIGNAL",
            f"**Alasan:** {sig['reason']}",
        ]
    return "  \n".join(lines)


def render_signal_html(sig):
    """Kartu HTML: judul, baris label kiri / nilai kanan, garis pemisah, lalu Alasan."""
    e = _html.escape
    box = ("font-family:Roboto,Arial,sans-serif;max-width:460px;margin:8px 0;"
           "padding:18px 20px;border:1px solid rgba(128,128,128,.40);border-radius:16px;"
           "line-height:1.5")
    title = f'<div style="font-size:22px;font-weight:700;margin-bottom:8px">{e(sig["pair"])} &mdash; {e(sig["signal"])}</div>'
    divider = '<div style="border-top:1px solid rgba(128,128,128,.40);margin:12px 0 10px"></div>'
    reason = f'<div><b>Alasan:</b> {e(sig["reason"])}</div>'
    if not sig["valid"]:
        return f'<div style="{box}">{title}{reason}</div>'

    def row(label, value, color=None):
        style = "font-weight:600;font-size:18px" + (f";color:{color}" if color else "")
        return (
            '<div style="display:flex;justify-content:space-between;align-items:baseline;padding:6px 0">'
            f'<span style="opacity:.65;font-size:17px">{label}</span>'
            f'<span style="{style}">{e(str(value))}</span></div>'
        )

    entry_color = "#4caf50" if sig["signal"] == "BUY" else "#ef5350"
    rows = (
        row("Entry", sig["entry"], entry_color)
        + row("Stop Loss", sig["stop_loss"])
        + row("Take Profit", sig["take_profit"])
        + row("Risk : Reward", sig["rr"])
    )
    return f'<div style="{box}">{title}{rows}{divider}{reason}</div>'


# -------------------- RUN PIPELINE --------------------

def main():
    global TWELVE_DATA_KEY, GEMINI_KEY

    print("Masukkan API key. Input disembunyikan dan tidak dicetak.")
    TWELVE_DATA_KEY = getpass.getpass("Twelve Data API key: ").strip()
    if USE_GEMINI:
        GEMINI_KEY = getpass.getpass("Gemini API key: ").strip()
    if not TWELVE_DATA_KEY or (USE_GEMINI and not GEMINI_KEY):
        raise ValueError("API key wajib diisi.")

    pair = input("Pair Forex (contoh EUR/USD): ").strip().upper().replace(" ", "")
    if not pair or "/" not in pair or len(pair.split("/")) != 2:
        raise ValueError("Format pair tidak valid. Gunakan BASE/QUOTE, misalnya EUR/USD.")

    run_bt = input("Jalankan backtest walk-forward? Hasil disimpan ke file (y/N): ").strip().lower() == "y"
    bt_bars = BT_DEFAULT_BARS
    if run_bt:
        raw_bars = input(f"Jumlah candle H1 yang diuji (default {BT_DEFAULT_BARS}, maks {BT_MAX_BARS}): ").strip()
        if raw_bars.isdigit():
            bt_bars = min(max(int(raw_bars), 200), BT_MAX_BARS)

    sizes = {"D1": OUTPUTSIZE, "H4": OUTPUTSIZE, "H1": OUTPUTSIZE}
    if run_bt:
        sizes["H1"] = min(5000, OUTPUTSIZE + bt_bars + 60)
        sizes["H4"] = min(5000, OUTPUTSIZE + bt_bars // 4 + 60)
        sizes["D1"] = min(5000, OUTPUTSIZE + bt_bars // 20 + 40)

    print(f"\nMengambil data {pair} (D1/H4/H1)...")
    now = pd.Timestamp.now(tz="UTC")
    closed = {}
    reports = {}
    for tf, config in TIMEFRAMES.items():
        raw = fetch_raw(pair, config["interval"], sizes[tf])
        closed[tf] = closed_slice(raw, now, limit=len(raw))
        live = closed_slice(raw, now, limit=OUTPUTSIZE)
        if len(live) < 60:
            raise RuntimeError(f"Candle tertutup {tf} tidak cukup untuk analisis.")
        reports[tf] = analyze_timeframe(live, tf, as_of=now)

    market_status = "TUTUP (akhir pekan)" if market_is_closed_now(now) else "BUKA"
    trade_gate = evaluate_setup(reports["D1"], reports["H4"], reports["H1"], pair)

    public_reports = {
        tf: {k: v for k, v in rep.items() if not k.startswith("_")}
        for tf, rep in reports.items()
    }

    payload = {
        "pair": pair,
        "analysis_time_utc": datetime.now(timezone.utc).isoformat(),
        "market_status_estimate": market_status,
        "method": "D1 bias -> H4 dealing range/POI -> tap -> H1 CHoCH/MSS -> entry OB H1",
        "minimum_rr": MIN_RR,
        "minimum_score": MIN_SCORE,
        "risk_percent_ceiling": MAX_RISK_PERCENT,
        "timeframes": public_reports,
        "trade_gate": trade_gate,
        "limitations": [
            "Tidak ada berita/kalender ekonomi yang diambil.",
            "Definisi SMC di sini adalah pendekatan algoritmik, bukan satu-satunya tafsir.",
            "Struktur D1/H4 memakai level terlindungi (swing); H1 memakai pivot internal.",
            "Timestamp OHLC dianggap waktu buka candle (UTC); batas hari D1 mengikuti penyedia data.",
            "Jam penutupan akhir pekan diperkirakan Jumat 22:00 - Minggu 22:00 UTC.",
            "RR dan hasil backtest dihitung SEBELUM biaya trading (spread, swap, slippage tidak dihitung).",
            "Skor 10 poin adalah checklist kualitas aturan kode, bukan probabilitas menang.",
            "Backtest (jika dijalankan) adalah simulasi aturan kode pada data Twelve Data, bukan eksekusi nyata; tanpa tick data, urutan intra-candle tidak diketahui.",
            "Sel ini tidak mengeksekusi order.",
        ],
    }

    # -------------------- KARTU SINYAL --------------------
    signal = signal_summary(pair, trade_gate)
    payload["signal"] = signal
    print()
    try:
        display(HTML(render_signal_html(signal)))
    except Exception:
        display(Markdown(render_signal_card(signal)))

    # -------------------- BACKTEST (hanya ke file) --------------------
    bt_trades = []
    if run_bt:
        print(f"\nBacktest {bt_bars} candle H1 berjalan (bisa beberapa menit)...")
        try:
            stats, bt_trades = run_backtest(closed, pair, bt_bars)
            payload["backtest"] = stats
            print("Backtest selesai; hasilnya ada di file laporan.")
        except Exception as exc:
            print("Backtest gagal:", sanitize(exc))
            payload["backtest_error"] = sanitize(exc)

    # -------------------- GEMINI (opsional) --------------------
    if USE_GEMINI:
        try:
            model_name = discover_gemini_model()
            ai_payload = {k: v for k, v in payload.items()
                          if k not in ("backtest", "signal")}
            ai_report = call_gemini(model_name, ai_payload)
            display(Markdown(ai_report))
            payload["gemini_model"] = model_name
            payload["gemini_report"] = ai_report
        except Exception as exc:
            print("Gemini gagal:", sanitize(exc))
            payload["gemini_error"] = sanitize(exc)

    # -------------------- SAVE REPORT --------------------
    safe_pair = pair.replace("/", "_").replace(" ", "")
    output_file = f"{safe_pair.lower()}_smc_analysis.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print("\nDetail lengkap (tanpa API key):", output_file)

    if bt_trades:
        csv_file = f"{safe_pair.lower()}_smc_backtest_trades.csv"
        pd.DataFrame(bt_trades).drop(columns=["end_idx"], errors="ignore").to_csv(csv_file, index=False)
        print("Trade backtest:", csv_file)


if __name__ == "__main__":
    main()
