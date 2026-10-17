import os
from datetime import datetime, timezone

import streamlit as st
import analyzer as smc

st.set_page_config(page_title="SMC Analyzer v4", page_icon="📈", layout="centered")

st.markdown("""
<style>
.block-container {max-width: 780px; padding-top: 1.7rem; padding-bottom: 3rem;}
[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.28); padding: 14px; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

st.title("📈 Forex SMC Analyzer")
st.caption("Analyzer v4 · D1 bias → H4 POI → H1 CHoCH/MSS → OB H1")
st.info("Analisis hanya berjalan saat kamu menekan tombol. Web ini tidak mengeksekusi order trading.")

# Read secret from Streamlit Cloud Secrets or environment variable; never display it.
try:
    api_key = st.secrets.get("TWELVE_DATA_KEY", "")
except Exception:
    api_key = ""
api_key = api_key or os.getenv("TWELVE_DATA_KEY", "")

# Pilihan simbol. Simbol tetap perlu didukung oleh paket/data Twelve Data.
pairs = {
    "USDJPY": "USD/JPY",
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "XAUUSD (Gold)": "XAU/USD",
    "AUDUSD": "AUD/USD",
    "NZDUSD": "NZD/USD",
    "USDCAD": "USD/CAD",
    "USDCHF": "USD/CHF",
    "EURJPY": "EUR/JPY",
    "GBPJPY": "GBP/JPY",
    "AUDJPY": "AUD/JPY",
    "NZDJPY": "NZD/JPY",
    "EURGBP": "EUR/GBP",
    "EURAUD": "EUR/AUD",
    "GBPAUD": "GBP/AUD",
}

with st.container(border=True):
    st.subheader("Pengaturan analisis")
    choice = st.selectbox("Pilih pair", list(pairs.keys()), index=0)
    pair = pairs[choice]
    st.caption(
        "Timeframe yang dianalisis: D1, H4, dan H1. "
        "Backtest dan laporan tambahan disembunyikan agar fokus pada sinyal."
    )
    run = st.button("🔎 Analisis sekarang", type="primary", use_container_width=True)

if run:
    if not api_key:
        st.error(
            "API key Twelve Data belum dipasang. Tambahkan TWELVE_DATA_KEY "
            "pada App → Settings → Secrets di Streamlit Cloud."
        )
        st.stop()

    smc.TWELVE_DATA_KEY = api_key.strip()
    now = __import__("pandas").Timestamp.now(tz="UTC")
    reports = {}
    status_box = st.status(f"Mengambil data candle untuk {pair}…", expanded=True)

    try:
        for tf, config in smc.TIMEFRAMES.items():
            status_box.write(f"Memproses timeframe {tf}…")
            raw = smc.fetch_raw(pair, config["interval"], smc.OUTPUTSIZE)
            live = smc.closed_slice(raw, now, limit=smc.OUTPUTSIZE)
            if len(live) < 60:
                raise RuntimeError(f"Candle tertutup {tf} tidak cukup untuk analisis.")
            reports[tf] = smc.analyze_timeframe(live, tf, as_of=now)

        gate = smc.evaluate_setup(reports["D1"], reports["H4"], reports["H1"], pair)
        signal = smc.signal_summary(pair, gate)
        status_box.update(label="Analisis selesai", state="complete", expanded=False)

        st.markdown("---")
        st.subheader(f"Hasil: {pair}")
        if signal["valid"]:
            color = "#1f9d55" if signal["signal"] == "BUY" else "#e05252"
            st.markdown(
                f"<div style='font-size:1.65rem;font-weight:800;color:{color};margin-bottom:12px'>"
                f"{signal['signal']}</div>",
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns(2)
            c1.metric("Entry", signal["entry"])
            c2.metric("Stop Loss", signal["stop_loss"])
            c3, c4 = st.columns(2)
            c3.metric("Take Profit", signal["take_profit"])
            c4.metric("Risk : Reward", signal["rr"])
            st.markdown("**Alasan sinyal**")
            st.write(signal["reason"])
        else:
            st.warning("TUNGGU — belum ada setup SMC yang memenuhi seluruh syarat.")
            st.markdown("**Alasan**")
            st.write(signal["reason"])

        with st.expander("Detail pemeriksaan SMC"):
            st.write(f"**Status setup:** {gate.get('status', 'Tidak tersedia')}")
            reasons = gate.get("reasons") or []
            if reasons:
                for reason in reasons:
                    st.write(f"- {reason}")
            st.caption(
                "Skor kualitas adalah checklist aturan kode, bukan probabilitas menang. "
                "Harga dan RR dihitung sebelum biaya trading."
            )

        st.caption("Waktu analisis: " + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    except Exception as exc:
        status_box.update(label="Analisis gagal", state="error", expanded=False)
        st.error(smc.sanitize(str(exc)))
        st.caption(
            "Periksa batas paket/API Twelve Data, simbol pair, dan ketersediaan candle. "
            "API key tidak ditampilkan."
        )

st.markdown("---")
st.caption(
    "Untuk edukasi dan analisis teknikal. Bukan nasihat keuangan atau jaminan hasil trading. "
    "Verifikasi sinyal di chart sebelum mengambil keputusan."
)
