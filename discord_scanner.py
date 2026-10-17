"""Hourly SMC signal scanner for GitHub Actions. Does not place trades."""
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import analyzer as smc

# Start with the four pairs already present in the current app.py.
# 4 pairs x 3 timeframes x 24 runs = 288 time-series requests/day.
PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "USD/CAD", "AUD/USD", "NZD/USD", "XAU/USD"]
STATE_FILE = Path(".cache/signal_state.json")
REQUEST_PAUSE_SECONDS = 8  # Free Twelve Data tier is listed as 8 API credits/minute.


def send_discord(text: str) -> None:
    webhook = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
    if not webhook:
        raise RuntimeError("GitHub Actions Secret DISCORD_WEBHOOK_URL belum dipasang.")
    # Discord webhook message content has a 2000-character limit.
    response = requests.post(webhook, json={"content": text[:1900]}, timeout=20)
    if not response.ok:
        # Never print the webhook URL; it is a credential.
        raise RuntimeError(f"Discord gagal mengirim pesan (HTTP {response.status_code}).")


def load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def scan_pair(pair: str) -> tuple[dict, dict]:
    now = pd.Timestamp.now(tz="UTC")
    reports = {}
    for tf, config in smc.TIMEFRAMES.items():
        raw = smc.fetch_raw(pair, config["interval"], smc.OUTPUTSIZE)
        live = smc.closed_slice(raw, now, limit=smc.OUTPUTSIZE)
        if len(live) < 60:
            raise RuntimeError(f"Candle tertutup {tf} tidak cukup untuk {pair}.")
        reports[tf] = smc.analyze_timeframe(live, tf, as_of=now)
        time.sleep(REQUEST_PAUSE_SECONDS)

    gate = smc.evaluate_setup(reports["D1"], reports["H4"], reports["H1"], pair)
    signal = smc.signal_summary(pair, gate)
    return signal, gate


def signature(signal: dict) -> str:
    return "|".join(str(signal.get(k, "")) for k in ("pair", "signal", "entry", "stop_loss", "take_profit", "rr"))


def signal_message(signal: dict, reason: str, timestamp: str) -> str:
    return (
        f"FOREX SMC ANALYZER\n"
        f"{signal['pair']} — {signal['signal']}\n\n"
        f"Entry: {signal['entry']}\n"
        f"Stop Loss: {signal['stop_loss']}\n"
        f"Take Profit: {signal['take_profit']}\n"
        f"RR: {signal['rr']}\n\n"
        f"Alasan SMC: {reason or 'Seluruh syarat setup SMC terpenuhi.'}\n\n"
        f"Waktu: {timestamp} UTC\n"
        f"Catatan: sinyal algoritmik; bukan eksekusi order. Periksa harga dan validitas setup sebelum entry."
    )


def main() -> None:
    api_key = os.environ.get("TWELVE_DATA_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GitHub Actions Secret TWELVE_DATA_KEY belum dipasang.")
    smc.TWELVE_DATA_KEY = api_key

    state = load_state()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    new_signals = []
    failures = []

    for pair in PAIRS:
        try:
            signal, gate = scan_pair(pair)
            if not signal.get("valid"):
                continue
            plan = gate.get("plan") or {}
            rr = float(plan.get("rr_gross", 0))
            if rr < float(smc.MIN_RR):
                continue
            sig_key = signature(signal)
            # Avoid sending the exact same plan every hour.
            if state.get(sig_key):
                continue
            reason = signal.get("reason") or "; ".join(gate.get("reasons") or [])
            new_signals.append(signal_message(signal, reason, now))
            state[sig_key] = now
        except Exception as exc:
            # Keep errors short and never expose secrets.
            failures.append(f"{pair}: {smc.sanitize(str(exc))[:220]}")

    # Keep recent signal keys only to prevent an ever-growing state file.
    state = dict(list(state.items())[-200:])
    save_state(state)

    if new_signals:
        for message in new_signals:
            send_discord(message)
    elif os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch":
        send_discord(
            "FOREX SMC ANALYZER — TEST PEMINDAIAN\n"
            f"Pemindaian selesai pada {now} UTC. Belum ada setup baru yang memenuhi seluruh syarat SMC.\n"
            "Jika tidak ada error di GitHub Actions, koneksi Discord dan pemindai berjalan."
        )

    if failures:
        # Keep Discord quiet unless there is a valid new BUY/SELL signal.
        print("Scan errors (secrets are not printed): " + " | ".join(failures[:8]))
        raise RuntimeError(f"Pemindaian selesai dengan {len(failures)} error; lihat log Actions.")

    print(f"Scan complete: {len(PAIRS)} pairs; {len(new_signals)} new signal(s); {len(failures)} error(s).")


if __name__ == "__main__":
    main()
