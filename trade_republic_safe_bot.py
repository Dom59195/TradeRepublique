"""
Bot Scanner SAFE PRO — Version utile
====================================
- Analyse une watchlist multi-actifs
- Envoie un résumé Telegram cohérent
- Envoie les vrais signaux si présents
- Sinon envoie les 3 meilleurs actifs à surveiller
- Mode SAFE : pas d'auto-trading, sélection prudente

Dépendances :
- requests
- yfinance
- pandas
"""

import os
import math
from datetime import datetime, timezone

import pandas as pd
import requests
import yfinance as yf

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

MAX_SIGNALS = int(os.environ.get("MAX_SIGNALS", "5"))
CAPITAL = float(os.environ.get("CAPITAL", "1000"))
RISK_PER_TRADE = float(os.environ.get("RISK_PER_TRADE", "0.01"))

TIMEOUT = 15

WATCHLIST = [
    {"ticker": "SPY", "type": "ETF"},
    {"ticker": "QQQ", "type": "ETF"},
    {"ticker": "VWCE.DE", "type": "ETF"},
    {"ticker": "CW8.PA", "type": "ETF"},
    {"ticker": "AAPL", "type": "ACTION"},
    {"ticker": "MSFT", "type": "ACTION"},
    {"ticker": "ASML.AS", "type": "ACTION"},
    {"ticker": "MC.PA", "type": "ACTION"},
    {"ticker": "BTC-USD", "type": "CRYPTO"},
    {"ticker": "ETH-USD", "type": "CRYPTO"},
]

# ─────────────────────────────────────────
# OUTILS
# ─────────────────────────────────────────

def log(msg: str) -> None:
    print(msg, flush=True)

def send_telegram(message: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log("❌ Telegram non configuré")
        log(message)
        return False

    try:
        response = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=TIMEOUT,
        )
        log(f"📨 Telegram HTTP {response.status_code}")
        response.raise_for_status()
        return True
    except Exception as e:
        log(f"❌ Erreur Telegram : {e}")
        return False

def round_or_none(value, digits=2):
    try:
        return round(float(value), digits)
    except Exception:
        return None

# ─────────────────────────────────────────
# INDICATEURS
# ─────────────────────────────────────────

def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

# ─────────────────────────────────────────
# DATA
# ─────────────────────────────────────────

def fetch_data(ticker: str) -> pd.DataFrame | None:
    try:
        df = yf.download(
            ticker,
            period="9mo",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )

        if df is None or df.empty or len(df) < 120:
            log(f"⚠️ Pas assez de données pour {ticker}")
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        required = {"Open", "High", "Low", "Close", "Volume"}
        if not required.issubset(set(df.columns)):
            log(f"⚠️ Colonnes manquantes pour {ticker}")
            return None

        return df.dropna()

    except Exception as e:
        log(f"❌ Erreur récupération {ticker} : {e}")
        return None

# ─────────────────────────────────────────
# LOGIQUE SAFE PRO
# ─────────────────────────────────────────

def thresholds_for_asset(asset_type: str) -> dict:
    if asset_type == "CRYPTO":
        return {
            "rsi_min": 42,
            "rsi_max": 62,
            "distance_min": -7.0,
            "distance_max": 5.0,
            "too_hot_distance": 6.5,
            "too_hot_perf_20d": 22.0,
        }

    if asset_type == "ACTION":
        return {
            "rsi_min": 43,
            "rsi_max": 62,
            "distance_min": -5.5,
            "distance_max": 3.5,
            "too_hot_distance": 4.5,
            "too_hot_perf_20d": 16.0,
        }

    return {
        "rsi_min": 44,
        "rsi_max": 62,
        "distance_min": -4.5,
        "distance_max": 3.0,
        "too_hot_distance": 4.0,
        "too_hot_perf_20d": 12.0,
    }

def decide_label(score: int) -> str:
    if score >= 72:
        return "🟢 ACHAT SAFE"
    if score >= 55:
        return "🟡 SURVEILLER"
    if score >= 40:
        return "⚪ RIEN"
    return "🔴 À ÉVITER"

def analyze_ticker(ticker: str, asset_type: str) -> dict | None:
    df = fetch_data(ticker)
    if df is None:
        return None

    close = df["Close"]
    volume = df["Volume"]

    ema20 = compute_ema(close, 20)
    ema50 = compute_ema(close, 50)
    ema100 = compute_ema(close, 100)
    rsi = compute_rsi(close, 14)
    atr = compute_atr(df, 14)

    last_close = float(close.iloc[-1])
    last_ema20 = float(ema20.iloc[-1])
    last_ema50 = float(ema50.iloc[-1])
    last_ema100 = float(ema100.iloc[-1]) if not pd.isna(ema100.iloc[-1]) else None
    last_rsi = float(rsi.iloc[-1])
    last_atr = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else None
    avg_volume_20 = float(volume.tail(20).mean())

    perf_5d = ((close.iloc[-1] / close.iloc[-6]) - 1) * 100 if len(close) >= 6 else 0.0
    perf_20d = ((close.iloc[-1] / close.iloc[-21]) - 1) * 100 if len(close) >= 21 else 0.0
    perf_60d = ((close.iloc[-1] / close.iloc[-61]) - 1) * 100 if len(close) >= 61 else 0.0

    distance_ema20_pct = ((last_close - last_ema20) / last_ema20) * 100 if last_ema20 else 0.0
    distance_ema50_pct = ((last_close - last_ema50) / last_ema50) * 100 if last_ema50 else 0.0

    th = thresholds_for_asset(asset_type)

    trend_ok = (
        last_close > last_ema20 and
        last_ema20 > last_ema50 and
        (last_ema100 is None or last_ema50 > last_ema100)
    )

    pullback_ok = th["distance_min"] <= distance_ema20_pct <= th["distance_max"]
    rsi_ok = th["rsi_min"] <= last_rsi <= th["rsi_max"]

    too_hot = (
        distance_ema20_pct > th["too_hot_distance"] or
        perf_20d > th["too_hot_perf_20d"] or
        last_rsi > 68
    )

    score = 50

    if trend_ok:
        score += 18
    else:
        score -= 16

    if pullback_ok:
        score += 10
    elif distance_ema20_pct > th["distance_max"]:
        score -= 10
    else:
        score -= 6

    if rsi_ok:
        score += 10
    elif last_rsi > 68:
        score -= 12
    elif last_rsi < 36:
        score -= 8
    else:
        score -= 5

    if avg_volume_20 > 20_000_000:
        score += 8
    elif avg_volume_20 > 5_000_000:
        score += 6
    elif avg_volume_20 > 1_000_000:
        score += 4
    elif avg_volume_20 < 100_000:
        score -= 10

    if last_close < last_ema50:
        score -= 10

    if last_close < last_ema20:
        score -= 5

    if too_hot:
        score -= 10

    if perf_5d > 6:
        score -= 6
    elif perf_5d > 3:
        score -= 3

    if 0 < perf_60d < 20:
        score += 4
    elif perf_60d < -12:
        score -= 6

    if trend_ok and rsi_ok and abs(distance_ema20_pct) <= 1.5:
        score += 8

    score = max(0, min(100, round(score)))
    decision = decide_label(score)

    risk_eur = round(CAPITAL * RISK_PER_TRADE, 2)
    stop_loss = None
    qty = 0

    if last_atr and last_atr > 0:
        stop_loss = round(last_close - (1.5 * last_atr), 2)
        risk_per_unit = max(last_close - stop_loss, 0.01)
        qty = math.floor(risk_eur / risk_per_unit)

    risk_note = "OK"
    if qty <= 0:
        risk_note = "taille trop petite / actif trop cher ou trop volatil"

    return {
        "ticker": ticker,
        "type": asset_type,
        "price": round(last_close, 2),
        "ema20": round(last_ema20, 2),
        "ema50": round(last_ema50, 2),
        "ema100": round_or_none(last_ema100, 2),
        "rsi": round(last_rsi, 1),
        "atr": round_or_none(last_atr, 2),
        "distance_ema20_pct": round(distance_ema20_pct, 2),
        "distance_ema50_pct": round(distance_ema50_pct, 2),
        "perf_5d": round(perf_5d, 2),
        "perf_20d": round(perf_20d, 2),
        "perf_60d": round(perf_60d, 2),
        "avg_volume_20": int(avg_volume_20),
        "score": score,
        "decision": decision,
        "stop_loss": stop_loss,
        "risk_eur": risk_eur,
        "qty": qty,
        "risk_note": risk_note,
        "trend_ok": trend_ok,
        "pullback_ok": pullback_ok,
        "rsi_ok": rsi_ok,
        "too_hot": too_hot,
    }

# ─────────────────────────────────────────
# MESSAGES TELEGRAM
# ─────────────────────────────────────────

def build_signal_message(item: dict) -> str:
    tendance = "Oui" if item["trend_ok"] else "Non"
    repli = "Oui" if item["pullback_ok"] else "Non"
    rsi = "Oui" if item["rsi_ok"] else "Non"
    chauffe = "Oui" if item["too_hot"] else "Non"

    return f"""<b>{item['decision']}</b>

📌 <b>{item['ticker']}</b> ({item['type']})
💵 Prix : <b>{item['price']}</b>
📈 EMA20 : {item['ema20']}
📉 EMA50 : {item['ema50']}
🧠 RSI : <b>{item['rsi']}</b>

📏 Distance EMA20 : <b>{item['distance_ema20_pct']}%</b>
⚡ Perf 20j : <b>{item['perf_20d']}%</b>
⚡ Perf 60j : <b>{item['perf_60d']}%</b>
📊 Volume moyen 20j : <b>{item['avg_volume_20']}</b>

⭐ Score SAFE PRO : <b>{item['score']}/100</b>

✅ Tendance OK : {tendance}
✅ Repli OK : {repli}
✅ RSI OK : {rsi}
🔥 Trop étiré : {chauffe}

🛑 Stop indicatif : <b>{item['stop_loss']}</b>
💸 Risque/trade : <b>{item['risk_eur']}€</b>
📦 Taille max indicative : <b>{item['qty']}</b>
📝 Note risque : <b>{item['risk_note']}</b>

⚠️ Aide à la décision uniquement, pas un conseil financier."""

def build_watchlist_message(item: dict) -> str:
    tendance = "Oui" if item["trend_ok"] else "Non"
    repli = "Oui" if item["pullback_ok"] else "Non"
    rsi = "Oui" if item["rsi_ok"] else "Non"
    chauffe = "Oui" if item["too_hot"] else "Non"

    return f"""👀 <b>TOP WATCHLIST</b>

📌 <b>{item['ticker']}</b> ({item['type']})
💵 Prix : <b>{item['price']}</b>
📈 EMA20 : {item['ema20']}
📉 EMA50 : {item['ema50']}
🧠 RSI : <b>{item['rsi']}</b>

📏 Distance EMA20 : <b>{item['distance_ema20_pct']}%</b>
⚡ Perf 20j : <b>{item['perf_20d']}%</b>
⚡ Perf 60j : <b>{item['perf_60d']}%</b>
📊 Volume moyen 20j : <b>{item['avg_volume_20']}</b>

⭐ Score SAFE PRO : <b>{item['score']}/100</b>

✅ Tendance OK : {tendance}
✅ Repli OK : {repli}
✅ RSI OK : {rsi}
🔥 Trop étiré : {chauffe}

🛑 Stop indicatif : <b>{item['stop_loss']}</b>
💸 Risque/trade : <b>{item['risk_eur']}€</b>
📦 Taille max indicative : <b>{item['qty']}</b>
📝 Note risque : <b>{item['risk_note']}</b>

⚠️ Pas un signal d'achat.
C'est juste un des meilleurs actifs à surveiller aujourd'hui."""

def build_status_message(results: list[dict], top_signals: list[dict]) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    buy_count = sum(1 for r in results if r["decision"] == "🟢 ACHAT SAFE")
    watch_count = sum(1 for r in results if r["decision"] == "🟡 SURVEILLER")
    neutral_count = sum(1 for r in results if r["decision"] == "⚪ RIEN")
    avoid_count = sum(1 for r in results if r["decision"] == "🔴 À ÉVITER")

    best_line = ""
    if top_signals:
        best = top_signals[0]
        best_line = f"\n🏆 Meilleur setup : <b>{best['ticker']}</b> ({best['score']}/100)"

    return f"""✅ <b>Bot SAFE PRO actif</b>

🕒 Scan : {now}
📦 Actifs analysés : <b>{len(results)}</b>

🟢 Achats safe : <b>{buy_count}</b>
🟡 À surveiller : <b>{watch_count}</b>
⚪ Neutres : <b>{neutral_count}</b>
🔴 À éviter : <b>{avoid_count}</b>{best_line}

Le bot cherche :
• tendance propre
• entrée pas trop loin de l'EMA20
• RSI raisonnable
• gestion du risque stricte

Mieux vaut rater un trade que forcer une mauvaise entrée."""

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def run_bot():
    log("=" * 60)
    log("🤖 BOT SAFE PRO — Version utile")
    log("=" * 60)

    results = []

    for item in WATCHLIST:
        log(f"🔎 Analyse {item['ticker']}")
        analyzed = analyze_ticker(item["ticker"], item["type"])
        if analyzed:
            results.append(analyzed)
            log(
                f"   -> {analyzed['decision']} | "
                f"score={analyzed['score']} | "
                f"trend={analyzed['trend_ok']} | "
                f"rsi={analyzed['rsi']} | "
                f"dist20={analyzed['distance_ema20_pct']}%"
            )

    if not results:
        send_telegram("❌ Le bot n'a pas réussi à analyser la watchlist.")
        return

    ranked = sorted(results, key=lambda x: x["score"], reverse=True)

    top_signals = [
        r for r in ranked
        if r["decision"] in ("🟢 ACHAT SAFE", "🟡 SURVEILLER")
    ][:MAX_SIGNALS]

    send_telegram(build_status_message(results, top_signals))

    if top_signals:
        for sig in top_signals:
            send_telegram(build_signal_message(sig))
    else:
        send_telegram("😴 Aucun setup SAFE PRO propre aujourd'hui.")
        for item in ranked[:3]:
            send_telegram(build_watchlist_message(item))

    log("✅ Fin du scan")

if __name__ == "__main__":
    run_bot()
