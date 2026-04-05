"""
Bot Scanner SAFE — Actions / ETF / Crypto
=========================================
- Analyse une watchlist
- Donne des alertes Telegram
- Profil SAFE :
    * pas d'auto-trading
    * évite les actifs trop tendus
    * cherche des entrées raisonnables

Dépendances :
- requests
- yfinance
- pandas

Secrets GitHub :
- TELEGRAM_BOT_TOKEN
- TELEGRAM_CHAT_ID

Variables GitHub optionnelles :
- MAX_SIGNALS
- CAPITAL
- RISK_PER_TRADE
"""

import os
import math
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

MAX_SIGNALS = int(os.environ.get("MAX_SIGNALS", "5"))
CAPITAL = float(os.environ.get("CAPITAL", "1000"))
RISK_PER_TRADE = float(os.environ.get("RISK_PER_TRADE", "0.01"))  # 1% par trade

# Watchlist SAFE de départ
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

TIMEOUT = 15

# ─────────────────────────────────────────
# OUTILS
# ─────────────────────────────────────────

def log(msg: str) -> None:
    print(msg, flush=True)

def send_telegram(message: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log("Telegram non configuré")
        log(message)
        return False

    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=TIMEOUT,
        )
        log(f"Telegram HTTP {r.status_code}")
        r.raise_for_status()
        return True
    except Exception as e:
        log(f"Erreur Telegram: {e}")
        return False

def safe_round(x, digits=2):
    try:
        return round(float(x), digits)
    except Exception:
        return None

# ─────────────────────────────────────────
# INDICATEURS
# ─────────────────────────────────────────

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

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

    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

# ─────────────────────────────────────────
# ANALYSE SAFE
# ─────────────────────────────────────────

def fetch_data(ticker: str) -> pd.DataFrame | None:
    try:
        df = yf.download(
            ticker,
            period="6mo",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if df is None or df.empty or len(df) < 60:
            return None

        # yfinance peut retourner colonnes multi-index selon versions
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        needed = {"Open", "High", "Low", "Close", "Volume"}
        if not needed.issubset(set(df.columns)):
            return None

        return df.dropna()
    except Exception as e:
        log(f"{ticker} fetch error: {e}")
        return None

def analyze_ticker(ticker: str, asset_type: str) -> dict | None:
    df = fetch_data(ticker)
    if df is None or len(df) < 60:
        return None

    close = df["Close"]
    volume = df["Volume"]

    ema20 = compute_ema(close, 20)
    ema50 = compute_ema(close, 50)
    rsi = compute_rsi(close, 14)
    atr = compute_atr(df, 14)

    last_close = float(close.iloc[-1])
    last_ema20 = float(ema20.iloc[-1])
    last_ema50 = float(ema50.iloc[-1])
    last_rsi = float(rsi.iloc[-1])
    last_atr = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else None
    avg_volume_20 = float(volume.tail(20).mean())

    distance_ema20_pct = ((last_close - last_ema20) / last_ema20) * 100 if last_ema20 else 0
    trend_ok = last_close > last_ema20 > last_ema50
    pullback_ok = -3.5 <= distance_ema20_pct <= 2.0
    rsi_ok = 42 <= last_rsi <= 62

    # Profil SAFE : on évite les actifs trop loin de leur tendance et trop “chauds”
    score = 50
    if trend_ok:
        score += 18
    if pullback_ok:
        score += 16
    if rsi_ok:
        score += 12
    if avg_volume_20 > 1_000_000:
        score += 8
    elif avg_volume_20 > 100_000:
        score += 4

    if distance_ema20_pct > 4:
        score -= 18
    if last_rsi > 68:
        score -= 18
    if last_rsi < 35:
        score -= 8

    score = max(0, min(100, score))

    if score >= 78:
        decision = "🟢 ACHAT SAFE"
    elif score >= 65:
        decision = "🟡 SURVEILLER"
    else:
        decision = "⚪ RIEN"

    stop_loss = None
    risk_eur = round(CAPITAL * RISK_PER_TRADE, 2)

    if last_atr and last_atr > 0:
        stop_loss = round(last_close - (1.5 * last_atr), 2)
        risk_per_unit = max(last_close - stop_loss, 0.01)
        qty = math.floor(risk_eur / risk_per_unit)
    else:
        qty = 0

    return {
        "ticker": ticker,
        "type": asset_type,
        "price": round(last_close, 2),
        "ema20": round(last_ema20, 2),
        "ema50": round(last_ema50, 2),
        "rsi": round(last_rsi, 1),
        "atr": safe_round(last_atr, 2),
        "distance_ema20_pct": round(distance_ema20_pct, 2),
        "avg_volume_20": int(avg_volume_20),
        "score": score,
        "decision": decision,
        "stop_loss": stop_loss,
        "risk_eur": risk_eur,
        "qty": qty,
        "trend_ok": trend_ok,
        "pullback_ok": pullback_ok,
        "rsi_ok": rsi_ok,
    }

def build_signal_message(item: dict) -> str:
    return f"""<b>{item['decision']}</b>

📌 <b>{item['ticker']}</b> ({item['type']})
💵 Prix: <b>{item['price']}</b>
📈 EMA20: {item['ema20']}
📉 EMA50: {item['ema50']}
🧠 RSI: <b>{item['rsi']}</b>
📏 Distance EMA20: <b>{item['distance_ema20_pct']}%</b>
📊 Volume moyen 20j: <b>{item['avg_volume_20']}</b>
⭐ Score SAFE: <b>{item['score']}/100</b>

✅ Tendance OK: {item['trend_ok']}
✅ Repli OK: {item['pullback_ok']}
✅ RSI OK: {item['rsi_ok']}

🛑 Stop indicatif: <b>{item['stop_loss']}</b>
💸 Risque/trade: <b>{item['risk_eur']}€</b>
📦 Taille max indicative: <b>{item['qty']}</b>

⚠️ Aide à la décision uniquement, pas un conseil financier."""

def build_status_message(results: list[dict]) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    buys = sum(1 for r in results if r["decision"] == "🟢 ACHAT SAFE")
    watch = sum(1 for r in results if r["decision"] == "🟡 SURVEILLER")

    return f"""✅ <b>Bot SAFE actif</b>

🕒 Scan: {now}
📦 Actifs analysés: <b>{len(results)}</b>
🟢 Achats safe: <b>{buys}</b>
🟡 À surveiller: <b>{watch}</b>

Règles:
• tendance haussière
• pas trop loin de l'EMA20
• RSI raisonnable
• gestion du risque stricte

Mieux vaut rater un trade que forcer une mauvaise entrée."""

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def run_bot():
    log("=" * 60)
    log("BOT SAFE Actions / ETF / Crypto")
    log("=" * 60)

    results = []
    for item in WATCHLIST:
        log(f"Analyse {item['ticker']}")
        analyzed = analyze_ticker(item["ticker"], item["type"])
        if analyzed:
            results.append(analyzed)

    if not results:
        send_telegram("❌ Le bot n'a pas réussi à analyser la watchlist.")
        return

    ranked = sorted(results, key=lambda x: x["score"], reverse=True)
    top_signals = [r for r in ranked if r["decision"] != "⚪ RIEN"][:MAX_SIGNALS]

    send_telegram(build_status_message(results))

    if top_signals:
        for sig in top_signals:
            send_telegram(build_signal_message(sig))
    else:
        send_telegram("😴 Aucun setup SAFE propre aujourd'hui.")

    log("✅ Fin du scan")

if __name__ == "__main__":
    run_bot()
