"""
Bot Scanner SAFE PRO — Actions / ETF / Crypto
=============================================
- Analyse une watchlist multi-actifs
- Envoie des alertes Telegram
- Profil SAFE :
    * pas d'auto-trading
    * filtre les actifs trop faibles ou trop tendus
    * score plus strict et cohérent

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
RISK_PER_TRADE = float(os.environ.get("RISK_PER_TRADE", "0.01"))  # 1% du capital max

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
        log(f"📨 Telegram HTTP {r.status_code}")
        r.raise_for_status()
        return True
    except Exception as e:
        log(f"❌ Erreur Telegram : {e}")
        return False

def safe_float(value, default=None):
    try:
        return float(value)
    except Exception:
        return default

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

        if df is None or df.empty or len(df) < 80:
            log(f"⚠️ Pas assez de données pour {ticker}")
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        needed = {"Open", "High", "Low", "Close", "Volume"}
        if not needed.issubset(df.columns):
            log(f"⚠️ Colonnes manquantes pour {ticker}")
            return None

        return df.dropna()

    except Exception as e:
        log(f"❌ Erreur récupération {ticker} : {e}")
        return None

# ─────────────────────────────────────────
# LOGIQUE SAFE PRO
# ─────────────────────────────────────────

def get_thresholds(asset_type: str) -> dict:
    """
    Réglages un peu différents selon le type d'actif.
    """
    if asset_type == "CRYPTO":
        return {
            "max_distance_above_ema20": 4.5,
            "min_distance_below_ema20": -6.0,
            "rsi_buy_min": 43,
            "rsi_buy_max": 60,
        }

    if asset_type == "ACTION":
        return {
            "max_distance_above_ema20": 3.0,
            "min_distance_below_ema20": -5.0,
            "rsi_buy_min": 44,
            "rsi_buy_max": 60,
        }

    # ETF par défaut
    return {
        "max_distance_above_ema20": 2.5,
        "min_distance_below_ema20": -4.0,
        "rsi_buy_min": 45,
        "rsi_buy_max": 60,
    }

def decide_label(score: int) -> str:
    if score >= 82:
        return "🟢 ACHAT SAFE"
    if score >= 68:
        return "🟡 SURVEILLER"
    if score >= 45:
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

    thresholds = get_thresholds(asset_type)

    distance_ema20_pct = ((last_close - last_ema20) / last_ema20) * 100 if last_ema20 else 0.0
    distance_ema50_pct = ((last_close - last_ema50) / last_ema50) * 100 if last_ema50 else 0.0

    trend_ok = (
        last_close > last_ema20 and
        last_ema20 > last_ema50 and
        (last_ema100 is None or last_ema50 > last_ema100)
    )

    pullback_ok = (
        thresholds["min_distance_below_ema20"] <= distance_ema20_pct <= thresholds["max_distance_above_ema20"]
    )

    rsi_ok = thresholds["rsi_buy_min"] <= last_rsi <= thresholds["rsi_buy_max"]

    # Momentum court terme
    perf_5d = ((close.iloc[-1] / close.iloc[-6]) - 1) * 100 if len(close) >= 6 else 0.0
    perf_20d = ((close.iloc[-1] / close.iloc[-21]) - 1) * 100 if len(close) >= 21 else 0.0

    # ─────────────────────────────────────
    # SCORE STRICT
    # ─────────────────────────────────────

    score = 50

    # Tendance = critère principal
    if trend_ok:
        score += 18
    else:
        score -= 25

    # Position vs EMA20
    if pullback_ok:
        score += 12
    else:
        score -= 12

    # RSI
    if rsi_ok:
        score += 10
    else:
        score -= 12

    # Volume
    if avg_volume_20 > 5_000_000:
        score += 6
    elif avg_volume_20 > 1_000_000:
        score += 4
    elif avg_volume_20 < 100_000:
        score -= 10

    # Trop étiré à la hausse = danger
    if distance_ema20_pct > thresholds["max_distance_above_ema20"]:
        score -= 15

    # Trop cassé = danger
    if distance_ema20_pct < thresholds["min_distance_below_ema20"]:
        score -= 10

    # RSI extrêmes
    if last_rsi > 67:
        score -= 16
    elif last_rsi > 63:
        score -= 8

    if last_rsi < 38:
        score -= 8

    # Momentum trop violent = moins safe
    if perf_5d > 6:
        score -= 10
    elif perf_5d > 3:
        score -= 4

    if perf_20d > 12:
        score -= 8

    # Si le prix est sous EMA50, forte pénalité
    if last_close < last_ema50:
        score -= 14

    # Si prix très proche EMA20 + tendance ok + RSI ok = setup plus propre
    if trend_ok and rsi_ok and abs(distance_ema20_pct) <= 1.2:
        score += 8

    score = max(0, min(100, round(score)))
    decision = decide_label(score)

    stop_loss = None
    risk_eur = round(CAPITAL * RISK_PER_TRADE, 2)
    qty = 0

    if last_atr and last_atr > 0:
        stop_loss = round(last_close - (1.5 * last_atr), 2)
        risk_per_unit = max(last_close - stop_loss, 0.01)
        qty = math.floor(risk_eur / risk_per_unit)

    # Si qty = 0, on signale que l'actif est trop cher pour le risque choisi
    risk_note = "OK"
    if qty <= 0:
        risk_note = "taille trop petite / actif trop volatil ou trop cher"

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
    }

# ─────────────────────────────────────────
# TELEGRAM MESSAGES
# ─────────────────────────────────────────

def build_signal_message(item: dict) -> str:
    tendance = "Oui" if item["trend_ok"] else "Non"
    repli = "Oui" if item["pullback_ok"] else "Non"
    rsi = "Oui" if item["rsi_ok"] else "Non"

    return f"""<b>{item['decision']}</b>

📌 <b>{item['ticker']}</b> ({item['type']})
💵 Prix: <b>{item['price']}</b>
📈 EMA20: {item['ema20']}
📉 EMA50: {item['ema50']}
🧠 RSI: <b>{item['rsi']}</b>
📏 Distance EMA20: <b>{item['distance_ema20_pct']}%</b>
📏 Distance EMA50: <b>{item['distance_ema50_pct']}%</b>
⚡ Perf 5j: <b>{item['perf_5d']}%</b>
⚡ Perf 20j: <b>{item['perf_20d']}%</b>
📊 Volume moyen 20j: <b>{item['avg_volume_20']}</b>

⭐ Score SAFE PRO: <b>{item['score']}/100</b>

✅ Tendance OK: {tendance}
✅ Repli OK: {repli}
✅ RSI OK: {rsi}

🛑 Stop indicatif: <b>{item['stop_loss']}</b>
💸 Risque/trade: <b>{item['risk_eur']}€</b>
📦 Taille max indicative: <b>{item['qty']}</b>
📝 Note risque: <b>{item['risk_note']}</b>

⚠️ Aide à la décision uniquement, pas un conseil financier."""

def build_status_message(results: list[dict]) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    buy_count = sum(1 for r in results if r["decision"] == "🟢 ACHAT SAFE")
    watch_count = sum(1 for r in results if r["decision"] == "🟡 SURVEILLER")
    avoid_count = sum(1 for r in results if r["decision"] == "🔴 À ÉVITER")

    return f"""✅ <b>Bot SAFE PRO actif</b>

🕒 Scan: {now}
📦 Actifs analysés: <b>{len(results)}</b>
🟢 Achats safe: <b>{buy_count}</b>
🟡 À surveiller: <b>{watch_count}</b>
🔴 À éviter: <b>{avoid_count}</b>

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
    log("🤖 BOT SAFE PRO — Actions / ETF / Crypto")
    log("=" * 60)

    results = []

    for item in WATCHLIST:
        log(f"🔎 Analyse {item['ticker']}")
        analyzed = analyze_ticker(item["ticker"], item["type"])
        if analyzed:
            results.append(analyzed)

    if not results:
        send_telegram("❌ Le bot n'a pas réussi à analyser la watchlist.")
        return

    ranked = sorted(results, key=lambda x: x["score"], reverse=True)

    # On envoie seulement les actifs intéressants
    top_signals = [
        r for r in ranked
        if r["decision"] in ("🟢 ACHAT SAFE", "🟡 SURVEILLER")
    ][:MAX_SIGNALS]

    send_telegram(build_status_message(results))

    if top_signals:
        for sig in top_signals:
            send_telegram(build_signal_message(sig))
    else:
        send_telegram("😴 Aucun setup SAFE PRO propre aujourd'hui.")

    log("✅ Fin du scan")

if __name__ == "__main__":
    run_bot()
