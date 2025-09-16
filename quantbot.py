import os
import math
import asyncio
import random
import time
from datetime import datetime
from typing import Dict, Tuple, List, Optional, Any
from zoneinfo import ZoneInfo

import discord
from discord.ext import commands, tasks
from discord import app_commands
from dotenv import load_dotenv

import pandas as pd
import yfinance as yf
import aiosqlite

# =============================
# CONFIG & CONSTANTS
# =============================
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
SIGNALS_CHANNEL_ID = os.getenv("SIGNALS_CHANNEL_ID")  # optional explicit channel id
SIGNALS_CHANNEL_NAME = os.getenv("SIGNALS_CHANNEL_NAME", "signals")  # auto-create if missing
GUILD_ID = None  # set to your server id to speed up command sync

# Features / flags
ENABLE_PAPER = os.getenv("ENABLE_PAPER", "0").strip() == "1"  # OFF by default

# India market hours (IST)
IST_TZ = "Asia/Kolkata"
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MIN = 0
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MIN = 30
REPORT_INTERVAL_MIN = 30  # post every 30 minutes

# API keys
FINNHUB_TOKEN = os.getenv("FINNHUB_TOKEN")
MARKETAUX_TOKEN = os.getenv("MARKETAUX_TOKEN")

# News polling (conserve MarketAux quota)
NEWS_POLL_SECONDS = 3600  # hourly

# Indicators / strategy
SHORT_SMA = 20
LONG_SMA = 50
RSI_LEN = 14
EMA_FAST = 12
EMA_SLOW = 26
MACD_SIG = 9
EMA_TREND_LEN = 200
VOL_AVG_LEN = 20
VOL_SURGE_MULT = 1.5
STARTING_CASH = 100_000.0

SCHEDULE_TICK_SECONDS = 60
DB_PATH = "ta_bot.db"

# =============================
# MARKET DATA
# =============================
class MarketData:
    @staticmethod
    async def price(ticker: str) -> float:
        """Prefer Finnhub real-time quote if key is set; fallback to yfinance."""
        try:
            if FINNHUB_TOKEN:
                import urllib.parse, urllib.request, json
                url = f"https://finnhub.io/api/v1/quote?symbol={urllib.parse.quote(ticker)}&token={FINNHUB_TOKEN}"
                def _req():
                    with urllib.request.urlopen(url, timeout=10) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                        return float(data.get("c") or float("nan"))
                try:
                    q = await asyncio.to_thread(_req)
                    if not math.isnan(q):
                        return q
                except Exception:
                    pass
            # fallback to Yahoo
            def _p():
                t = yf.Ticker(ticker)
                fi = getattr(t, "fast_info", None)
                if fi and getattr(fi, "last_price", None):
                    return float(fi.last_price)
                hist = t.history(period="1d", interval="1m")
                return float(hist["Close"].dropna().iloc[-1]) if not hist.empty else float("nan")
            return await asyncio.to_thread(_p)
        except Exception as e:
            print(f"Error fetching price for {ticker}: {e}")
            return float("nan")

    @staticmethod
    async def history(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """Historical OHLCV from Yahoo (good coverage + auto-adjust)."""
        try:
            def _h():
                df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
                if df is None or df.empty:
                    return pd.DataFrame()
                cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
                return df[cols].dropna()
            return await asyncio.to_thread(_h)
        except Exception as e:
            print(f"Error fetching history for {ticker}: {e}")
            return pd.DataFrame()

# =============================
# INDICATORS
# =============================

def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average"""
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average"""
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(series: pd.Series, length: int = RSI_LEN) -> pd.Series:
    """Relative Strength Index"""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    roll_up = gain.ewm(alpha=1/length, adjust=False).mean()
    roll_dn = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / (roll_dn.replace(0, 1e-9))
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = EMA_FAST, slow: int = EMA_SLOW, signal: int = MACD_SIG):
    """MACD Indicator"""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

# =============================
# SIGNAL ENGINE
# =============================
async def build_ta_snapshot(ticker: str, *, period: str = "1y", interval: str = "1d") -> Dict[str, str]:
    """Build technical analysis snapshot for a ticker"""
    try:
        df = await MarketData.history(ticker, period=period, interval=interval)
        if df.empty:
            return {"error": "No data"}

        close = df["Close"]
        volume = df.get("Volume") if "Volume" in df.columns else pd.Series(index=df.index, dtype=float)

        need = max(LONG_SMA, RSI_LEN, EMA_SLOW + MACD_SIG, EMA_TREND_LEN) + 2
        if len(close) < need:
            return {"error": f"Not enough data (need ~{need} bars)."}

        sma_short = sma(close, SHORT_SMA)
        sma_long = sma(close, LONG_SMA)
        ema_trend = ema(close, EMA_TREND_LEN)
        rsi14 = rsi(close, RSI_LEN)
        macd_line, macd_sig, macd_hist = macd(close)

        px = float(close.iloc[-1])
        sma_s_now = float(sma_short.iloc[-1])
        sma_l_now = float(sma_long.iloc[-1])
        ema200_now = float(ema_trend.iloc[-1])
        rsi_now = float(rsi14.iloc[-1])
        macd_now = float(macd_line.iloc[-1])
        macd_sig_now = float(macd_sig.iloc[-1])
        macd_hist_now = float(macd_hist.iloc[-1])

        # Crossovers
        prev_s_s, prev_s_l = float(sma_short.iloc[-2]), float(sma_long.iloc[-2])
        sma_cross = "none"
        if prev_s_s <= prev_s_l and sma_s_now > sma_l_now:
            sma_cross = "bullish_cross"
        elif prev_s_s >= prev_s_l and sma_s_now < sma_l_now:
            sma_cross = "bearish_cross"

        sma50 = sma(close, 50)
        prev50, now50 = float(sma50.iloc[-2]), float(sma50.iloc[-1])
        prev200, now200 = float(ema_trend.iloc[-2]), float(ema_trend.iloc[-1])
        gdcross = "none"
        if prev50 <= prev200 and now50 > now200:
            gdcross = "golden_cross"
        elif prev50 >= prev200 and now50 < now200:
            gdcross = "death_cross"

        prev_macd, prev_sig = float(macd_line.iloc[-2]), float(macd_sig.iloc[-2])
        macd_cross = "none"
        if prev_macd <= prev_sig and macd_now > macd_sig_now:
            macd_cross = "bullish_cross"
        elif prev_macd >= prev_sig and macd_now < macd_sig_now:
            macd_cross = "bearish_cross"
        macd_center = "above_zero" if macd_now > 0 else ("below_zero" if macd_now < 0 else "at_zero")

        # Volume
        vol_ok = None
        breakout = None
        if volume is not None and not volume.empty:
            vol_ma = volume.rolling(VOL_AVG_LEN, min_periods=VOL_AVG_LEN).mean()
            vol_now = float(volume.iloc[-1])
            vol_avg = float(vol_ma.iloc[-1]) if not math.isnan(float(vol_ma.iloc[-1])) else 0.0
            vol_ok = (vol_now >= VOL_SURGE_MULT * vol_avg) if vol_avg > 0 else False
            prev_px = float(close.iloc[-2])
            breakout = (px >= prev_px * 1.01) and bool(vol_ok)

        trend = "uptrend" if px > ema200_now else ("downtrend" if px < ema200_now else "sideways")

        rules = []
        rules.append(f"Trend: price {'>' if px>ema200_now else '<' if px<ema200_now else '='} EMA{EMA_TREND_LEN} → {trend}")
        rules.append("SMA20/50: " + ("bullish cross" if sma_cross=="bullish_cross" else "bearish cross" if sma_cross=="bearish_cross" else "no new cross"))
        if gdcross != "none":
            rules.append("Long-term: " + ("Golden Cross (50 over 200)" if gdcross=="golden_cross" else "Death Cross (50 under 200)"))
        rules.append("MACD: " + ("bullish cross" if macd_cross=="bullish_cross" else "bearish cross" if macd_cross=="bearish_cross" else "no new cross") + f", centerline: {macd_center}")
        if rsi_now < 30:
            rules.append("RSI < 30 (oversold) → BUY bias")
        elif rsi_now > 70:
            rules.append("RSI > 70 (overbought) → SELL bias")
        else:
            rules.append("RSI neutral (30–70)")
        if vol_ok is not None:
            rules.append(f"Volume: {'surge' if vol_ok else 'normal/low'} vs {VOL_AVG_LEN}-period avg")
            if breakout:
                rules.append("Breakout + volume confirmation")

        return {
            "ticker": ticker.upper(),
            "price": f"{px:,.2f}",
            "ema200": f"{ema200_now:,.2f}",
            "sma20": f"{sma_s_now:,.2f}",
            "sma50": f"{sma_l_now:,.2f}",
            "rsi14": f"{rsi_now:,.2f}",
            "macd": f"{macd_now:,.4f}",
            "macd_sig": f"{macd_sig_now:,.4f}",
            "macd_hist": f"{macd_hist_now:,.4f}",
            "trend": trend,
            "rules": " | ".join(rules)
        }
    except Exception as e:
        print(f"Error building TA snapshot for {ticker}: {e}")
        return {"error": f"Analysis error: {str(e)}"}

# =============================
# SQLITE PERSISTENCE
# =============================
class DB:
    _conn: Optional[aiosqlite.Connection] = None

    @classmethod
    async def init(cls):
        """Initialize database connection and create tables"""
        try:
            cls._conn = await aiosqlite.connect(DB_PATH)
            await cls._conn.execute("PRAGMA journal_mode=WAL;")
            await cls._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    cash REAL NOT NULL
                );
                """
            )
            await cls._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS positions (
                    user_id INTEGER NOT NULL,
                    ticker TEXT NOT NULL,
                    qty INTEGER NOT NULL,
                    avg_price REAL NOT NULL,
                    PRIMARY KEY(user_id, ticker)
                );
                """
            )
            await cls._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS keywords (
                    guild_id INTEGER NOT NULL,
                    keyword TEXT NOT NULL,
                    PRIMARY KEY(guild_id, keyword)
                );
                """
            )
            await cls._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS watchlist (
                    guild_id INTEGER NOT NULL,
                    ticker TEXT NOT NULL,
                    PRIMARY KEY(guild_id, ticker)
                );
                """
            )
            await cls._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS holidays (
                    guild_id INTEGER NOT NULL,
                    ymd TEXT NOT NULL,
                    PRIMARY KEY(guild_id, ymd)
                );
                """
            )
            await cls._conn.commit()

            # Seed defaults (global and existing guilds)
            async def seed_defaults(guild_id: int):
                async with cls._conn.execute("SELECT COUNT(*) FROM watchlist WHERE guild_id=?", (guild_id,)) as cur:
                    row = await cur.fetchone()
                if row and row[0] == 0:
                    defaults = [
                        'RELIANCE.NS','HDFCBANK.NS','INFY.NS','TCS.NS','ICICIBANK.NS',
                        'KOTAKBANK.NS','AXISBANK.NS','SBIN.NS','BHARTIARTL.NS','HINDUNILVR.NS']
                    for t in defaults:
                        await cls._conn.execute("INSERT INTO watchlist(guild_id, ticker) VALUES(?,?)", (guild_id, t))
                    await cls._conn.commit()
            await seed_defaults(0)
        except Exception as e:
            print(f"Database initialization error: {e}")

    @classmethod
    async def conn(cls) -> aiosqlite.Connection:
        """Get database connection"""
        if cls._conn is None:
            await cls.init()
        return cls._conn

    @classmethod
    async def close(cls):
        """Close database connection"""
        if cls._conn:
            await cls._conn.close()
            cls._conn = None

# =============================
# PORTFOLIO (persisted; disabled unless ENABLE_PAPER)
# =============================
class Portfolio:
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.cash: float = STARTING_CASH
        self.positions: Dict[str, Tuple[int, float]] = {}

    async def load(self):
        """Load portfolio from database"""
        try:
            db = await DB.conn()
            async with db.execute("SELECT cash FROM users WHERE user_id=?", (self.user_id,)) as cur:
                row = await cur.fetchone()
                if row is None:
                    await db.execute("INSERT INTO users(user_id, cash) VALUES (?, ?)", (self.user_id, STARTING_CASH))
                    await db.commit()
                    self.cash = STARTING_CASH
                else:
                    self.cash = float(row[0])
            async with db.execute("SELECT ticker, qty, avg_price FROM positions WHERE user_id=?", (self.user_id,)) as cur:
                async for tkr, qty, avg in cur:
                    self.positions[tkr] = (int(qty), float(avg))
        except Exception as e:
            print(f"Error loading portfolio for user {self.user_id}: {e}")

    async def _save_cash(self):
        """Save cash balance to database"""
        try:
            db = await DB.conn()
            await db.execute("UPDATE users SET cash=? WHERE user_id=?", (self.cash, self.user_id))
            await db.commit()
        except Exception as e:
            print(f"Error saving cash for user {self.user_id}: {e}")

    async def _save_position(self, ticker: str):
        """Save position to database"""
        try:
            db = await DB.conn()
            if ticker in self.positions:
                qty, avg = self.positions[ticker]
                sql = (
                    "INSERT INTO positions(user_id,ticker,qty,avg_price) VALUES(?,?,?,?) "
                    "ON CONFLICT(user_id,ticker) DO UPDATE SET qty=excluded.qty, avg_price=excluded.avg_price"
                )
                await db.execute(sql, (self.user_id, ticker, qty, avg))
            else:
                await db.execute("DELETE FROM positions WHERE user_id=? AND ticker=?", (self.user_id, ticker))
            await db.commit()
        except Exception as e:
            print(f"Error saving position {ticker} for user {self.user_id}: {e}")

    async def buy(self, ticker: str, qty: int):
        """Buy shares"""
        try:
            px = await MarketData.price(ticker)
            if math.isnan(px):
                raise ValueError("Price unavailable.")
            cost = qty * px
            if cost > self.cash:
                raise ValueError("Insufficient cash.")
            if ticker in self.positions:
                old_qty, old_avg = self.positions[ticker]
                new_qty = old_qty + qty
                new_avg = (old_qty * old_avg + qty * px) / new_qty
                self.positions[ticker] = (new_qty, new_avg)
            else:
                self.positions[ticker] = (qty, px)
            self.cash -= cost
            await self._save_cash()
            await self._save_position(ticker)
            return px
        except Exception as e:
            print(f"Error buying {ticker} for user {self.user_id}: {e}")
            raise

    async def sell(self, ticker: str, qty: int):
        """Sell shares"""
        try:
            if ticker not in self.positions:
                raise ValueError("No position.")
            pos_qty, avg = self.positions[ticker]
            if qty > pos_qty:
                raise ValueError("Not enough shares.")
            px = await MarketData.price(ticker)
            if math.isnan(px):
                raise ValueError("Price unavailable.")
            proceeds = qty * px
            self.cash += proceeds
            remaining = pos_qty - qty
            if remaining == 0:
                del self.positions[ticker]
            else:
                self.positions[ticker] = (remaining, avg)
            await self._save_cash()
            await self._save_position(ticker)
            pnl_trade = (px - avg) * qty
            return px, pnl_trade
        except Exception as e:
            print(f"Error selling {ticker} for user {self.user_id}: {e}")
            raise

    async def snapshot(self) -> str:
        """Get portfolio snapshot"""
        try:
            value = self.cash
            lines = [f"**Cash:** ₹{self.cash:,.2f}"]
            total_unreal = 0.0
            for tkr, (q, avg) in self.positions.items():
                px = await MarketData.price(tkr)
                if math.isnan(px):
                    px = 0.0
                mkt = q * px
                unreal = (px - avg) * q
                total_unreal += unreal
                value += mkt
                lines.append(f"**{tkr}**: {q} @ ₹{avg:,.2f} | Px ₹{px:,.2f} | MV ₹{mkt:,.2f} | UPL ₹{unreal:,.2f}")
            lines.append(f"**Unrealized P&L:** ₹{total_unreal:,.2f}")
            lines.append(f"**Portfolio Value:** ₹{value:,.2f}")
            return "\n".join(lines)
        except Exception as e:
            print(f"Error generating portfolio snapshot for user {self.user_id}: {e}")
            return f"Error generating portfolio snapshot: {str(e)}"

# =============================
# WATCHLIST / KEYWORDS / HOLIDAYS HELPERS
# =============================
async def normalize_ticker(t: str) -> str:
    """Normalize ticker symbol"""
    t = t.strip().upper()
    return t if t.endswith('.NS') else f"{t}.NS"

async def get_watchlist(guild_id: int) -> List[str]:
    """Get watchlist for guild"""
    try:
        db = await DB.conn()
        async with db.execute("SELECT ticker FROM watchlist WHERE guild_id IN (?, 0) GROUP BY ticker ORDER BY ticker", (guild_id,)) as cur:
            return [row[0] async for row in cur]
    except Exception as e:
        print(f"Error getting watchlist for guild {guild_id}: {e}")
        return []

async def add_watch(guild_id: int, ticker: str) -> None:
    """Add ticker to watchlist"""
    try:
        db = await DB.conn()
        t = await normalize_ticker(ticker)
        await db.execute("INSERT OR IGNORE INTO watchlist(guild_id, ticker) VALUES(?,?)", (guild_id, t))
        await db.commit()
    except Exception as e:
        print(f"Error adding {ticker} to watchlist for guild {guild_id}: {e}")

async def remove_watch(guild_id: int, ticker: str) -> None:
    """Remove ticker from watchlist"""
    try:
        db = await DB.conn()
        t = await normalize_ticker(ticker)
        await db.execute("DELETE FROM watchlist WHERE guild_id=? AND ticker=?", (guild_id, t))
        await db.commit()
    except Exception as e:
        print(f"Error removing {ticker} from watchlist for guild {guild_id}: {e}")

async def get_keywords(guild_id: int) -> List[str]:
    """Get keywords for guild"""
    try:
        db = await DB.conn()
        async with db.execute("SELECT keyword FROM keywords WHERE guild_id=? ORDER BY keyword", (guild_id,)) as cur:
            return [row[0] async for row in cur]
    except Exception as e:
        print(f"Error getting keywords for guild {guild_id}: {e}")
        return []

async def add_keyword(guild_id: int, kw: str) -> None:
    """Add keyword"""
    try:
        db = await DB.conn()
        await db.execute("INSERT OR IGNORE INTO keywords(guild_id, keyword) VALUES(?,?)", (guild_id, kw.lower()))
        await db.commit()
    except Exception as e:
        print(f"Error adding keyword {kw} for guild {guild_id}: {e}")

async def remove_keyword(guild_id: int, kw: str) -> None:
    """Remove keyword"""
    try:
        db = await DB.conn()
        await db.execute("DELETE FROM keywords WHERE guild_id=? AND keyword=?", (guild_id, kw.lower()))
        await db.commit()
    except Exception as e:
        print(f"Error removing keyword {kw} for guild {guild_id}: {e}")

async def list_holidays(guild_id: int) -> List[str]:
    """List holidays for guild"""
    try:
        db = await DB.conn()
        async with db.execute("SELECT ymd FROM holidays WHERE guild_id=? ORDER BY ymd", (guild_id,)) as cur:
            return [row[0] async for row in cur]
    except Exception as e:
        print(f"Error listing holidays for guild {guild_id}: {e}")
        return []

async def add_holiday(guild_id: int, ymd: str) -> None:
    """Add holiday"""
    try:
        db = await DB.conn()
        await db.execute("INSERT OR IGNORE INTO holidays(guild_id, ymd) VALUES(?,?)", (guild_id, ymd))
        await db.commit()
    except Exception as e:
        print(f"Error adding holiday {ymd} for guild {guild_id}: {e}")

async def remove_holiday(guild_id: int, ymd: str) -> None:
    """Remove holiday"""
    try:
        db = await DB.conn()
        await db.execute("DELETE FROM holidays WHERE guild_id=? AND ymd=?", (guild_id, ymd))
        await db.commit()
    except Exception as e:
        print(f"Error removing holiday {ymd} for guild {guild_id}: {e}")

# =============================
# DISCORD BOT
# =============================
intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)

# Global variables for tracking
_last_posted_keys: set = set()

async def _format_bias_line(ticker: str) -> str:
    """Format bias line for market pulse"""
    try:
        # Intraday granularity
        snap = await build_ta_snapshot(ticker, period="7d", interval="15m")
        if "error" in snap:
            return f"{ticker}: data error"

        # Gap calculation: today's open vs yesterday close
        def _hist_daily():
            df = yf.download(ticker, period="7d", interval="1d", auto_adjust=True, progress=False)
            return df.dropna() if df is not None else pd.DataFrame()
        
        ddf = await asyncio.to_thread(_hist_daily)
        gap_txt = ""
        if not ddf.empty and len(ddf) >= 2:
            prev_close = float(ddf["Close"].iloc[-2])
            today_open = float(ddf["Open"].iloc[-1])
            if prev_close > 0:
                gap_pct = (today_open/prev_close - 1) * 100
                arrow = "↑" if gap_pct >= 0 else "↓"
                gap_txt = f" | gap{arrow} {gap_pct:+.2f}%"

        text = snap["rules"].lower()
        buy = (("bullish cross" in text or "oversold" in text) and ("trend: price >" in snap["rules"])) or ("centerline: above_zero" in snap["rules"])
        sell = (("bearish cross" in text or "overbought" in text) and ("trend: price <" in snap["rules"])) or ("centerline: below_zero" in snap["rules"])
        bias = "BUY" if buy and not sell else ("SELL" if sell and not buy else "NEUTRAL")
        macd_state = 'bull' if 'bullish cross' in text else 'bear' if 'bearish cross' in text else 'flat'
        return f"{ticker}: {bias} | RSI {snap['rsi14']} | MACD {macd_state}{gap_txt}"
    except Exception as e:
        print(f"Error formatting bias line for {ticker}: {e}")
        return f"{ticker}: error"

@tasks.loop(seconds=SCHEDULE_TICK_SECONDS)
async def market_pulse():
    """Periodic market pulse during trading hours"""
    try:
        now = datetime.now(ZoneInfo(IST_TZ))
        if now.weekday() >= 5:  # Skip weekends
            return
        if not ((now.hour, now.minute) >= (MARKET_OPEN_HOUR, MARKET_OPEN_MIN) and (now.hour, now.minute) <= (MARKET_CLOSE_HOUR, MARKET_CLOSE_MIN)):
            return
        if (now.minute % REPORT_INTERVAL_MIN) != 0:
            return
        
        minute_key = f"{now:%Y-%m-%d %H:%M}"
        for guild in bot.guilds:
            key = (guild.id, minute_key)
            if key in _last_posted_keys:
                continue
            
            # Skip holidays
            hols = set(await list_holidays(guild.id))
            if now.strftime('%Y-%m-%d') in hols:
                continue
            
            wl = await get_watchlist(guild.id)
            if not wl:
                continue
            
            # Find appropriate channel
            channel: Optional[discord.TextChannel] = None
            if SIGNALS_CHANNEL_ID:
                ch = bot.get_channel(int(SIGNALS_CHANNEL_ID))
                if isinstance(ch, discord.TextChannel) and ch.guild.id == guild.id:
                    channel = ch
            if channel is None:
                found = discord.utils.get(guild.text_channels, name=SIGNALS_CHANNEL_NAME)
                if found and found.permissions_for(guild.me).send_messages:
                    channel = found
                else:
                    for c in guild.text_channels:
                        if c.permissions_for(guild.me).send_messages:
                            channel = c
                            break
            if not channel:
                continue
            
            lines = [f"📊 NSE Watchlist Update ({now:%H:%M} IST)"]
            for t in wl[:25]:  # Limit to 25 tickers
                try:
                    lines.append(await _format_bias_line(t))
                except Exception:
                    lines.append(f"{t}: error")
            
            await channel.send("\n".join(lines))
            _last_posted_keys.add(key)
    except Exception as e:
        print("market_pulse error:", e)

# =============================
# MARKET NEWS (MarketAux, cached hourly)
# =============================
_MA_CACHE: Dict[str, Any] = {"ts": 0, "data": []}

class NewsData:
    @staticmethod
    async def fetch_marketaux(keywords: List[str]) -> List[Dict[str, str]]:
        """Fetch news from MarketAux API"""
        try:
            if not MARKETAUX_TOKEN or not keywords:
                return []
            
            if time.time() - _MA_CACHE["ts"] < NEWS_POLL_SECONDS - 5 and _MA_CACHE["data"]:
                data = _MA_CACHE["data"]
            else:
                import urllib.request, json
                url = f"https://api.marketaux.com/v1/news/all?filter_entities=true&limit=100&api_token={MARKETAUX_TOKEN}"
                def _req():
                    with urllib.request.urlopen(url, timeout=20) as resp:
                        payload = json.loads(resp.read().decode("utf-8"))
                        return payload.get("data", [])
                try:
                    data = await asyncio.to_thread(_req)
                    _MA_CACHE["ts"] = time.time()
                    _MA_CACHE["data"] = data
                except Exception:
                    return []
            
            kws = [k.lower() for k in keywords]
            out = []
            for a in data:
                title = (a.get("title") or "").strip()
                if not title:
                    continue
                if any(k in title.lower() for k in kws):
                    out.append({
                        "title": title,
                        "url": a.get("url", ""),
                        "published_at": a.get("published_at", ""),
                    })
            return out
        except Exception as e:
            print(f"Error fetching MarketAux news: {e}")
            return []

@tasks.loop(seconds=NEWS_POLL_SECONDS)
async def news_watcher():
    """Watch for relevant news"""
    try:
        for guild in bot.guilds:
            kws = await get_keywords(guild.id)
            if not kws:
                continue
            
            items = await NewsData.fetch_marketaux(kws)
            if not items:
                continue
            
            # Find appropriate channel
            channel: Optional[discord.TextChannel] = None
            if SIGNALS_CHANNEL_ID:
                ch = bot.get_channel(int(SIGNALS_CHANNEL_ID))
                if isinstance(ch, discord.TextChannel) and ch.guild.id == guild.id:
                    channel = ch
            if channel is None:
                found = discord.utils.get(guild.text_channels, name=SIGNALS_CHANNEL_NAME)
                if found and found.permissions_for(guild.me).send_messages:
                    channel = found
                else:
                    for c in guild.text_channels:
                        if c.permissions_for(guild.me).send_messages:
                            channel = c
                            break
            if not channel:
                continue
            
            seen = set()
            for it in items[:5]:  # Limit to 5 news items
                title = it.get("title", "")
                url = it.get("url", "")
                if not title or title in seen:
                    continue
                seen.add(title)
                await channel.send(f"📰 **News:** {title}\n{url}")
    except Exception as e:
        print("news_watcher error:", e)

@news_watcher.before_loop
async def before_news():
    """Wait for bot to be ready before starting news watcher"""
    await bot.wait_until_ready()
    try:
        await asyncio.sleep(random.uniform(60, 420))  # 1–7 minutes jitter
    except Exception:
        pass

# =============================
# BOT EVENTS
# =============================
@bot.event
async def on_ready():
    """Bot ready event"""
    try:
        await DB.init()
        if GUILD_ID:
            guild = discord.Object(id=int(GUILD_ID))
            await bot.tree.sync(guild=guild)
        else:
            await bot.tree.sync()
        
        news_watcher.start()
        market_pulse.start()
        print(f"Logged in as {bot.user} (ID: {bot.user.id})")
        
        # Announce defaults once per guild
        defaults = ['RELIANCE.NS','HDFCBANK.NS','INFY.NS','TCS.NS','ICICIBANK.NS',
                    'KOTAKBANK.NS','AXISBANK.NS','SBIN.NS','BHARTIARTL.NS','HINDUNILVR.NS']
        for guild in bot.guilds:
            channel = None
            if SIGNALS_CHANNEL_ID:
                ch = bot.get_channel(int(SIGNALS_CHANNEL_ID))
                if isinstance(ch, discord.TextChannel) and ch.guild.id == guild.id:
                    channel = ch
            if channel is None:
                # prefer/create #signals
                found = discord.utils.get(guild.text_channels, name=SIGNALS_CHANNEL_NAME)
                if found and found.permissions_for(guild.me).send_messages:
                    channel = found
                else:
                    try:
                        overwrites = {guild.default_role: discord.PermissionOverwrite(read_messages=True, send_messages=True)}
                        channel = await guild.create_text_channel(SIGNALS_CHANNEL_NAME, overwrites=overwrites)
                    except Exception:
                        for c in guild.text_channels:
                            if c.permissions_for(guild.me).send_messages:
                                channel = c
                                break
            if channel:
                try:
                    await channel.send("📈 Bot is live! Default NSE watchlist seeded:\n" + ", ".join(defaults))
                except Exception:
                    pass
    except Exception as e:
        print("Startup error:", e)

@bot.event
async def on_guild_join(guild: discord.Guild):
    """Handle new guild joins"""
    try:
        await DB.init()
        # Seed default list
        async with (await DB.conn()).execute("SELECT COUNT(*) FROM watchlist WHERE guild_id=?", (guild.id,)) as cur:
            row = await cur.fetchone()
        if row and row[0] == 0:
            defaults = [
                'RELIANCE.NS','HDFCBANK.NS','INFY.NS','TCS.NS','ICICIBANK.NS',
                'KOTAKBANK.NS','AXISBANK.NS','SBIN.NS','BHARTIARTL.NS','HINDUNILVR.NS']
            for t in defaults:
                await add_watch(guild.id, t)
        
        # Find appropriate channel and announce
        channel: Optional[discord.TextChannel] = None
        if SIGNALS_CHANNEL_ID:
            ch = bot.get_channel(int(SIGNALS_CHANNEL_ID))
            if isinstance(ch, discord.TextChannel) and ch.guild.id == guild.id:
                channel = ch
        if channel is None:
            found = discord.utils.get(guild.text_channels, name=SIGNALS_CHANNEL_NAME)
            if found and found.permissions_for(guild.me).send_messages:
                channel = found
            else:
                for c in guild.text_channels:
                    if c.permissions_for(guild.me).send_messages:
                        channel = c
                        break
        if channel:
            wl = await get_watchlist(guild.id)
            lines = [
                "👋 **Thanks for adding the bot!**",
                "I've seeded a default NSE watchlist:",
                ", ".join(wl),
                "\nI'll post combined updates every 30 min from 09:00–15:30 IST.",
                "Manage with `/watch add`, `/watch remove`, `/watch list`."
            ]
            await channel.send("\n".join(lines))
    except Exception as e:
        print("on_guild_join error:", e)

# =============================
# BASIC COMMANDS
# =============================
@bot.tree.command(description="Get the latest price for a ticker")
@app_commands.describe(ticker="e.g., INFY.NS, RELIANCE.NS")
async def price(interaction: discord.Interaction, ticker: str):
    """Get current price for a ticker"""
    await interaction.response.defer(thinking=True)
    try:
        px = await MarketData.price(ticker)
        if math.isnan(px):
            await interaction.followup.send(f"Could not fetch price for `{ticker}`.")
            return
        await interaction.followup.send(f"**{ticker.upper()}**: ₹{px:,.2f}")
    except Exception as e:
        await interaction.followup.send(f"Error fetching price for `{ticker}`: {str(e)}")

@bot.tree.command(description="Generate TA snapshot (SMA/EMA/RSI/MACD/Volume)")
@app_commands.describe(ticker="e.g., INFY.NS", period="6mo, 1y, 5y", interval="1d, 1h, 1wk")
async def signal(interaction: discord.Interaction, ticker: str, period: str = "1y", interval: str = "1d"):
    """Generate technical analysis signal"""
    await interaction.response.defer(thinking=True)
    try:
        snap = await build_ta_snapshot(ticker, period=period, interval=interval)
        if "error" in snap:
            await interaction.followup.send(f"`{ticker}` → {snap['error']}")
            return
        
        msg = (
            f"**{snap['ticker']}** @ ₹{snap['price']}\n"
            f"EMA{EMA_TREND_LEN}: ₹{snap['ema200']} | SMA20: ₹{snap['sma20']} | SMA50: ₹{snap['sma50']}\n"
            f"RSI14: {snap['rsi14']} | MACD: {snap['macd']} | Signal: {snap['macd_sig']} | Hist: {snap['macd_hist']}\n"
            f"**Assessment:** {snap['rules']}"
        )
        await interaction.followup.send(msg)
        
        # Also send to signals channel if configured
        if SIGNALS_CHANNEL_ID:
            ch = bot.get_channel(int(SIGNALS_CHANNEL_ID))
            if ch and isinstance(ch, discord.TextChannel):
                await ch.send(f"[Signal] {msg}")
    except Exception as e:
        await interaction.followup.send(f"Error generating signal for `{ticker}`: {str(e)}")

# =============================
# KEYWORDS COMMANDS
# =============================
keywords = app_commands.Group(name="keywords", description="Manage news keywords")

@keywords.command(name="add", description="Add a news keyword for this server")
@app_commands.describe(word="e.g., earnings beat, acquisition, RELIANCE")
async def kw_add(interaction: discord.Interaction, word: str):
    """Add news keyword"""
    if interaction.guild is None:
        await interaction.response.send_message("Use in a server.", ephemeral=True)
        return
    try:
        await add_keyword(interaction.guild.id, word)
        await interaction.response.send_message(f"Added keyword: `{word}`")
    except Exception as e:
        await interaction.response.send_message(f"Error adding keyword: {str(e)}", ephemeral=True)

@keywords.command(name="remove", description="Remove a news keyword for this server")
@app_commands.describe(word="keyword to remove")
async def kw_remove(interaction: discord.Interaction, word: str):
    """Remove news keyword"""
    if interaction.guild is None:
        await interaction.response.send_message("Use in a server.", ephemeral=True)
        return
    try:
        await remove_keyword(interaction.guild.id, word)
        await interaction.response.send_message(f"Removed keyword: `{word}`")
    except Exception as e:
        await interaction.response.send_message(f"Error removing keyword: {str(e)}", ephemeral=True)

@keywords.command(name="list", description="List news keywords for this server")
async def kw_list(interaction: discord.Interaction):
    """List news keywords"""
    if interaction.guild is None:
        await interaction.response.send_message("Use in a server.", ephemeral=True)
        return
    try:
        kws = await get_keywords(interaction.guild.id)
        await interaction.response.send_message("Keywords: " + (", ".join(kws) if kws else "(none)"))
    except Exception as e:
        await interaction.response.send_message(f"Error listing keywords: {str(e)}", ephemeral=True)

bot.tree.add_command(keywords)

# =============================
# WATCHLIST COMMANDS
# =============================
watch = app_commands.Group(name="watch", description="Manage NSE watchlist (.NS)")

@watch.command(name="add", description="Add ticker to watchlist (.NS auto-appended)")
@app_commands.describe(ticker="e.g., RELIANCE or RELIANCE.NS")
async def watch_add(interaction: discord.Interaction, ticker: str):
    """Add ticker to watchlist"""
    if interaction.guild is None:
        await interaction.response.send_message("Use in a server.", ephemeral=True)
        return
    try:
        await add_watch(interaction.guild.id, ticker)
        normalized = await normalize_ticker(ticker)
        await interaction.response.send_message(f"Added `{normalized}` to watchlist")
    except Exception as e:
        await interaction.response.send_message(f"Error adding ticker: {str(e)}", ephemeral=True)

@watch.command(name="remove", description="Remove ticker from watchlist")
@app_commands.describe(ticker="Ticker to remove")
async def watch_remove(interaction: discord.Interaction, ticker: str):
    """Remove ticker from watchlist"""
    if interaction.guild is None:
        await interaction.response.send_message("Use in a server.", ephemeral=True)
        return
    try:
        await remove_watch(interaction.guild.id, ticker)
        normalized = await normalize_ticker(ticker)
        await interaction.response.send_message(f"Removed `{normalized}` from watchlist")
    except Exception as e:
        await interaction.response.send_message(f"Error removing ticker: {str(e)}", ephemeral=True)

@watch.command(name="list", description="List watchlist tickers")
async def watch_list(interaction: discord.Interaction):
    """List watchlist tickers"""
    if interaction.guild is None:
        await interaction.response.send_message("Use in a server.", ephemeral=True)
        return
    try:
        wl = await get_watchlist(interaction.guild.id)
        if not wl:
            await interaction.response.send_message("Watchlist is empty. Try `/watch add INFY`.")
        else:
            await interaction.response.send_message("Watchlist: " + ", ".join(wl))
    except Exception as e:
        await interaction.response.send_message(f"Error listing watchlist: {str(e)}", ephemeral=True)

bot.tree.add_command(watch)

# =============================
# HOLIDAY COMMANDS
# =============================
holidays = app_commands.Group(name="holidays", description="Manage NSE holiday dates (YYYY-MM-DD)")

@holidays.command(name="add", description="Add a holiday (YYYY-MM-DD)")
async def holiday_add(interaction: discord.Interaction, ymd: str):
    """Add holiday"""
    if interaction.guild is None:
        await interaction.response.send_message("Use in a server.", ephemeral=True)
        return
    try:
        await add_holiday(interaction.guild.id, ymd)
        await interaction.response.send_message(f"Added holiday `{ymd}`")
    except Exception as e:
        await interaction.response.send_message(f"Error adding holiday: {str(e)}", ephemeral=True)

@holidays.command(name="remove", description="Remove a holiday (YYYY-MM-DD)")
async def holiday_remove(interaction: discord.Interaction, ymd: str):
    """Remove holiday"""
    if interaction.guild is None:
        await interaction.response.send_message("Use in a server.", ephemeral=True)
        return
    try:
        await remove_holiday(interaction.guild.id, ymd)
        await interaction.response.send_message(f"Removed holiday `{ymd}`")
    except Exception as e:
        await interaction.response.send_message(f"Error removing holiday: {str(e)}", ephemeral=True)

@holidays.command(name="list", description="List configured holidays")
async def holiday_list(interaction: discord.Interaction):
    """List holidays"""
    if interaction.guild is None:
        await interaction.response.send_message("Use in a server.", ephemeral=True)
        return
    try:
        out = await list_holidays(interaction.guild.id)
        await interaction.response.send_message("Holidays: " + (", ".join(out) if out else "(none)"))
    except Exception as e:
        await interaction.response.send_message(f"Error listing holidays: {str(e)}", ephemeral=True)

bot.tree.add_command(holidays)

# =============================
# PAPER TRADING COMMANDS (if enabled)
# =============================
if ENABLE_PAPER:
    paper = app_commands.Group(name="paper", description="Paper trading commands")

    @paper.command(name="buy", description="Buy shares (paper trading)")
    @app_commands.describe(ticker="Ticker symbol", quantity="Number of shares")
    async def paper_buy(interaction: discord.Interaction, ticker: str, quantity: int):
        """Buy shares in paper trading"""
        await interaction.response.defer(thinking=True)
        try:
            if quantity <= 0:
                await interaction.followup.send("Quantity must be positive.")
                return
            
            portfolio = Portfolio(interaction.user.id)
            await portfolio.load()
            
            ticker = await normalize_ticker(ticker)
            px = await portfolio.buy(ticker, quantity)
            
            await interaction.followup.send(
                f"✅ Bought {quantity} shares of {ticker} @ ₹{px:,.2f}\n"
                f"Total cost: ₹{quantity * px:,.2f}"
            )
        except Exception as e:
            await interaction.followup.send(f"Error buying shares: {str(e)}")

    @paper.command(name="sell", description="Sell shares (paper trading)")
    @app_commands.describe(ticker="Ticker symbol", quantity="Number of shares")
    async def paper_sell(interaction: discord.Interaction, ticker: str, quantity: int):
        """Sell shares in paper trading"""
        await interaction.response.defer(thinking=True)
        try:
            if quantity <= 0:
                await interaction.followup.send("Quantity must be positive.")
                return
            
            portfolio = Portfolio(interaction.user.id)
            await portfolio.load()
            
            ticker = await normalize_ticker(ticker)
            px, pnl = await portfolio.sell(ticker, quantity)
            
            pnl_emoji = "📈" if pnl >= 0 else "📉"
            await interaction.followup.send(
                f"✅ Sold {quantity} shares of {ticker} @ ₹{px:,.2f}\n"
                f"Proceeds: ₹{quantity * px:,.2f}\n"
                f"{pnl_emoji} Trade P&L: ₹{pnl:,.2f}"
            )
        except Exception as e:
            await interaction.followup.send(f"Error selling shares: {str(e)}")

    @paper.command(name="portfolio", description="View your paper trading portfolio")
    async def paper_portfolio(interaction: discord.Interaction):
        """View paper trading portfolio"""
        await interaction.response.defer(thinking=True)
        try:
            portfolio = Portfolio(interaction.user.id)
            await portfolio.load()
            snapshot = await portfolio.snapshot()
            await interaction.followup.send(f"📊 **Your Portfolio**\n{snapshot}")
        except Exception as e:
            await interaction.followup.send(f"Error viewing portfolio: {str(e)}")

    bot.tree.add_command(paper)

# =============================
# BACKTEST
# =============================
def _max_drawdown(equity: pd.Series) -> float:
    """Calculate maximum drawdown"""
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())

async def run_backtest(ticker: str, start: str, end: str) -> Dict[str, Any]:
    """Run backtest on ticker"""
    try:
        def _hist():
            df = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
            return df.dropna() if df is not None else pd.DataFrame()
        
        df = await asyncio.to_thread(_hist)
        if df.empty:
            return {"error": 1}
        
        close = df["Close"].copy()

        s20 = close.rolling(20).mean()
        s50 = close.rolling(50).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()
        macd_line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
        macd_sig = macd_line.ewm(span=9, adjust=False).mean()

        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        roll_up = gain.ewm(alpha=1/14, adjust=False).mean()
        roll_dn = loss.ewm(alpha=1/14, adjust=False).mean()
        rs = roll_up / (roll_dn.replace(0, 1e-9))
        rsi14 = 100 - (100 / (1 + rs))

        pos = 0
        entry_px = 0.0
        trades: List[Tuple[pd.Timestamp, pd.Timestamp, float]] = []
        equity = [1.0]
        dates = [close.index[0]]

        for i in range(2, len(close)):
            px_prev, px_now = close.iloc[i-1], close.iloc[i]
            s20_prev, s50_prev = s20.iloc[i-1], s50.iloc[i-1]
            s20_now, s50_now = s20.iloc[i], s50.iloc[i]
            ema_now = ema200.iloc[i]
            macd_now, macd_sig_now = macd_line.iloc[i], macd_sig.iloc[i]
            rsi_now = rsi14.iloc[i]
            date_now = close.index[i]

            equity.append(equity[-1] if pos == 0 else equity[-1] * (px_now / px_prev))
            dates.append(date_now)

            if any(math.isnan(float(x)) for x in [s20_prev, s50_prev, s20_now, s50_now, ema_now, macd_now, macd_sig_now, rsi_now]):
                continue

            bullish_cross = s20_prev <= s50_prev and s20_now > s50_now
            bearish_cross = s20_prev >= s50_prev and s20_now < s50_now

            if pos == 0 and bullish_cross and px_now > ema_now and macd_now > macd_sig_now and rsi_now < 70:
                pos = 1
                entry_px = px_now
                in_time = date_now
            elif pos == 1 and (bearish_cross or rsi_now > 70):
                pos = 0
                ret = (px_now / entry_px) - 1.0
                trades.append((in_time, date_now, float(ret)))

        if pos == 1:
            ret = (close.iloc[-1] / entry_px) - 1.0
            trades.append((in_time, close.index[-1], float(ret)))

        eq_series = pd.Series(equity, index=pd.DatetimeIndex(dates))
        total_ret = float(eq_series.iloc[-1] - 1.0)
        years = (eq_series.index[-1] - eq_series.index[0]).days / 365.25
        cagr = (eq_series.iloc[-1]) ** (1/years) - 1 if years > 0 else total_ret
        mdd = _max_drawdown(eq_series)

        wins = [t for t in trades if t[2] > 0]
        win_rate = (len(wins) / len(trades)) if trades else 0.0

        return {
            "trades": len(trades),
            "win_rate": win_rate,
            "total_return": total_ret,
            "cagr": cagr,
            "max_drawdown": mdd,
            "start": str(eq_series.index[0].date()),
            "end": str(eq_series.index[-1].date())
        }
    except Exception as e:
        print(f"Backtest error for {ticker}: {e}")
        return {"error": str(e)}

@bot.tree.command(description="Backtest SMA/RSI/MACD trend-follow strategy")
@app_commands.describe(ticker="Ticker like INFY.NS", start="YYYY-MM-DD", end="YYYY-MM-DD")
async def backtest(interaction: discord.Interaction, ticker: str, start: str, end: str):
    """Run backtest"""
    await interaction.response.defer(thinking=True)
    try:
        res = await run_backtest(ticker, start, end)
        if "error" in res:
            await interaction.followup.send(f"Backtest failed: {res.get('error', 'No data')}")
            return
        
        msg = (
            f"**Backtest {ticker.upper()}** {res['start']} → {res['end']}\n"
            f"Trades: {res['trades']} | Win rate: {res['win_rate']*100:.1f}%\n"
            f"Total return: {res['total_return']*100:.1f}% | CAGR: {res['cagr']*100:.1f}% | Max DD: {res['max_drawdown']*100:.1f}%"
        )
        await interaction.followup.send(msg)
    except Exception as e:
        await interaction.followup.send(f"Backtest error: {str(e)}")

# =============================
# SELF-TESTS (optional)
# =============================
async def _self_tests():
    """Run self-tests"""
    print("Running self-tests…")
    try:
        # SMA sanity
        s = pd.Series([1, 2, 3, 4, 5], dtype=float)
        s2 = sma(s, 2)
        assert math.isnan(s2.iloc[0]) and abs(s2.iloc[1] - 1.5) < 1e-9
        
        # EMA, RSI, MACD shapes
        e2 = ema(s, 2)
        assert not e2.isna().all()
        r = rsi(s, 2)
        assert r.between(0, 100).all()
        m_line, m_sig, m_hist = macd(s)
        assert abs((m_line.iloc[-1] - m_sig.iloc[-1]) - m_hist.iloc[-1]) < 1e-6
        print("Indicators OK ✔")
    except Exception as e:
        print(f"Self-test failed: {e}")

# =============================
# CLEANUP ON EXIT
# =============================
async def cleanup():
    """Cleanup resources"""
    try:
        await DB.close()
        print("Database connection closed")
    except Exception as e:
        print(f"Cleanup error: {e}")

# =============================
# MAIN
# =============================
if __name__ == "__main__":
    if os.getenv("RUN_SELF_TESTS") == "1":
        asyncio.run(_self_tests())
    else:
        if not DISCORD_TOKEN:
            raise SystemExit("Missing DISCORD_TOKEN in .env")
        try:
            bot.run(DISCORD_TOKEN)
        except KeyboardInterrupt:
            print("Bot stopped by user")
        finally:
            asyncio.run(cleanup())
