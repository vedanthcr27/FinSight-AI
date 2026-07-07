"""
Stock Data Module
Handles real-time stock data fetching, caching, and formatting.
Uses yfinance for reliable market data and implements smart caching strategies.
"""

import math
import yfinance as yf
import logging
import pandas as pd
from functools import lru_cache
from datetime import datetime, timedelta
from typing import Dict, Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# FORMATTING UTILITIES
# ============================================================================

def format_currency(value: float | None, decimals: int = 2) -> str:
    """
    Format value as currency with intelligent scaling.
    
    Examples:
        3,000,000,000 -> "$3.00B"
        3,000,000 -> "$3.00M"
        1,234.56 -> "$1,234.56"
    
    Args:
        value: Numeric value to format
        decimals: Decimal places (default 2)
        
    Returns:
        Formatted currency string
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    
    if not isinstance(value, (int, float)):
        return str(value)
    
    abs_value = abs(value)
    sign = "-" if value < 0 else ""
    
    if abs_value >= 1_000_000_000_000:
        return f"{sign}${abs_value/1_000_000_000_000:.{decimals}f}T"
    elif abs_value >= 1_000_000_000:
        return f"{sign}${abs_value/1_000_000_000:.{decimals}f}B"
    elif abs_value >= 1_000_000:
        return f"{sign}${abs_value/1_000_000:.{decimals}f}M"
    elif abs_value >= 1_000:
        return f"{sign}${abs_value:,.{decimals}f}"
    else:
        return f"{sign}${abs_value:.{decimals}f}"


def format_pct(value: float | None, decimals: int = 2) -> str:
    """
    Format value as percentage.
    
    Args:
        value: Numeric value (as decimal, e.g., 0.25 for 25%)
        decimals: Decimal places
        
    Returns:
        Formatted percentage string (e.g., "25.00%")
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{value:.{decimals}f}%"


def format_number(value: float | None, decimals: int = 2) -> str:
    """
    Format value as readable number.
    
    Args:
        value: Numeric value
        decimals: Decimal places
        
    Returns:
        Formatted number string
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{value:,.{decimals}f}"


# ============================================================================
# DATA FETCHING WITH CACHING
# ============================================================================

class StockDataFetcher:
    """
    Smart stock data fetcher with caching and error handling.
    Minimizes API calls and handles network failures gracefully.
    """
    
    # Cache expiration times (in seconds)
    CACHE_TTL = {
        "intraday": 60,      # 1 minute for real-time data
        "daily": 300,        # 5 minutes for daily data
        "historical": 3600,  # 1 hour for historical data
    }
    
    # Fallback data for unavailable tickers
    FALLBACK_DATA = {
        "ticker": "UNKNOWN",
        "company": "Unknown Company",
        "current_price": 100.0,
        "change_pct": 0.0,
        "market_cap": 1_000_000_000,
        "pe_ratio": 25.0,
        "high_52": 120.0,
        "low_52": 80.0,
        "sector": "Technology",
        "summary": "Market data temporarily unavailable.",
        "beta": 1.0,
        "profit_margin": 0.0,
        "revenue_growth": 0.0,
        "earnings_growth": 0.0,
        "enterprise_value": None,
        "price_to_book": None,
        "target_mean_price": None,
        "recommendation_mean": None,
        "market_status": "Unknown",
        "dividend_yield": 0.0,
        "peg_ratio": None,
    }
    
    def __init__(self):
        """Initialize the fetcher."""
        self._cache = {}
        self._cache_times = {}
        logger.info("✓ StockDataFetcher initialized")
    
    def _is_cache_valid(self, key: str, ttl: int) -> bool:
        """Check if cached data is still valid."""
        if key not in self._cache_times:
            return False
        elapsed = (datetime.now() - self._cache_times[key]).total_seconds()
        return elapsed < ttl
    
    def get_stock_data(self, ticker: str) -> Dict:
        """
        Fetch comprehensive stock data.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with stock information
        """
        ticker = (ticker or "NVDA").strip().upper()
        
        # Check cache
        cache_key = f"stock_{ticker}"
        if self._is_cache_valid(cache_key, self.CACHE_TTL["daily"]):
            logger.debug(f"✓ Using cached data for {ticker}")
            return self._cache[cache_key]
        
        try:
            logger.info(f"📡 Fetching data for {ticker}...")
            stock = yf.Ticker(ticker)
            info = stock.info
            fast = stock.fast_info
            
            # Safely extract data with type conversion
            data = {
                "ticker": ticker,
                "company": self._safe_get(info, "longName", info.get("shortName", ticker)),
                "current_price": self._safe_float(
                    fast.get("lastPrice") or info.get("currentPrice")
                ),

                "market_cap": self._safe_float(
                    fast.get("marketCap") or info.get("marketCap")
                ),

                "change_pct": self._safe_float(
                    info.get("regularMarketChangePercent")
                ),
                "pe_ratio": self._safe_float(info.get("trailingPE"), 25.0),
                "high_52": self._safe_float(info.get("fiftyTwoWeekHigh"), 120.0),
                "low_52": self._safe_float(info.get("fiftyTwoWeekLow"), 80.0),
                "sector": self._safe_get(info, "sector", "Technology"),
                "summary": self._safe_get(info, "longBusinessSummary", ""),
                "beta": self._safe_float(info.get("beta"), 1.0),
                "profit_margin": self._safe_float(info.get("profitMargins"), 0.0) * 100,
                "revenue_growth": self._safe_float(info.get("revenueGrowth"), 0.0) * 100,
                "earnings_growth": self._safe_float(info.get("earningsGrowth"), 0.0) * 100,
                "enterprise_value": info.get("enterpriseValue"),
                "price_to_book": self._safe_float(info.get("priceToBook")),
                "target_mean_price": self._safe_float(info.get("targetMeanPrice")),
                "recommendation_mean": info.get("recommendationMean"),
                "market_status": "Open",
                "dividend_yield": self._safe_float(info.get("dividendYield"), 0.0) * 100,
                "peg_ratio": self._safe_float(info.get("pegRatio")),
            }
            
            # Cache the data
            self._cache[cache_key] = data
            self._cache_times[cache_key] = datetime.now()
            logger.info(f"✓ Successfully fetched {ticker}")
            
            return data
            
        except Exception as e:
            logger.warning(f"⚠ Error fetching {ticker}: {e}. Using fallback data.")
            fallback = self.FALLBACK_DATA.copy()
            fallback["ticker"] = ticker
            return fallback
    
    def get_historical_data(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """
        Fetch historical stock data for technical analysis.
        
        Args:
            ticker: Stock ticker symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            
        Returns:
            DataFrame with OHLCV data
        """
        ticker = ticker.strip().upper()
        
        # Check cache
        cache_key = f"hist_{ticker}_{period}"
        if self._is_cache_valid(cache_key, self.CACHE_TTL["historical"]):
            logger.debug(f"✓ Using cached historical data for {ticker}")
            return self._cache[cache_key]
        
        try:
            logger.info(f"📡 Fetching historical data for {ticker} ({period})...")
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            
            if hist.empty:
                logger.warning(f"No historical data for {ticker}")
                return pd.DataFrame()
            
            # Cache the data
            self._cache[cache_key] = hist
            self._cache_times[cache_key] = datetime.now()
            logger.info(f"✓ Successfully fetched historical data for {ticker}")
            
            return hist
            
        except Exception as e:
            logger.error(f"✗ Error fetching historical data for {ticker}: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def _safe_float(value: any, default: float = None) -> float | None:
        """Safely convert value to float."""
        try:
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return default
            return float(value)
        except (ValueError, TypeError):
            return default
    
    @staticmethod
    def _safe_get(obj: dict, key: str, default: str = "N/A") -> str:
        """Safely get dictionary value."""
        try:
            value = obj.get(key, default)
            return str(value) if value else default
        except Exception:
            return default
    
    def clear_cache(self, ticker: Optional[str] = None) -> None:
        """
        Clear cache for specific ticker or entire cache.
        
        Args:
            ticker: Specific ticker to clear, or None for all
        """
        if ticker:
            prefix = f"stock_{ticker.upper()}"
            keys_to_remove = [k for k in self._cache.keys() if k.startswith(prefix)]
            for key in keys_to_remove:
                del self._cache[key]
                del self._cache_times[key]
            logger.info(f"✓ Cleared cache for {ticker}")
        else:
            self._cache.clear()
            self._cache_times.clear()
            logger.info("✓ Cleared all cache")


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_fetcher = StockDataFetcher()


def get_stock_data(ticker: str) -> Dict:
    """
    Get stock data using the global fetcher instance.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary with stock information
    """
    return _fetcher.get_stock_data(ticker)


def get_historical_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    Get historical stock data using the global fetcher instance.
    
    Args:
        ticker: Stock ticker symbol
        period: Data period
        
    Returns:
        DataFrame with historical data
    """
    return _fetcher.get_historical_data(ticker, period)


# ============================================================================
# ANALYSIS UTILITIES
# ============================================================================

def build_insight_bullets(profile: Dict) -> List[str]:
    """
    Generate investment insight bullets from stock metrics.
    
    Args:
        profile: Stock profile dictionary
        
    Returns:
        List of insight strings with emojis
    """
    bullets = []
    
    growth = profile.get("revenue_growth") or 0.0
    margin = profile.get("profit_margin") or 0.0
    change_pct = profile.get("change_pct") or 0.0
    pe = profile.get("pe_ratio") or 0.0
    beta = profile.get("beta") or 1.0

    # Growth Analysis
    if growth > 20:
        bullets.append("🚀 Exceptional growth momentum with accelerating revenue expansion.")
    elif growth > 15:
        bullets.append("📈 Strong growth trajectory supported by market demand.")
    elif growth > 5:
        bullets.append("📊 Moderate growth pace, stable trajectory evident.")
    else:
        bullets.append("⚠️ Growth is modest, monitor against peer performance.")

    # Profitability Analysis
    if margin > 30:
        bullets.append("💎 Superior margins demonstrating pricing power and efficiency.")
    elif margin > 15:
        bullets.append("✓ Healthy profitability with solid margin management.")
    elif margin > 5:
        bullets.append("⚖️ Reasonable margins, some room for improvement.")
    else:
        bullets.append("📉 Tight margins, scalability concerns present.")

    # Momentum Analysis
    if change_pct > 5:
        bullets.append("🔥 Strong momentum with significant recent gains.")
    elif change_pct > 2:
        bullets.append("💹 Positive momentum building above baseline levels.")
    elif change_pct < -5:
        bullets.append("📉 Weakness present, potential accumulation zone.")
    elif change_pct < -2:
        bullets.append("⚠️ Recent weakness creating possible entry opportunity.")
    else:
        bullets.append("➡️ Trading stable near recent support/resistance.")

    # Valuation Analysis
    if pe > 50:
        bullets.append("🎯 Premium valuation requiring exceptional execution.")
    elif pe > 35:
        bullets.append("📊 Elevated valuation, growth must justify multiples.")
    elif pe > 20:
        bullets.append("✓ Fair valuation relative to growth profile.")
    elif pe > 10:
        bullets.append("✅ Attractive valuation supporting upside potential.")
    else:
        bullets.append("💰 Deep value metrics, potential asymmetric returns.")

    # Risk Analysis
    if beta > 1.5:
        bullets.append("⚡ High volatility stock, suitable for risk-tolerant investors.")
    elif beta > 1.0:
        bullets.append("📈 Above-market volatility, more aggressive positioning.")
    elif beta < 0.7:
        bullets.append("🛡️ Defensive characteristics, lower volatility profile.")

    return bullets


def calculate_technical_metrics(historical_data: pd.DataFrame) -> Dict:
    """
    Calculate technical indicators from historical data.
    
    Args:
        historical_data: DataFrame with OHLCV data
        
    Returns:
        Dictionary with technical metrics
    """
    if historical_data.empty:
        return {}
    
    try:
        close = historical_data['Close']
        
        # Moving Averages
        sma20 = close.rolling(20).mean().iloc[-1]
        sma50 = close.rolling(50).mean().iloc[-1]
        sma200 = close.rolling(200).mean().iloc[-1]
        
        # Current Price
        current = close.iloc[-1]
        
        # Relative Strength
        gains = (close.diff().clip(lower=0)).rolling(14).mean()
        losses = (-close.diff().clip(upper=0)).rolling(14).mean()
        rs = gains / losses
        rsi = 100 - (100 / (1 + rs))
        
        return {
            "current_price": float(current),
            "sma20": float(sma20),
            "sma50": float(sma50),
            "sma200": float(sma200),
            "rsi": float(rsi.iloc[-1]),
            "52_week_high": float(close.rolling(252).max().iloc[-1]),
            "52_week_low": float(close.rolling(252).min().iloc[-1]),
            "volatility": float(close.pct_change().std() * 100),
        }
    except Exception as e:
        logger.warning(f"Error calculating technical metrics: {e}")
        return {}


if __name__ == "__main__":
    # Test the module
    test_ticker = "NVDA"
    print(f"\nFetching data for {test_ticker}...")
    
    data = get_stock_data(test_ticker)
    print(f"\nStock: {data['company']}")
    print(f"Price: {format_currency(data['current_price'])}")
    print(f"Change: {format_pct(data['change_pct'])}")
    print(f"Market Cap: {format_currency(data['market_cap'])}")
    
    bullets = build_insight_bullets(data)
    print("\nKey Insights:")
    for bullet in bullets:
        print(f"  {bullet}")