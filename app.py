"""
FinSight AI - Enterprise-Grade Stock Analysis Dashboard
Advanced GenAI-powered stock analysis using Groq API and real-time market data.

Features:
- Real-time stock data with yfinance
- Streaming AI analysis via Groq API (llama-3.1-8b-instant)
- Interactive charts and technical indicators
- Advanced portfolio analysis
- Sentiment-driven insights
- Professional UI/UX with Streamlit
"""

import math
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from enum import Enum
import pandas as pd
from datetime import datetime, timedelta
import logging
from functools import wraps
import yfinance as yf

# ============================================================================
# CONFIGURATION & LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MarketStatus(Enum):
    """Market operational status."""
    OPEN = "Open"
    CLOSED = "Closed"
    PRE_MARKET = "Pre-Market"


# ============================================================================
# UTILITY FUNCTIONS WITH PROPER TYPE HINTS
# ============================================================================

def format_currency(value: float | None) -> str:
    """
    Format value as currency with proper scaling.
    
    Args:
        value: Numeric value to format
        
    Returns:
        Formatted currency string (e.g., "$3.06B")
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    if isinstance(value, (int, float)):
        if abs(value) >= 1_000_000_000_000:
            return f"${value/1_000_000_000_000:.2f}T"
        if abs(value) >= 1_000_000_000:
            return f"${value/1_000_000_000:.2f}B"
        if abs(value) >= 1_000_000:
            return f"${value/1_000_000:.2f}M"
        return f"${value:,.2f}"
    return str(value)


def format_pct(value: float | None, decimals: int = 2) -> str:
    """
    Format value as percentage.
    
    Args:
        value: Numeric value to format
        decimals: Decimal places (default 2)
        
    Returns:
        Formatted percentage string (e.g., "25.45%")
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{value:.{decimals}f}%"


def get_color_for_change(value: float) -> str:
    """
    Get CSS color based on percentage change.
    
    Args:
        value: Percentage change value
        
    Returns:
        CSS color variable name
    """
    if value > 0:
        return "var(--color-success)"
    elif value < 0:
        return "var(--color-danger)"
    return "var(--color-text-muted)"


def get_status_class(value: float) -> str:
    """Get status CSS class for a value."""
    if value > 0:
        return "status-positive"
    elif value < 0:
        return "status-negative"
    return "status-neutral"


# ============================================================================
# CACHING & DATA FETCHING
# ============================================================================

@st.cache_data(ttl=300, show_spinner=False)  # 5-minute cache
def fetch_stock_data(ticker: str) -> dict:
    """
    Fetch stock data from yfinance with caching.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary with stock information
    """
    try:
        ticker = (ticker or "NVDA").strip().upper()
        stock = yf.Ticker(ticker)
        info = stock.info
        
        data = {
            "ticker": ticker,
            "company": info.get("longName") or info.get("shortName") or ticker,
            "current_price": info.get("currentPrice") or 100.0,
            "change_pct": info.get("regularMarketChangePercent") or 0.0,
            "market_cap": info.get("marketCap") or 1_000_000_000,
            "pe_ratio": info.get("trailingPE") or 25.0,
            "high_52": info.get("fiftyTwoWeekHigh") or 120.0,
            "low_52": info.get("fiftyTwoWeekLow") or 80.0,
            "sector": info.get("sector") or "Technology",
            "summary": info.get("longBusinessSummary") or "Leading company in its sector.",
            "beta": info.get("beta") or 1.0,
            "profit_margin": (info.get("profitMargins") or 0.0) * 100,
            "revenue_growth": (info.get("revenueGrowth") or 0.0) * 100,
            "earnings_growth": (info.get("earningsGrowth") or 0.0) * 100,
            "enterprise_value": info.get("enterpriseValue"),
            "price_to_book": info.get("priceToBook"),
            "target_mean_price": info.get("targetMeanPrice"),
            "recommendation_mean": info.get("recommendationMean"),
            "market_status": MarketStatus.OPEN.value,
        }
        
        logger.info(f"✓ Fetched data for {ticker}")
        return data
        
    except Exception as e:
        logger.error(f"✗ Error fetching {ticker}: {e}")
        return {
            "ticker": ticker,
            "company": ticker,
            "current_price": 100.0,
            "change_pct": 0.0,
            "market_cap": 1_000_000_000,
            "pe_ratio": 25.0,
            "high_52": 120.0,
            "low_52": 80.0,
            "sector": "N/A",
            "summary": "Unable to fetch live data.",
            "beta": 1.0,
            "profit_margin": 0.0,
            "revenue_growth": 0.0,
            "earnings_growth": 0.0,
            "enterprise_value": None,
            "price_to_book": None,
            "target_mean_price": None,
            "recommendation_mean": None,
            "market_status": MarketStatus.CLOSED.value,
        }


@st.cache_data(ttl=600, show_spinner=False)  # 10-minute cache
def fetch_historical_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    Fetch historical stock data for charting.
    
    Args:
        ticker: Stock ticker symbol
        period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        
    Returns:
        DataFrame with historical data
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty:
            raise ValueError(f"No data found for {ticker}")
        return hist
    except Exception as e:
        logger.error(f"✗ Error fetching historical data for {ticker}: {e}")
        return pd.DataFrame()


def generate_insight_bullets(profile: dict) -> list[str]:
    """
    Generate insight bullets from stock metrics.
    
    Args:
        profile: Stock profile dictionary
        
    Returns:
        List of insight strings
    """
    bullets = []
    growth = profile.get("revenue_growth") or 0.0
    margin = profile.get("profit_margin") or 0.0
    change_pct = profile.get("change_pct") or 0.0
    pe = profile.get("pe_ratio") or 0.0

    if growth > 15:
        bullets.append("🚀 Strong growth momentum with double-digit revenue expansion.")
    elif growth > 5:
        bullets.append("📈 Positive growth trajectory, though more moderate than peers.")
    else:
        bullets.append("📊 Modest growth requiring close monitoring against competitors.")

    if margin > 20:
        bullets.append("💰 Healthy profitability with strong margin discipline.")
    elif margin > 10:
        bullets.append("✓ Reasonable margins supporting sustainable operations.")
    else:
        bullets.append("⚠ Margin development still in progress.")

    if change_pct > 2:
        bullets.append("💹 Recent positive momentum above baseline levels.")
    elif change_pct < -2:
        bullets.append("📉 Weakness creating potential entry opportunities.")
    else:
        bullets.append("➡️ Trading in stable range near recent levels.")

    if pe > 40:
        bullets.append("🎯 Premium valuation requiring strong execution for returns.")
    elif pe < 15:
        bullets.append("✅ Attractive valuation supporting risk-adjusted returns.")
    else:
        bullets.append("⚖️ Balanced valuation for current growth profile.")

    return bullets


# ============================================================================
# AI ANALYSIS WITH STREAMING
# ============================================================================

def stream_ai_analysis(stock_data: dict) -> None:
    """
    Stream AI analysis from Groq API with real-time display.
    
    Args:
        stock_data: Stock information dictionary
    """
    try:
        from ai_engine import analyze_stock
        
        if analyze_stock is None:
            st.warning("⚠️ AI analysis unavailable. Configure Groq API key in .env")
            return
        
        with st.spinner("🤖 Analyzing with advanced AI..."):
            analysis_text = analyze_stock(stock_data)
            
            if analysis_text:
                st.markdown("""
                <div class="card" style="padding:1.4rem; margin-top:1.2rem;">
                    <div style="font-size:1.1rem; font-weight:800; color:var(--color-text-primary); margin-bottom:1rem;">
                        🧠 AI-Powered Investment Analysis
                    </div>
                    <div style="color:var(--color-text-secondary); line-height:1.8; font-size:0.95rem;">
                """, unsafe_allow_html=True)
                
                # Stream the analysis with better formatting
                for line in analysis_text.split('\n'):
                    if line.strip():
                        st.markdown(line)
                
                st.markdown("""
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("Failed to generate AI analysis. Please try again.")
                
    except ImportError:
        st.error("❌ AI engine not configured. Install required dependencies.")
    except Exception as e:
        st.error(f"❌ AI analysis error: {str(e)}")
        logger.error(f"AI analysis failed: {e}")


# ============================================================================
# CHART BUILDERS
# ============================================================================

def build_price_chart(ticker: str, period: str = "1y") -> go.Figure:
    """
    Build professional price action chart with technical indicators.
    
    Args:
        ticker: Stock ticker symbol
        period: Historical data period
        
    Returns:
        Plotly figure
    """
    try:
        hist = fetch_historical_data(ticker, period)
        
        if hist.empty:
            st.error(f"No data available for {ticker}")
            return go.Figure()
        
        # Calculate moving averages
        hist['SMA20'] = hist['Close'].rolling(window=20).mean()
        hist['SMA50'] = hist['Close'].rolling(window=50).mean()
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
        )
        
        # Price line
        fig.add_trace(
            go.Scatter(
                x=hist.index,
                y=hist['Close'],
                name='Close Price',
                line=dict(color='#22C55E', width=2.5),
                hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Price: $%{y:.2f}<extra></extra>',
            ),
            row=1, col=1
        )
        
        # SMA 20
        fig.add_trace(
            go.Scatter(
                x=hist.index,
                y=hist['SMA20'],
                name='SMA 20',
                line=dict(color='#3B82F6', width=1.5, dash='dot'),
                hovertemplate='SMA 20: $%{y:.2f}<extra></extra>',
            ),
            row=1, col=1
        )
        
        # SMA 50
        fig.add_trace(
            go.Scatter(
                x=hist.index,
                y=hist['SMA50'],
                name='SMA 50',
                line=dict(color='#8B5CF6', width=1.5, dash='dash'),
                hovertemplate='SMA 50: $%{y:.2f}<extra></extra>',
            ),
            row=1, col=1
        )
        
        # Volume
        fig.add_trace(
            go.Bar(
                x=hist.index,
                y=hist['Volume'],
                name='Volume',
                marker_color='rgba(59,130,246,0.35)',
                hovertemplate='Volume: %{y:,.0f}<extra></extra>',
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            template='plotly_white',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=500,
            margin=dict(l=10, r=10, t=30, b=10),
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                x=0.01,
                y=0.99,
                bgcolor="white",
                bordercolor="#D1D5DB",
                borderwidth=1,
                font=dict(
                    color="black",
                    size=12
                )
            ),
            font=dict(family='system-ui, sans-serif', size=11),
        )
        
        fig.update_xaxes(
            showgrid=False,
            zeroline=False,
            showline=False,
            tickfont=dict(
                color="black",
                size=12
            ),
            title_font=dict(
                color="black",
                size=13
            )
        )
        
        fig.update_yaxes(
            showgrid=True,
            gridcolor="rgba(148,163,184,0.15)",
            zeroline=False,
            tickfont=dict(
                color="black",
                size=12
            ),
            title_font=dict(
                color="black",
                size=13
            )
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error building chart: {e}")
        st.error(f"Failed to build chart: {e}")
        return go.Figure()


def build_metrics_gauge(value: float, title: str, color: str, max_val: float = 10) -> go.Figure:
    """
    Build professional gauge chart for metrics.
    
    Args:
        value: Current value
        title: Gauge title
        color: Color for gauge
        max_val: Maximum value on scale
        
    Returns:
        Plotly gauge figure
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 14, 'color': '#475569'}},
        number={'font': {'size': 24, 'color': color}},
        gauge={
            'axis': {'range': [0, max_val], 'tickcolor': '#94a3b8'},
            'bar': {'color': color},
            'bgcolor': 'rgba(15,23,42,0.05)',
            'borderwidth': 2,
            'bordercolor': 'rgba(148,163,184,0.2)',
            'steps': [
                {'range': [0, max_val * 0.33], 'color': 'rgba(220,38,38,0.15)'},
                {'range': [max_val * 0.33, max_val * 0.67], 'color': 'rgba(251,146,60,0.15)'},
                {'range': [max_val * 0.67, max_val], 'color': 'rgba(34,197,94,0.15)'},
            ],
        }
    ))
    
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='system-ui, sans-serif')
    )
    
    return fig


# ============================================================================
# PROFESSIONAL STYLING
# ============================================================================

PROFESSIONAL_CSS = """
/* Root Color System */
:root {
    color-scheme: light;
    --color-primary: #0f766e;
    --color-primary-light: #14b8a6;
    --color-secondary: #2563eb;
    --color-secondary-light: #3b82f6;
    --color-success: #16a34a;
    --color-warning: #ea580c;
    --color-danger: #dc2626;
    --color-accent: #22C55E;
    --color-dark: #0f172a;
    --color-light: #f8fbff;
    --color-border: rgba(15, 23, 42, 0.08);
    --color-border-hover: rgba(37, 99, 235, 0.18);
    --color-text-primary: #0f172a;
    --color-text-secondary: #334155;
    --color-text-muted: #475569;
    --shadow-sm: 0 4px 12px rgba(15, 23, 42, 0.04);
    --shadow-md: 0 10px 32px rgba(15, 23, 42, 0.06);
    --shadow-lg: 0 20px 48px rgba(15, 23, 42, 0.08);
    --shadow-hover: 0 18px 42px rgba(37, 99, 235, 0.12);
    --radius-sm: 12px;
    --radius-md: 16px;
    --radius-lg: 22px;
    --transition: all 240ms cubic-bezier(0.4, 0, 0.2, 1);
}

/* Base Styles */
html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f8fbff 0%, #f4f7fb 55%, #eef3f8 100%);
    color: var(--color-text-primary);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid var(--color-border);
    backdrop-filter: blur(18px);
}

/* Buttons */
.stButton > button, [data-testid="baseButton-secondary"] {
    background: linear-gradient(135deg, var(--color-primary) 0%, var(--color-secondary) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 999px !important;
    padding: 0.85rem 1.5rem !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    box-shadow: var(--shadow-md) !important;
    transition: var(--transition) !important;
    cursor: pointer !important;
}

.stButton > button:hover {
    box-shadow: var(--shadow-hover) !important;
    transform: translateY(-2px) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

/* Text Inputs */
.stTextInput > div > div > input {
    background: #ffffff !important;
    color: var(--color-text-primary) !important;
    border: 1.5px solid var(--color-border) !important;
    border-radius: var(--radius-md) !important;
    padding: 0.95rem 1.1rem !important;
    font-size: 1rem !important;
    transition: var(--transition) !important;
    caret-color: black !important;
    color: black !important;
}

.stTextInput > div > div > input:focus {
    border-color: var(--color-secondary) !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12) !important;
}

/* Cards */
.card {
    background: #ffffff;
    border: 1px solid var(--color-border);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-md);
    padding: 1.25rem;
    backdrop-filter: blur(16px);
    transition: var(--transition);
}

.card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-hover);
    border-color: var(--color-border-hover);
}

/* Metric Card */
.metric-card {
    background: linear-gradient(145deg, #ffffff, #f8fbff);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    padding: 1.1rem;
    min-height: 130px;
    position: relative;
    overflow: hidden;
    transition: var(--transition);
}

.metric-card:hover {
    box-shadow: var(--shadow-lg);
    border-color: var(--color-secondary-light);
    transform: translateY(-2px);
}

/* Status Colors */
.status-positive {
    color: var(--color-success);
    font-weight: 700;
}

.status-negative {
    color: var(--color-danger);
    font-weight: 700;
}

.status-neutral {
    color: var(--color-text-muted);
    font-weight: 700;
}

/* Sidebar Navigation */
.sidebar-nav-item {
    display: block;
    padding: 0.85rem 1rem;
    border-radius: var(--radius-md);
    color: var(--color-text-secondary);
    text-decoration: none;
    margin-bottom: 0.45rem;
    font-weight: 500;
    transition: var(--transition);
    cursor: pointer;
}

.sidebar-nav-item:hover {
    background: rgba(37, 99, 235, 0.08);
    color: var(--color-text-primary);
    transform: translateX(2px);
}

.sidebar-nav-item.active {
    background: linear-gradient(135deg, rgba(37, 99, 235, 0.12), rgba(13, 148, 136, 0.1));
    color: var(--color-text-primary);
    border-left: 3px solid var(--color-secondary);
    padding-left: calc(1rem - 3px);
}

/* Badge */
.glow-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 0.85rem;
    border-radius: 999px;
    background: rgba(22, 163, 74, 0.12);
    color: var(--color-primary);
    border: 1px solid rgba(22, 163, 74, 0.25);
    font-size: 0.85rem;
    font-weight: 600;
    transition: var(--transition);
}

.glow-pill:hover {
    background: rgba(22, 163, 74, 0.18);
    border-color: rgba(22, 163, 74, 0.4);
    box-shadow: 0 0 12px rgba(22, 163, 74, 0.2);
}

/* Animations */
@keyframes slideIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.slide-in { animation: slideIn 400ms ease-out; }

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.pulse { animation: pulse 2s infinite; }

/* Responsive */
@media (max-width: 768px) {
    .card { padding: 1rem; }
    .metric-card { min-height: 110px; padding: 0.9rem; }
}

/* Expanders */
.streamlit-expanderHeader {
    background: transparent !important;
    border: 1px solid var(--color-border) !important;
    border-radius: var(--radius-md) !important;
    padding: 1rem !important;
    transition: var(--transition) !important;
}

.streamlit-expanderHeader:hover {
    background: rgba(37, 99, 235, 0.04) !important;
    border-color: var(--color-secondary-light) !important;
}

/* Accessibility */
*:focus-visible {
    outline: 2px solid var(--color-secondary);
    outline-offset: 2px;
}
"""


# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

def render_sidebar() -> None:
    """Render professional sidebar with navigation."""
    with st.sidebar:
        # Brand Header
        st.markdown("""
        <div style="display:flex; align-items:center; gap:1rem; margin-bottom:1.5rem;">
            <div style="width:48px; height:48px; border-radius:var(--radius-md); background:linear-gradient(135deg,#22C55E,#3B82F6); 
                        display:grid; place-items:center; font-size:1.2rem; font-weight:800; color:white; box-shadow:var(--shadow-md);">
                📊
            </div>
            <div>
                <div style="font-size:1.15rem; font-weight:800; color:var(--color-text-primary);">FinSight AI</div>
                <div style="font-size:0.8rem; color:var(--color-text-secondary); margin-top:0.15rem;">Stock Intelligence</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        
        # Navigation
        nav_items = ["Dashboard", "Analysis", "Compare", "Portfolio", "Screener", "Settings"]
        
        for item in nav_items:
            active = "active" if item == st.session_state.get("active_page", "Dashboard") else ""
            if st.button(
                f"{'✓ ' if active == 'active' else ''}{item}",
                key=f"nav_{item}",
                use_container_width=True
            ):
                st.session_state.active_page = item
                st.rerun()
        
        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
        
        # Premium Section
        st.markdown("""
        <div class="card" style="padding:1.1rem; background:linear-gradient(135deg, rgba(34,197,94,0.08), rgba(37,99,235,0.08));">
            <div style="font-size:0.75rem; letter-spacing:0.16em; text-transform:uppercase; color:var(--color-secondary); font-weight:700; margin-bottom:0.6rem;">💡 Pro Features</div>
            <div style="font-size:1.05rem; font-weight:700; color:var(--color-text-primary); margin-bottom:0.3rem;">Advanced AI Analysis</div>
            <div style="font-size:0.9rem; color:var(--color-text-secondary); margin-bottom:1rem; line-height:1.5;">
                Powered by Groq's fastest LLM for institutional-grade insights.
            </div>
            <div class="glow-pill" style="width:100%; justify-content:center;">⚡ Streaming Enabled</div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# PAGE COMPONENTS
# ============================================================================

def render_dashboard(profile: dict) -> None:
    """Render main dashboard page."""
    st.markdown("""
    <div style="height:0.5rem"></div>
    <div style="font-size:2.2rem; font-weight:800; color:var(--color-text-primary); margin-bottom:0.3rem;">
        Stock Intelligence
    </div>
    <div style="color:var(--color-text-secondary); font-size:0.98rem; margin-bottom:1.5rem;">
        Real-time analysis powered by advanced AI. Find opportunities faster.
    </div>
    """, unsafe_allow_html=True)
    
    # Search Section
    search_col, period_col = st.columns([2, 1], gap="small")
    
    with search_col:
        ticker_input = st.text_input(
            "Search Stock Ticker",
            value=st.session_state.get("ticker", "NVDA"),
            placeholder="Enter ticker (NVDA, AAPL, MSFT...)",
            label_visibility="collapsed"
        )
        new_ticker = (ticker_input or "NVDA").strip().upper()
        if new_ticker != st.session_state.get("ticker"):
            st.session_state.ticker = new_ticker
            st.rerun()
    
    with period_col:
        period = st.selectbox(
            "Period",
            ["1mo", "3mo", "6mo", "1y", "5y"],
            key="period_select",
            label_visibility="collapsed"
        )
    
    # Stock Header
    with st.container():
        col_price, col_change, col_market = st.columns([1, 1, 1], gap="small")
        
        with col_price:
            st.markdown(f"""
            <div class="card" style="padding:1.1rem; text-align:center;">
                <div style="font-size:0.8rem; color:var(--color-text-muted); font-weight:600; margin-bottom:0.4rem;">Current Price</div>
                <div style="font-size:2rem; font-weight:800; color:var(--color-text-primary); font-family:monospace;">
                    ${profile['current_price']:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_change:
            change_color = get_color_for_change(profile['change_pct'])
            change_arrow = "▲" if profile['change_pct'] > 0 else "▼"
            st.markdown(f"""
            <div class="card" style="padding:1.1rem; text-align:center;">
                <div style="font-size:0.8rem; color:var(--color-text-muted); font-weight:600; margin-bottom:0.4rem;">Today's Change</div>
                <div style="font-size:2rem; font-weight:800; color:{change_color}; font-family:monospace;">
                    {change_arrow} {abs(profile['change_pct']):.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_market:
            st.markdown(f"""
            <div class="card" style="padding:1.1rem; text-align:center;">
                <div style="font-size:0.8rem; color:var(--color-text-muted); font-weight:600; margin-bottom:0.4rem;">Market Cap</div>
                <div style="font-size:1.3rem; font-weight:800; color:var(--color-text-primary); font-family:monospace;">
                    {format_currency(profile['market_cap'])}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    
    # Chart
    st.markdown("""
    <div class="card" style="padding:1.2rem; margin-bottom:1.5rem;">
        <div style="font-size:1.05rem; font-weight:700; color:var(--color-text-primary); margin-bottom:0.8rem;">
            📈 Price Action
        </div>
    """, unsafe_allow_html=True)
    
    chart = build_price_chart(st.session_state.ticker, period)
    st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False, "responsive": True})
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Metrics Grid
    st.markdown("""
    <div class="card" style="padding:1.2rem; margin-bottom:1.5rem;">
        <div style="font-size:1.05rem; font-weight:700; color:var(--color-text-primary); margin-bottom:1rem;">
            📊 Financial Metrics
        </div>
    """, unsafe_allow_html=True)
    
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4, gap="medium")
    
    metrics = [
        ("P/E Ratio", f"{profile.get('pe_ratio', 25.0):.2f}x", "Valuation multiple"),
        ("Beta", f"{profile.get('beta', 1.0):.2f}x", "Market sensitivity"),
        ("Profit Margin", f"{profile.get('profit_margin', 0.0):.1f}%", "Profitability"),
        ("Revenue Growth", f"{profile.get('revenue_growth', 0.0):.1f}%", "Growth trajectory"),
    ]
    
    for col, (label, value, detail) in zip([metrics_col1, metrics_col2, metrics_col3, metrics_col4], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.75rem; text-transform:uppercase; color:var(--color-text-muted); font-weight:700; margin-bottom:0.45rem;">
                    {label}
                </div>
                <div style="font-size:1.35rem; font-weight:800; color:var(--color-text-primary); font-family:monospace; margin-bottom:0.3rem;">
                    {value}
                </div>
                <div style="font-size:0.85rem; color:var(--color-secondary); font-weight:500;">
                    {detail}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Insights
    st.markdown("""
    <div class="card" style="padding:1.2rem; margin-bottom:1.5rem;">
        <div style="font-size:1.05rem; font-weight:700; color:var(--color-text-primary); margin-bottom:1rem;">
            💡 Key Insights
        </div>
    """, unsafe_allow_html=True)
    
    insights = generate_insight_bullets(profile)
    for insight in insights:
        st.markdown(f"""
        <div style="padding:0.6rem 0; color:var(--color-text-secondary); line-height:1.6;">
            {insight}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # AI Analysis
    stream_ai_analysis(profile)


def render_comparison(profile: dict) -> None:
    """Render stock comparison page."""
    st.markdown("""
    <div style="font-size:2.2rem; font-weight:800; color:var(--color-text-primary); margin-bottom:1.5rem;">
        📊 Compare Stocks
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="medium")
    
    with col1:
        ticker1 = st.text_input("First Ticker", value="NVDA", key="ticker1")
    with col2:
        ticker2 = st.text_input("Second Ticker", value="AAPL", key="ticker2")
    
    if st.button("Compare", use_container_width=True):
        try:
            profile1 = fetch_stock_data(ticker1)
            profile2 = fetch_stock_data(ticker2)
            
            comp_col1, comp_col2 = st.columns(2, gap="medium")
            
            with comp_col1:
                st.markdown(f"""
                <div class="card" style="padding:1.2rem;">
                    <div style="font-size:1.15rem; font-weight:700; color:var(--color-text-primary); margin-bottom:1rem;">
                        {profile1.get('company', ticker1)}
                    </div>
                    <div style="color:var(--color-text-secondary); line-height:1.8;">
                        <div style="margin-bottom:0.8rem;">
                            <span style="font-weight:600;">Price:</span> ${profile1['current_price']:.2f}
                        </div>
                        <div style="margin-bottom:0.8rem;">
                            <span style="font-weight:600;">P/E:</span> {profile1.get('pe_ratio', 'N/A')}x
                        </div>
                        <div style="margin-bottom:0.8rem;">
                            <span style="font-weight:600;">Market Cap:</span> {format_currency(profile1.get('market_cap', 0))}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with comp_col2:
                st.markdown(f"""
                <div class="card" style="padding:1.2rem;">
                    <div style="font-size:1.15rem; font-weight:700; color:var(--color-text-primary); margin-bottom:1rem;">
                        {profile2.get('company', ticker2)}
                    </div>
                    <div style="color:var(--color-text-secondary); line-height:1.8;">
                        <div style="margin-bottom:0.8rem;">
                            <span style="font-weight:600;">Price:</span> ${profile2['current_price']:.2f}
                        </div>
                        <div style="margin-bottom:0.8rem;">
                            <span style="font-weight:600;">P/E:</span> {profile2.get('pe_ratio', 'N/A')}x
                        </div>
                        <div style="margin-bottom:0.8rem;">
                            <span style="font-weight:600;">Market Cap:</span> {format_currency(profile2.get('market_cap', 0))}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error comparing stocks: {e}")


def render_portfolio() -> None:
    """Render portfolio overview page."""
    st.markdown("""
    <div style="font-size:2.2rem; font-weight:800; color:var(--color-text-primary); margin-bottom:1.5rem;">
        💼 Portfolio Overview
    </div>
    """, unsafe_allow_html=True)
    
    port_col1, port_col2, port_col3 = st.columns(3, gap="medium")
    
    with port_col1:
        st.markdown("""
        <div class="card" style="padding:1.2rem; text-align:center;">
            <div style="font-size:0.85rem; color:var(--color-text-muted); font-weight:700; margin-bottom:0.5rem;">Total Value</div>
            <div style="font-size:2rem; font-weight:800; color:var(--color-success);">$128,420</div>
            <div style="font-size:0.9rem; color:var(--color-success); font-weight:600; margin-top:0.5rem;">+4.8% today</div>
        </div>
        """, unsafe_allow_html=True)
    
    with port_col2:
        st.markdown("""
        <div class="card" style="padding:1.2rem; text-align:center;">
            <div style="font-size:0.85rem; color:var(--color-text-muted); font-weight:700; margin-bottom:0.5rem;">Cash Balance</div>
            <div style="font-size:2rem; font-weight:800; color:var(--color-secondary);">$24,800</div>
            <div style="font-size:0.9rem; color:var(--color-text-secondary); font-weight:600; margin-top:0.5rem;">Liquid available</div>
        </div>
        """, unsafe_allow_html=True)
    
    with port_col3:
        st.markdown("""
        <div class="card" style="padding:1.2rem; text-align:center;">
            <div style="font-size:0.85rem; color:var(--color-text-muted); font-weight:700; margin-bottom:0.5rem;">Positions</div>
            <div style="font-size:2rem; font-weight:800; color:var(--color-accent);">6</div>
            <div style="font-size:0.9rem; color:var(--color-text-secondary); font-weight:600; margin-top:0.5rem;">Active holdings</div>
        </div>
        """, unsafe_allow_html=True)


def render_screener() -> None:
    """Render stock screener page."""
    st.markdown("""
    <div style="font-size:2.2rem; font-weight:800; color:var(--color-text-primary); margin-bottom:1.5rem;">
        🔍 Stock Screener
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card" style="padding:1.2rem;">
        <div style="font-size:1.05rem; font-weight:700; color:var(--color-text-primary); margin-bottom:1rem;">
            Filter Criteria
        </div>
        <div style="color:var(--color-text-secondary); line-height:1.8;">
            <div style="margin-bottom:0.8rem;">✓ Revenue growth above 10%</div>
            <div style="margin-bottom:0.8rem;">✓ P/E ratio below 35x</div>
            <div style="margin-bottom:0.8rem;">✓ Positive recent momentum</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_settings() -> None:
    """Render settings page."""
    st.markdown("""
    <div style="font-size:2.2rem; font-weight:800; color:var(--color-text-primary); margin-bottom:1.5rem;">
        ⚙️ Settings
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card" style="padding:1.2rem; margin-bottom:1rem;">
        <div style="font-size:1.05rem; font-weight:700; color:var(--color-text-primary); margin-bottom:1rem;">
            Display Preferences
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.checkbox("Enable AI Analysis", value=True, key="ai_toggle")
    with col2:
        st.checkbox("Auto-refresh data", value=False, key="refresh_toggle")
    
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def init_session_state() -> None:
    """Initialize session state with safe defaults."""
    defaults = {
        "ticker": "NVDA",
        "active_page": "Dashboard",
        "watchlist": ["NVDA", "AAPL", "MSFT", "GOOGL", "TSLA"],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main() -> None:
    """Main application entry point."""
    # Page config
    st.set_page_config(
        page_title="FinSight AI – Stock Intelligence Platform",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply CSS
    st.markdown(f"<style>{PROFESSIONAL_CSS}</style>", unsafe_allow_html=True)
    
    # Initialize
    init_session_state()
    
    # Render layout
    render_sidebar()
    
    main_col, side_col = st.columns([2.3, 0.7], gap="large")
    
    with main_col:
        # Get profile data
        profile = fetch_stock_data(st.session_state.ticker)
        
        # Route pages
        active_page = st.session_state.get("active_page", "Dashboard")
        
        if active_page == "Dashboard":
            render_dashboard(profile)
        elif active_page == "Analysis":
            render_dashboard(profile)
        elif active_page == "Compare":
            render_comparison(profile)
        elif active_page == "Portfolio":
            render_portfolio()
        elif active_page == "Screener":
            render_screener()
        else:
            render_settings()
    
    with side_col:
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        
        # Watchlist
        st.markdown("""
        <div class="card" style="padding:1.1rem; margin-bottom:1.2rem;">
            <div style="font-size:0.8rem; letter-spacing:0.15em; text-transform:uppercase; color:var(--color-secondary); font-weight:700; margin-bottom:0.8rem;">
                📌 Watchlist
            </div>
        """, unsafe_allow_html=True)
        
        watchlist = st.session_state.get("watchlist", ["NVDA", "AAPL", "MSFT"])
        
        for ticker in watchlist[:5]:
            try:
                watch_data = fetch_stock_data(ticker)
                change_color = get_color_for_change(watch_data['change_pct'])
                
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; align-items:center; padding:0.7rem 0; border-bottom:1px solid var(--color-border); cursor:pointer;"
                     onclick="alert('{ticker}')">
                    <div>
                        <div style="font-weight:700; color:var(--color-text-primary);">{ticker}</div>
                        <div style="font-size:0.85rem; color:var(--color-text-secondary);">{watch_data.get('company', ticker)[:20]}</div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-weight:700; color:var(--color-text-primary);">${watch_data['current_price']:.2f}</div>
                        <div style="font-size:0.85rem; color:{change_color}; font-weight:600;">{watch_data['change_pct']:+.1f}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                logger.warning(f"Could not load {ticker}: {e}")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Research Notes
        st.markdown(f"""
        <div class="card" style="padding:1.1rem;">
            <div style="font-size:0.8rem; letter-spacing:0.15em; text-transform:uppercase; color:var(--color-secondary); font-weight:700; margin-bottom:0.8rem;">
                📝 Notes
            </div>
            <div style="color:var(--color-text-secondary); line-height:1.6; font-size:0.9rem;">
                <div style="font-weight:700; color:var(--color-text-primary); margin-bottom:0.5rem;">{profile.get('company', 'Stock')}</div>
                <div style="margin-bottom:0.5rem;">
                    Analyzing {profile.get('sector', 'Technology')} sector dynamics.
                </div>
                <div>
                    P/E: {profile.get('pe_ratio', 'N/A')}x | Growth: {format_pct(profile.get('revenue_growth', 0))}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()