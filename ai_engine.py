"""
AI Engine Module
Provides AI-powered stock analysis using Groq's fast inference API.
Uses streaming for real-time response display and advanced LLM models.
"""

import os
import logging
from typing import Optional, Generator
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Groq client
try:
    from groq import Groq
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.warning("⚠️ GROQ_API_KEY not found in environment variables")
        client = None
    else:
        client = Groq(api_key=api_key)
        logger.info("✓ Groq client initialized successfully")
except ImportError:
    logger.error("✗ Groq library not installed. Install with: pip install groq")
    client = None


# ============================================================================
# ANALYSIS TEMPLATES
# ============================================================================

STOCK_ANALYSIS_SYSTEM_PROMPT = """You are an elite financial analyst and investment research expert with 20+ years of market experience.

Your analysis must be:
- PRACTICAL: Focus on actionable insights, not generic observations
- STRUCTURED: Organized clearly for both novice and advanced investors
- BALANCED: Present both bullish and bearish perspectives
- SPECIFIC: Use concrete metrics and data from the provided information
- CONCISE: Avoid filler text, be direct and meaningful

Always maintain professional tone while being accessible to all investor levels.
Provide a clear investment thesis supported by fundamental metrics."""


STOCK_ANALYSIS_PROMPT = """Analyze this stock in a highly structured format for investment decision-making.

STOCK DATA:
- Company: {company}
- Ticker: {ticker}
- Current Price: ${current_price:.2f}
- Price Change: {change_pct:.2f}%
- Market Cap: ${market_cap:,.0f}
- P/E Ratio: {pe_ratio:.2f}x
- 52 Week High: ${high_52:.2f}
- 52 Week Low: ${low_52:.2f}
- Sector: {sector}
- Beta: {beta:.2f}
- Profit Margin: {profit_margin:.1f}%
- Revenue Growth: {revenue_growth:.1f}%
- Earnings Growth: {earnings_growth:.1f}%

ANALYSIS REQUIREMENTS:
Provide analysis in this exact format with clear sections:

1. COMPANY OVERVIEW
(2-3 sentences about the company and its market position)

2. KEY METRICS ANALYSIS
(Analyze: valuation (P/E), growth metrics, profitability)

3. GROWTH DRIVERS
(List 3 key catalysts for future growth with brief explanation)

4. RISK FACTORS
(List 3 key risks with mitigation strategies if applicable)

5. BULL CASE
(Best case scenario: what needs to happen for strong returns)

6. BEAR CASE
(Worst case scenario: what could cause significant decline)

7. TECHNICAL INSIGHTS
(Current momentum, price positioning, volatility assessment)

8. AI INVESTMENT SCORE
Growth Potential: X/10 (explain)
Financial Strength: X/10 (explain)
Risk-Adjusted Return Potential: X/10 (explain)
Momentum Score: X/10 (explain)

9. FINAL VERDICT
(One of: STRONG BUY / BUY / HOLD / REDUCE / AVOID)

10. INVESTMENT THESIS
(4-5 sentence summary of why investors should consider this opportunity)

Be specific, data-driven, and avoid generic investment advice."""


# ============================================================================
# STREAMING ANALYSIS
# ============================================================================

def analyze_stock_streaming(stock_data: dict) -> Generator[str, None, None]:
    """
    Generate streaming AI analysis of stock with real-time response.
    
    This function uses Groq's fast inference to provide immediate, 
    actionable investment analysis with streaming output for better UX.
    
    Args:
        stock_data: Dictionary containing stock information
                   Required keys: company, ticker, current_price, change_pct,
                                market_cap, pe_ratio, high_52, low_52, sector,
                                beta, profit_margin, revenue_growth, earnings_growth
    
    Yields:
        Chunks of analysis text as they're generated
    """
    if not client:
        logger.error("❌ Groq client not initialized. Check GROQ_API_KEY.")
        yield "❌ AI analysis unavailable. Ensure GROQ_API_KEY is configured."
        return
    
    try:
        # Format the prompt with stock data
        analysis_prompt = STOCK_ANALYSIS_PROMPT.format(
            company=stock_data.get("company", "Unknown"),
            ticker=stock_data.get("ticker", "N/A"),
            current_price=stock_data.get("current_price", 0),
            change_pct=stock_data.get("change_pct", 0),
            market_cap=stock_data.get("market_cap", 0),
            pe_ratio=stock_data.get("pe_ratio", 0),
            high_52=stock_data.get("high_52", 0),
            low_52=stock_data.get("low_52", 0),
            sector=stock_data.get("sector", "N/A"),
            beta=stock_data.get("beta", 1.0),
            profit_margin=stock_data.get("profit_margin", 0),
            revenue_growth=stock_data.get("revenue_growth", 0),
            earnings_growth=stock_data.get("earnings_growth", 0),
        )
        
        logger.info(f"📡 Streaming analysis for {stock_data.get('ticker', 'unknown')}...")
        
        # Call Groq API with streaming
        with client.messages.stream(
            model="llama-3.1-8b-instant",  # Fast inference model
            max_tokens=2000,
            messages=[
                {
                    "role": "system",
                    "content": STOCK_ANALYSIS_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": analysis_prompt
                }
            ],
            temperature=0.7,  # Balanced creativity and consistency
            top_p=0.95,
        ) as stream:
            for text in stream.text_stream:
                yield text
        
        logger.info(f"✓ Analysis completed for {stock_data.get('ticker', 'unknown')}")
        
    except Exception as e:
        logger.error(f"✗ Streaming analysis error: {e}")
        yield f"\n\n❌ Analysis Error: {str(e)}\n\nPlease try again or check your API configuration."


def analyze_stock(stock_data: dict) -> str:
    """
    Generate complete AI analysis of stock (non-streaming version).
    
    Args:
        stock_data: Dictionary containing stock information
        
    Returns:
        Complete analysis as single string
    """
    if not client:
        logger.error("❌ Groq client not initialized")
        return "AI analysis unavailable. Configure GROQ_API_KEY in .env"
    
    try:
        # Format the prompt
        analysis_prompt = STOCK_ANALYSIS_PROMPT.format(
            company=stock_data.get("company", "Unknown"),
            ticker=stock_data.get("ticker", "N/A"),
            current_price=stock_data.get("current_price", 0),
            change_pct=stock_data.get("change_pct", 0),
            market_cap=stock_data.get("market_cap", 0),
            pe_ratio=stock_data.get("pe_ratio", 0),
            high_52=stock_data.get("high_52", 0),
            low_52=stock_data.get("low_52", 0),
            sector=stock_data.get("sector", "N/A"),
            beta=stock_data.get("beta", 1.0),
            profit_margin=stock_data.get("profit_margin", 0),
            revenue_growth=stock_data.get("revenue_growth", 0),
            earnings_growth=stock_data.get("earnings_growth", 0),
        )
        
        logger.info(f"🤖 Analyzing {stock_data.get('ticker', 'unknown')}...")
        
        # Call Groq API
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": STOCK_ANALYSIS_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": analysis_prompt
                }
            ],
            model="llama-3.1-8b-instant",
            max_tokens=2000,
            temperature=0.7,
            top_p=0.95,
        )
        
        analysis = chat_completion.choices[0].message.content
        logger.info(f"✓ Analysis completed")
        
        return analysis
        
    except Exception as e:
        logger.error(f"✗ Analysis error: {e}")
        return f"Analysis failed: {str(e)}"


# ============================================================================
# SPECIALIZED ANALYSIS FUNCTIONS
# ============================================================================

def generate_sentiment_analysis(stock_data: dict, news_items: Optional[list] = None) -> str:
    """
    Generate sentiment-driven investment analysis.
    
    Args:
        stock_data: Stock information
        news_items: Optional list of recent news headlines
        
    Returns:
        Sentiment analysis text
    """
    if not client:
        return "Sentiment analysis unavailable."
    
    news_context = ""
    if news_items:
        news_context = "Recent news: " + "; ".join(news_items[:5])
    
    prompt = f"""Based on the stock metrics and recent sentiment:

Stock: {stock_data['company']} ({stock_data['ticker']})
Price Movement: {stock_data['change_pct']:.2f}%
News Context: {news_context}

Provide a brief sentiment-driven investment perspective (2-3 paragraphs)."""
    
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            max_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        return "Unable to generate sentiment analysis."


def generate_risk_assessment(stock_data: dict) -> str:
    """
    Generate detailed risk assessment.
    
    Args:
        stock_data: Stock information
        
    Returns:
        Risk assessment text
    """
    if not client:
        return "Risk assessment unavailable."
    
    prompt = f"""Conduct a risk assessment for {stock_data['company']} ({stock_data['ticker']}):

Key Metrics:
- Beta: {stock_data['beta']}
- P/E Ratio: {stock_data['pe_ratio']:.2f}x
- Profit Margin: {stock_data['profit_margin']:.1f}%
- Growth Rate: {stock_data['revenue_growth']:.1f}%

Identify and rate the top 5 risks (low/medium/high) with mitigation strategies."""
    
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            max_tokens=800,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Risk assessment error: {e}")
        return "Unable to generate risk assessment."


def compare_stocks(stock_data_1: dict, stock_data_2: dict) -> str:
    """
    Generate comparative analysis of two stocks.
    
    Args:
        stock_data_1: First stock information
        stock_data_2: Second stock information
        
    Returns:
        Comparative analysis text
    """
    if not client:
        return "Comparison unavailable."
    
    prompt = f"""Compare these two stocks for investment purposes:

Stock 1: {stock_data_1['company']} ({stock_data_1['ticker']})
- Price: ${stock_data_1['current_price']:.2f}
- P/E: {stock_data_1['pe_ratio']:.2f}x
- Growth: {stock_data_1['revenue_growth']:.1f}%
- Margin: {stock_data_1['profit_margin']:.1f}%

Stock 2: {stock_data_2['company']} ({stock_data_2['ticker']})
- Price: ${stock_data_2['current_price']:.2f}
- P/E: {stock_data_2['pe_ratio']:.2f}x
- Growth: {stock_data_2['revenue_growth']:.1f}%
- Margin: {stock_data_2['profit_margin']:.1f}%

Provide a structured comparison highlighting which is better positioned and why."""
    
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            max_tokens=1000,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Comparison error: {e}")
        return "Unable to generate comparison."


# ============================================================================
# UTILITIES
# ============================================================================

def is_ai_available() -> bool:
    """Check if AI analysis is available."""
    return client is not None


def test_connection() -> bool:
    """Test Groq API connection."""
    if not client:
        return False
    
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": "Say 'OK'"}],
            model="llama-3.1-8b-instant",
            max_tokens=10,
        )
        logger.info("✓ Groq API connection successful")
        return True
    except Exception as e:
        logger.error(f"✗ Groq API connection failed: {e}")
        return False


if __name__ == "__main__":
    # Test the module
    print("Testing AI Engine...\n")
    
    if test_connection():
        print("✓ AI Engine is ready for analysis\n")
        
        # Test with sample data
        sample_stock = {
            "company": "NVIDIA Corporation",
            "ticker": "NVDA",
            "current_price": 124.58,
            "change_pct": 2.35,
            "market_cap": 3_060_000_000_000,
            "pe_ratio": 67.42,
            "high_52": 140.0,
            "low_52": 80.0,
            "sector": "Technology",
            "beta": 1.63,
            "profit_margin": 24.1,
            "revenue_growth": 78.4,
            "earnings_growth": 45.2,
        }
        
        print("Generating sample analysis...\n")
        print("=" * 60)
        
        # Use streaming for demo
        for chunk in analyze_stock_streaming(sample_stock):
            print(chunk, end="", flush=True)
        
        print("\n" + "=" * 60)
        print("\n✓ Analysis complete")
    else:
        print("✗ AI Engine not available")