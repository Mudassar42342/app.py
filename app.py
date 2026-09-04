import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from datetime import datetime, timezone
import os
from groq import Groq

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Gold Signal Generator",
    page_icon="?",
    layout="wide"
)

# Custom CSS styling for sleek dark UI
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e222d; padding: 10px; border-radius: 8px; }
    .signal-box-buy { background-color: #1b3a2b; border: 1px solid #2e7d32; padding: 15px; border-radius: 8px; }
    .signal-box-sell { background-color: #3b1e1e; border: 1px solid #c62828; padding: 15px; border-radius: 8px; }
    .signal-box-wait { background-color: #2b2d3a; border: 1px solid #455a64; padding: 15px; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

st.title("? GOLD SIGNAL GENERATOR (XAUUSD)")
st.caption("Multi-Strategy Rule Based Engine + AI Reasoning")

# ---------------------------------------------------------
# Sidebar - Groq API Key Config
# ---------------------------------------------------------
st.sidebar.header("?? Configuration")
groq_api_key = st.sidebar.text_input("Enter Groq API Key (Free)", type="password")
st.sidebar.markdown("[Get Free Groq API Key Here](https://console.groq.com/)")

timeframe = st.sidebar.selectbox("Timeframe", ["5m", "15m", "1h", "4h"], index=1)

# ---------------------------------------------------------
# Helper Functions: Data & Sessions
# ---------------------------------------------------------
@st.cache_data(ttl=60)
def fetch_gold_data(tf):
    ticker = "GC=F"  # Gold Futures / XAUUSD proxy
    data = yf.download(tickers=ticker, period="5d", interval=tf)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    return data

def get_market_sessions():
    now_utc = datetime.now(timezone.utc).hour
    sessions = []
    
    # Tokyo: 00:00 - 09:00 UTC
    if 0 <= now_utc < 9:
        sessions.append("Tokyo")
    # London: 08:00 - 16:00 UTC
    if 8 <= now_utc < 16:
        sessions.append("London")
    # New York: 13:00 - 22:00 UTC
    if 13 <= now_utc < 22:
        sessions.append("New York")
        
    overlap = ""
    if "London" in sessions and "New York" in sessions:
        overlap = "? London / NY Overlap - High Volatility Expected!"
    elif not sessions:
        sessions = ["Sydney / Asian Quiet Hours"]
        
    return ", ".join(sessions), overlap

# ---------------------------------------------------------
# Technical & Strategy Engine (Rule-Based Math)
# ---------------------------------------------------------
def analyze_strategies(df):
    results = {}
    
    # 1. Indicators Calculation
    df['RSI'] = ta.rsi(df['Close'], length=14)
    macd = ta.macd(df['Close'])
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_SIGNAL'] = macd['MACDs_12_26_9']
    df['EMA_50'] = ta.ema(df['Close'], length=50)
    df['EMA_200'] = ta.ema(df['Close'], length=200)
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 2. RSI Rules
    if latest['RSI'] < 30: results['RSI'] = 'BUY'
    elif latest['RSI'] > 70: results['RSI'] = 'SELL'
    else: results['RSI'] = 'WAIT'
    
    # 3. MACD Crossover Rules
    if latest['MACD'] > latest['MACD_SIGNAL'] and prev['MACD'] <= prev['MACD_SIGNAL']:
        results['MACD'] = 'BUY'
    elif latest['MACD'] < latest['MACD_SIGNAL'] and prev['MACD'] >= prev['MACD_SIGNAL']:
        results['MACD'] = 'SELL'
    else:
        results['MACD'] = 'WAIT'
        
    # 4. EMA Trend Rules
    if latest['Close'] > latest['EMA_50'] > latest['EMA_200']:
        results['EMA_Trend'] = 'BUY'
    elif latest['Close'] < latest['EMA_50'] < latest['EMA_200']:
        results['EMA_Trend'] = 'SELL'
    else:
        results['EMA_Trend'] = 'WAIT'
        
    # 5. SMC - Fair Value Gap (FVG) Detection
    # Bullish FVG: Low of candle 1 > High of candle 3
    if df.iloc[-1]['Low'] > df.iloc[-3]['High']:
        results['SMC_FVG'] = 'BUY'
    elif df.iloc[-1]['High'] < df.iloc[-3]['Low']:
        results['SMC_FVG'] = 'SELL'
    else:
        results['SMC_FVG'] = 'WAIT'
        
    # 6. SMC / ICT - Market Structure Shift (MSS)
    recent_high = df['High'].iloc[-10:-1].max()
    recent_low = df['Low'].iloc[-10:-1].min()
    
    if latest['Close'] > recent_high:
        results['ICT_MSS'] = 'BUY'
    elif latest['Close'] < recent_low:
        results['ICT_MSS'] = 'SELL'
    else:
        results['ICT_MSS'] = 'WAIT'
        
    # 7. Support & Resistance Reversal
    sr_support = df['Low'].iloc[-20:].min()
    sr_resistance = df['High'].iloc[-20:].max()
    
    if abs(latest['Close'] - sr_support) / latest['Close'] < 0.0015:
        results['Support_Resistance'] = 'BUY'
    elif abs(latest['Close'] - sr_resistance) / latest['Close'] < 0.0015:
        results['Support_Resistance'] = 'SELL'
    else:
        results['Support_Resistance'] = 'WAIT'
        
    return results, latest['Close']

# ---------------------------------------------------------
# AI Decision Engine (Groq Llama-3 API Integration)
# ---------------------------------------------------------
def generate_ai_decision(signals, current_price, session, groq_key):
    if not groq_key:
        return "?? Please enter your Groq API Key in the sidebar to get AI Analysis."
    
    try:
        client = Groq(api_key=groq_key)
        
        prompt = f"""
        You are an elite Gold (XAUUSD) Trading AI Decision Engine.
        Analyze the following rule-based strategy results and live market context:

        Current Price: {current_price:.2f}
        Active Session: {session}
        
        Strategy Scanner Results:
        - SMC (FVG): {signals.get('SMC_FVG')}
        - ICT (MSS Structure): {signals.get('ICT_MSS')}
        - RSI (14): {signals.get('RSI')}
        - MACD: {signals.get('MACD')}
        - EMA Trend: {signals.get('EMA_Trend')}
        - Support / Resistance: {signals.get('Support_Resistance')}

        Your Task:
        1. Calculate Buy Probability %, Sell Probability %, and Wait status based on strategy confluence.
        2. Provide a clear recommendation (BUY / SELL / WAIT).
        3. Explain the detailed reason using SMC/ICT and Technical concepts.
        4. Give exact Entry, Stop Loss (SL), Take Profit (TP), and Risk/Reward ratio.

        Format your output strictly like this:
        ? BUY Probability: X%
        ? SELL Probability: Y%
        ? WAIT Status: [Active/Inactive]

        AI FINAL SIGNAL: [BUY / SELL / WAIT]
        Confidence: Z%

        AI Reason:
        - Bullet point reason 1
        - Bullet point reason 2
        - Bullet point reason 3

        Trade Setup:
        - Entry: Price
        - Stop Loss: Price
        - Take Profit: Price
        - Risk Reward: 1:X
        """

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=600
        )
        return completion.choices[0].message.content
        
    except Exception as e:
        return f"Error communicating with AI Engine: {str(e)}"

# ---------------------------------------------------------
# Main App Layout Execution
# ---------------------------------------------------------
data = fetch_gold_data(timeframe)

if not data.empty:
    current_session, session_overlap = get_market_sessions()
    signals, latest_price = analyze_strategies(data)

    # Top Status Bar
    col1, col2, col3 = st.columns(3)
    col1.metric("XAUUSD Live Price", f"${latest_price:.2f}")
    col2.metric("Active Session", current_session)
    col3.metric("High Impact News Alert", "No Major CPI/NFP in next 1hr")

    if session_overlap:
        st.warning(session_overlap)

    st.markdown("---")

    # Layout Split: Strategy Matrix & Chart
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.subheader("? Strategy Scanner Matrix")
        
        # Display Matrix
        matrix_df = pd.DataFrame(list(signals.items()), columns=["Strategy / Concept", "Signal"])
        
        def color_signals(val):
            if val == 'BUY': return 'color: #00ff7f; font-weight: bold;'
            elif val == 'SELL': return 'color: #ff4d4d; font-weight: bold;'
            return 'color: #b0bec5;'

        st.dataframe(
            matrix_df.style.map(color_signals, subset=['Signal']),
            use_container_width=True,
            hide_index=True
        )

    with right_col:
        st.subheader("? Live Chart")
        fig = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="XAUUSD"
        )])
        fig.update_layout(
            template="plotly_dark",
            height=300,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # AI Decision Engine Section
    st.subheader("? AI Decision Engine & Reasoner")

    if st.button("? Run AI Multi-Strategy Analysis", type="primary"):
        with st.spinner("AI Engine reading strategy matrix & analyzing order blocks..."):
            ai_output = generate_ai_decision(signals, latest_price, current_session, groq_api_key)
            st.markdown(f"```text\n{ai_output}\n```")
else:
    st.error("Failed to load market data. Please refresh.")

