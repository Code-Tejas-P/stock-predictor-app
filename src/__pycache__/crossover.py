import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import ta

# ---------------- Pivot Calculation ----------------
def calculate_pivots(df):
    #df['Pivot'] = (df['Prev_High'] + df['Prev_Low'] + df['Prev_Close']) / 3
    #df['Range'] = df['Prev_High'] - df['Prev_Low']

    # Supports
    df['S1'] = (2 * df['Pivot']) - (df['High']).shift(1)
    df['S2'] = df['Pivot'] - (df['Range']).shift(1)
    df['S3'] = (df['Low']).shift(1) - 2 * ((df['High']).shift(1) - df['Pivot'])
    df['S4'] = df['Pivot'] * 3 - ((3 * df['High']).shift(1) - (df['Low'].shift(1)))
    df['S5'] = df['Pivot'] * 4 - ((4 * df['High']).shift(1) - (df['Low'].shift(1)))
    
    # Resistances
    df['R1'] = (2 * df['Pivot']) - (df['Low']).shift(1)
    df['R2'] = df['Pivot'] + (df['Range']).shift(1)
    df['R3'] = (df['High']).shift(1) + 2 * (df['Pivot'] - (df['Low'].shift(1)))
    df['R4'] = df['Pivot'] * 3 + ((df['High']).shift(1) - 3 * (df['Low'].shift(1)))
    df['R5'] = df['Pivot'] * 4 + ((df['High']).shift(1) - 4 * (df['Low'].shift(1)))
    
    return df
    
# ---------------- Backtest Strategy ----------------
def backtest_strategy(ticker="^NSEI", period="3d", interval="5m", output_file="strategy_results.xlsx"):

    df = intraday(ticker, period, interval, output_file="NSEI_intraday_data.xlsx")
    # EMA9
    df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()

    # ADX(14)
    adx_indicator = ta.trend.ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    df['ADX'] = adx_indicator.adx()
    adx_threshold = 19.5

    # Conditions for bullish/bearish candles
    df['is_bullish'] = (df['Close'] > df['Open']) & (df['Close'] > df['EMA9']) & (df['Low'] > df['EMA9']) & (df['ADX'] > adx_threshold)
    df['is_bearish'] = (df['Close'] < df['Open']) & (df['Close'] < df['EMA9']) & (df['High'] < df['EMA9']) & (df['ADX'] > adx_threshold)

    # Candle between pivot levels
    df['candle_between_levels'] = (
        ((df['Low'] >= df['S5']) & (df['High'] <= df['S4'])) |
        ((df['Low'] >= df['S4']) & (df['High'] <= df['S3'])) |
        ((df['Low'] >= df['S3']) & (df['High'] <= df['S2'])) |
        ((df['Low'] >= df['S2']) & (df['High'] <= df['S1'])) |
        ((df['Low'] >= df['S1']) & (df['High'] <= df['Pivot'])) |
        ((df['Low'] >= df['Pivot']) & (df['High'] <= df['R1'])) |
        ((df['Low'] >= df['R1']) & (df['High'] <= df['R2'])) |
        ((df['Low'] >= df['R2']) & (df['High'] <= df['R3'])) |
        ((df['Low'] >= df['R3']) & (df['High'] <= df['R4'])) |
        ((df['Low'] >= df['R4']) & (df['High'] <= df['R5']))
    )

    # Generate signals
    df['Signal'] = 0
    df.loc[df['is_bullish'] & df['candle_between_levels'], 'Signal'] = 1
    df.loc[df['is_bearish'] & df['candle_between_levels'], 'Signal'] = -1

    # Strategy returns
    df['Return'] = df['Close'].pct_change()
    df['Strategy_Return'] = df['Return'] * df['Signal'].shift(1)
    df['Cumulative_Market'] = (1 + df['Return']).cumprod()
    df['Cumulative_Strategy'] = (1 + df['Strategy_Return']).cumprod()

    # Accuracy
    total_signals = df[df['Signal'] != 0].shape[0]
    profitable_trades = df[df['Strategy_Return'] > 0].shape[0]
    accuracy = (profitable_trades / total_signals * 100) if total_signals > 0 else 0
    print(f"\n{'='*30}")
    print(f"    BACKTEST RESULTS")
    print(f"{'='*30}")
    print(f"Total Signals: {total_signals}")
    print(f"Profitable Trades: {profitable_trades}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"{'='*30}\n")
    # Save to Excel
    df.to_excel(output_file, index=True)
    print(f"Backtest saved to: {output_file}")

    return df

def get_prev_data(ticker="^NSEI", period="1mo", interval="1d", output_file="NSEI_prev_data.xlsx"):
    df_pivot = yf.download(ticker, period=period, interval=interval, progress=False)
    df_pivot = df_pivot[['Open', 'High', 'Low', 'Close']]
    df_pivot.columns = ['Open', 'High', 'Low', 'Close']
    df_pivot.index = pd.to_datetime(df_pivot.index.tz_localize('utc').tz_convert('Asia/Kolkata'))
    df_pivot.index = pd.to_datetime(df_pivot.index).tz_localize(None)
    df_pivot['Pivot'] = (df_pivot['High'] + df_pivot['Low'] + df_pivot['Close']).shift(1) / 3
    df_pivot['Range'] = df_pivot['High'] - df_pivot['Low']
    df_pivot = calculate_pivots(df_pivot)
    #df_pivot['S1'] = (2 * df_pivot['Pivot']) - (df_pivot['High']).shift(1)
    df_pivot.to_excel(output_file, index=True)
    print(f"Backtest saved to: {output_file}") 
    return df_pivot

def intraday(ticker="^NSEI", period="1mo", interval="1d", output_file="data.xlsx"):
    df_data = yf.download(ticker, period=period, interval=interval, progress=False)
    df_data = df_data[['Open', 'High', 'Low', 'Close']]
    df_data.columns = ['Open', 'High', 'Low', 'Close']
    
    df_data.index = pd.to_datetime(df_data.index.tz_convert('Asia/Kolkata'))
    df_data.index = pd.to_datetime(df_data.index).tz_localize(None)
    
    # Store the full timestamp in a column
    df_data['Timestamp'] = df_data.index
    
    # Change index to date only
    df_data.index = df_data.index.date
    
    # Get previous data
    df_prev = get_prev_data(ticker, period="1mo", interval="1d", output_file="prev_data.xlsx")
    
    # Create date-based lookup for all previous data columns
    prev_by_date = df_prev.groupby(df_prev.index.date).first()  # Get first occurrence of each date
    
    # Map all previous data columns
    df_data['Pivot'] = df_data.index.map(prev_by_date['Pivot'])
    df_data['Prev_High'] = df_data.index.map(prev_by_date['High'])
    df_data['Prev_Low'] = df_data.index.map(prev_by_date['Low'])
    df_data['Prev_Close'] = df_data.index.map(prev_by_date['Close'])
    df_data['Range'] = df_data.index.map(prev_by_date['Range'])
    df_data['S1'] = df_data.index.map(prev_by_date['S1'])
    df_data['S2'] = df_data.index.map(prev_by_date['S2'])
    df_data['S3'] = df_data.index.map(prev_by_date['S3'])
    df_data['S4'] = df_data.index.map(prev_by_date['S4'])
    df_data['S5'] = df_data.index.map(prev_by_date['S5'])
    df_data['R1'] = df_data.index.map(prev_by_date['R1'])
    df_data['R2'] = df_data.index.map(prev_by_date['R2'])
    df_data['R3'] = df_data.index.map(prev_by_date['R3'])
    df_data['R4'] = df_data.index.map(prev_by_date['R4'])
    df_data['R5'] = df_data.index.map(prev_by_date['R5'])
    #df_data = calculate_pivots(df_data)

    # Reorder columns
    #df_data = df_data[['Timestamp', 'Open', 'High', 'Low', 'Close', 'Prev_High', 'Prev_Low', 'Prev_Close', 'Pivot']]
    
    # Check results
    matching_count = df_data['Pivot'].notna().sum()
    print(f"Previous data added for {matching_count} records based on date matching.")
    
    # Save with date index and all columns
    df_data.to_excel(output_file, index=True)
    print(f"Intraday data saved to: {output_file}")
    
    return df_data

# ---------------- Run Backtest ----------------
if __name__ == "__main__":
    print("\n" + "="*60)
    print("          PIVOT POINT TRADING STRATEGY")
    print("            EMA9 + ADX + Pivot Levels")
    print("="*60)
    ticker = input("\nEnter stock ticker (e.g., ^NSEI, RELIANCE.NS, AAPL): ").strip().upper()
    period = input("Enter data period (e.g., 3d, 5d, 1mo, 3mo): ").strip()
    interval = input("Enter data interval (e.g., 5m, 15m, 30m, 1h, 1d): ").strip()
    output_file = input("Enter output Excel file name (e.g., strategy_results.xlsx): ").strip()
    if not output_file:
        output_file = "strategy_results.xlsx"
    backtest_strategy(
        ticker=ticker, 
        period= period, 
        interval=interval, 
        output_file="NSEI_quick_test.xlsx")
   #backtest_strategy("^NSEI", period="1mo", interval="5m", output_file="NSEI_strategy.xlsx")
   #get_prev_data("^NSEI", period="10d", interval="1d", output_file="NSEI_prev_data.xlsx")
    #intraday("^NSEI", period="3d", interval="5m", output_file="NSEI_intraday_data.xlsx")