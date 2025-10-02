import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import ta

def generate_signals(df):
    """
    Generates trading signals based on a strategy that "blocks" a position until the price crosses the EMA9.

    Args:
        df (pd.DataFrame): DataFrame with 'Close', 'EMA9', 'is_bullish', 'is_bearish',
                          'candle_between_levels', and 'is_hammer' columns.

    Returns:
        pd.DataFrame: The input DataFrame with a new 'Signal' column.
    """
    df['Signal'] = 0
    
    block_signal = None

    for i in range(2, len(df)):
        row = df.iloc[i]

        close = row['Close']
        ema9 = row['EMA9']
        
        if block_signal == "bearish" and close > ema9:
            block_signal = None
        if block_signal == "bullish" and close < ema9:
            block_signal = None

        if block_signal is None:
            if row['is_bullish'] and row['candle_between_levels']:
                df.iloc[i, df.columns.get_loc('Signal')] = 1
                block_signal = "bullish"
            elif row['is_bearish'] and row['candle_between_levels'] and not row['is_hammer']:
                df.iloc[i, df.columns.get_loc('Signal')] = -1
                block_signal = "bearish"

    return df

def calutlate_profit_and_stoploss(df):
    """
    Calculates Take Profit and Stop Loss levels based on entry signals and a 1:1 ratio.
    """
    df['Take_Profit'] = pd.NA
    df['Stop_Loss'] = pd.NA

    bullish_mask = (df['is_bullish'] == True) & (df['candle_between_levels'] == True)
    bearish_mask = (df['is_bearish'] == True) & (df['candle_between_levels'] == True)

    bullish_stop_loss_distance = df.loc[bullish_mask, 'Close'] - df.loc[bullish_mask, 'Low'].shift(1)
    df.loc[bullish_mask, 'Stop_Loss'] = df.loc[bullish_mask, 'Low'].shift(1)
    df.loc[bullish_mask, 'Take_Profit'] = df.loc[bullish_mask, 'Close'] + bullish_stop_loss_distance

    bearish_stop_loss_distance = df.loc[bearish_mask, 'High'].shift(1) - df.loc[bearish_mask, 'Close']
    df.loc[bearish_mask, 'Stop_Loss'] = df.loc[bearish_mask, 'High'].shift(1)
    df.loc[bearish_mask, 'Take_Profit'] = df.loc[bearish_mask, 'Close'] - bearish_stop_loss_distance
    
    return df

# ---------------- Pivot Calculation ----------------
def calculate_pivots(df):
    """
    Calculates pivot points and corresponding support and resistance levels.
    """
    df['Pivot'] = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
    df['Range'] = df['High'].shift(1) - df['Low'].shift(1)

    df['S1'] = (2 * df['Pivot']) - df['High'].shift(1)
    df['S2'] = df['Pivot'] - df['Range']
    df['S3'] = df['Low'].shift(1) - 2 * (df['High'].shift(1) - df['Pivot'])
    df['S4'] = df['Pivot'] * 3 - (3 * df['High'].shift(1) - df['Low'].shift(1))
    df['S5'] = df['Pivot'] * 4 - (4 * df['High'].shift(1) - df['Low'].shift(1))
    
    df['R1'] = (2 * df['Pivot']) - df['Low'].shift(1)
    df['R2'] = df['Pivot'] + df['Range']
    df['R3'] = df['High'].shift(1) + 2 * (df['Pivot'] - df['Low'].shift(1))
    df['R4'] = df['Pivot'] * 3 + (df['High'].shift(1) - 3 * df['Low'].shift(1))
    df['R5'] = df['Pivot'] * 4 + (df['High'].shift(1) - 4 * df['Low'].shift(1))
    
    return df

# ---------------- Backtest Strategy ----------------
def backtest_strategy(ticker="^NSEI", period="3d", interval="5m", output_file="strategy_results.xlsx"):
    """
    Backtests a trading strategy based on EMA9, ADX, and Pivot levels.
    """
    df = intraday(ticker, period, interval, output_file="NSEI_intraday_data.xlsx")
    
    df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()

    adx_indicator = ta.trend.ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    df['ADX'] = adx_indicator.adx()
    adx_threshold = 19.5

    df['is_bullish'] = (df['Close'] > df['Open']) & (df['Close'] > df['EMA9']) & (df['Low'] > df['EMA9']) & (df['ADX'] > adx_threshold)
    df['is_bearish'] = (df['Close'] < df['Open']) & (df['Close'] < df['EMA9']) & (df['High'] < df['EMA9']) & (df['ADX'] > adx_threshold)

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

    df['Body'] = abs(df['Close'] - df['Open'])
    df['Lower_Wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['Upper_Wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['is_hammer'] = ((df['Lower_Wick'] >= 7 ) & (df['Upper_Wick'] <= 5) & (df['is_bearish'] == True))
    
    df = generate_signals(df)
    df = calutlate_profit_and_stoploss(df)

    # --- New Logic for evaluating trades and calculating profit ---
    df['Trade_Outcome'] = pd.NA
    df['Profit_Loss'] = pd.NA

    signal_indices = df[df['Signal'] != 0].index.tolist()

    for i in range(len(df)):
        if df.iloc[i]['Signal'] == 1:  # Bullish signal
            entry_price = df.iloc[i]['Close']
            take_profit = df.iloc[i]['Take_Profit']
            stop_loss = df.iloc[i]['Stop_Loss']

            # Find the first subsequent candle to hit TP or SL
            future_data = df.iloc[i+1:]
            
            # Check for a win or loss
            win_condition = future_data['High'] >= take_profit
            loss_condition = future_data['Low'] <= stop_loss
            
            if win_condition.any() and loss_condition.any():
                first_win_idx = win_condition.idxmax()
                first_loss_idx = loss_condition.idxmax()
                if first_win_idx < first_loss_idx:
                    df.iloc[i, df.columns.get_loc('Trade_Outcome')] = 'Win'
                    df.iloc[i, df.columns.get_loc('Profit_Loss')] = take_profit - entry_price
                else:
                    df.iloc[i, df.columns.get_loc('Trade_Outcome')] = 'Loss'
                    df.iloc[i, df.columns.get_loc('Profit_Loss')] = stop_loss - entry_price
            elif win_condition.any():
                df.iloc[i, df.columns.get_loc('Trade_Outcome')] = 'Win'
                df.iloc[i, df.columns.get_loc('Profit_Loss')] = take_profit - entry_price
            elif loss_condition.any():
                df.iloc[i, df.columns.get_loc('Trade_Outcome')] = 'Loss'
                df.iloc[i, df.columns.get_loc('Profit_Loss')] = stop_loss - entry_price

        elif df.iloc[i]['Signal'] == -1: # Bearish signal
            entry_price = df.iloc[i]['Close']
            take_profit = df.iloc[i]['Take_Profit']
            stop_loss = df.iloc[i]['Stop_Loss']

            # Find the first subsequent candle to hit TP or SL
            future_data = df.iloc[i+1:]
            
            # Check for a win or loss
            win_condition = future_data['Low'] <= take_profit
            loss_condition = future_data['High'] >= stop_loss

            if win_condition.any() and loss_condition.any():
                first_win_idx = win_condition.idxmax()
                first_loss_idx = loss_condition.idxmax()
                if first_win_idx < first_loss_idx:
                    df.iloc[i, df.columns.get_loc('Trade_Outcome')] = 'Win'
                    df.iloc[i, df.columns.get_loc('Profit_Loss')] = entry_price - take_profit
                else:
                    df.iloc[i, df.columns.get_loc('Trade_Outcome')] = 'Loss'
                    df.iloc[i, df.columns.get_loc('Profit_Loss')] = entry_price - stop_loss
            elif win_condition.any():
                df.iloc[i, df.columns.get_loc('Trade_Outcome')] = 'Win'
                df.iloc[i, df.columns.get_loc('Profit_Loss')] = entry_price - take_profit
            elif loss_condition.any():
                df.iloc[i, df.columns.get_loc('Trade_Outcome')] = 'Loss'
                df.iloc[i, df.columns.get_loc('Profit_Loss')] = entry_price - stop_loss
    
    # --- End of new logic ---
    
    df['Return'] = df['Close'].pct_change()
    df['Strategy_Return'] = df['Return'] * df['Signal'].shift(1)
    df['Cumulative_Market'] = (1 + df['Return']).cumprod()
    df['Cumulative_Strategy'] = (1 + df['Strategy_Return']).cumprod()

    total_signals = df[df['Signal'] != 0].shape[0]
    total_trades_completed = df['Trade_Outcome'].count()
    winning_trades = df[df['Trade_Outcome'] == 'Win'].shape[0]
    losing_trades = df[df['Trade_Outcome'] == 'Loss'].shape[0]
    
    win_accuracy = (winning_trades / total_trades_completed * 100) if total_trades_completed > 0 else 0
    total_profit = df['Profit_Loss'].sum()
    
    print(f"\n{'='*30}")
    print(f"         BACKTEST RESULTS")
    print(f"{'='*30}")
    print(f"Total Signals Generated: {total_signals}")
    print(f"Total Trades Completed: {total_trades_completed}")
    print(f"Winning Trades: {winning_trades}")
    print(f"Losing Trades: {losing_trades}")
    print(f"Win Accuracy: {win_accuracy:.2f}%")
    print(f"Total Profit/Loss: {total_profit:.2f}")
    print(f"{'='*30}\n")
    
    df.to_excel(output_file, index=True)
    print(f"Backtest saved to: {output_file}")

    return df

def get_prev_data(ticker="^NSEI", period="1mo", interval="1d", output_file="NSEI_prev_data.xlsx"):
    """
    Downloads and processes previous day's data for pivot point calculation.
    """
    df_pivot = yf.download(ticker, period=period, interval=interval, progress=False)
    df_pivot = df_pivot[['Open', 'High', 'Low', 'Close']]
    df_pivot.columns = ['Open', 'High', 'Low', 'Close']
    df_pivot.index = pd.to_datetime(df_pivot.index.tz_localize('utc').tz_convert('Asia/Kolkata'))
    df_pivot.index = pd.to_datetime(df_pivot.index).tz_localize(None)

    df_pivot['Prev_High'] = df_pivot['High'].shift(1)
    df_pivot['Prev_Low'] = df_pivot['Low'].shift(1)
    df_pivot['Prev_Close'] = df_pivot['Close'].shift(1)

    df_pivot['Pivot'] = (df_pivot['Prev_High'] + df_pivot['Prev_Low'] + df_pivot['Prev_Close']) / 3
    df_pivot['Range'] = df_pivot['Prev_High'] - df_pivot['Prev_Low']
    df_pivot = calculate_pivots(df_pivot)
    df_pivot.to_excel(output_file, index=True)
    print(f"Backtest saved to: {output_file}") 
    return df_pivot

def intraday(ticker="^NSEI", period="1mo", interval="1d", output_file="data.xlsx"):
    """
    Downloads intraday data and merges it with previous day's pivot levels.
    """
    df_data = yf.download(ticker, period=period, interval=interval, progress=False)
    df_data = df_data[['Open', 'High', 'Low', 'Close']]
    df_data.columns = ['Open', 'High', 'Low', 'Close']
    
    df_data.index = pd.to_datetime(df_data.index.tz_convert('Asia/Kolkata'))
    df_data.index = df_data.index.tz_localize(None)
    
    df_data['Timestamp'] = df_data.index
    df_data.index = df_data.index.date
    
    df_prev = get_prev_data(ticker, period="1mo", interval="1d", output_file="prev_data.xlsx")
    prev_by_date = df_prev.groupby(df_prev.index.date).first()
    
    for col in ['Pivot', 'Prev_High', 'Prev_Low', 'Prev_Close', 'Range', 'S1', 'S2', 'S3', 'S4', 'S5', 'R1', 'R2', 'R3', 'R4', 'R5']:
        df_data[col] = df_data.index.map(prev_by_date[col])

    matching_count = df_data['Pivot'].notna().sum()
    print(f"Previous data added for {matching_count} records based on date matching.")
    
    df_data.to_excel(output_file, index=True)
    print(f"Intraday data saved to: {output_file}")
    
    return df_data

if __name__ == "__main__":
    print("\n" + "="*60)
    print("           PIVOT POINT TRADING STRATEGY")
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
        period=period, 
        interval=interval, 
        output_file="NSEI_quick_test.xlsx")
