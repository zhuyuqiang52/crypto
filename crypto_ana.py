#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
btc_df = pd.read_csv('data/btcusd_1-min_data.csv')
eth_df = pd.read_csv('data/ethusd_1min_ohlc.csv')

# Standardize BTC column names to lowercase to match ETH and API responses
btc_df.rename(columns={
    'Timestamp': 'timestamp',
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Volume': 'volume'
}, inplace=True)

btc_df['datetime'] = pd.to_datetime(btc_df['timestamp'], unit='s', utc=True)
btc_df['datetime_et'] = btc_df['datetime'].dt.tz_convert('America/New_York')
eth_df['datetime'] = pd.to_datetime(eth_df['timestamp'], unit='s', utc=True)
eth_df['datetime_et'] = eth_df['datetime'].dt.tz_convert('America/New_York')


# %%
import requests
from datetime import datetime, timedelta
import pytz
import time

def update_crypto_data(df, product_id, file_path):
    """
    Updates a DataFrame with the latest minute-level crypto data from the Coinbase API.

    Args:
        df (pd.DataFrame): The DataFrame to update. Must have 'datetime_et' and 'timestamp' columns.
        product_id (str): The product ID for the API call (e.g., "ETH-USD", "BTC-USD").
        file_path (str): The path to the CSV file to save the updated data.

    Returns:
        pd.DataFrame: The updated DataFrame.
    """
    est = pytz.timezone('America/New_York')
    last_timestamp_et = df['datetime_et'].max()
    now_et = datetime.now(est)

    if last_timestamp_et >= now_et - timedelta(minutes=2):
        print(f"{product_id} data is already up-to-date.")
        print(f"Last timestamp: {last_timestamp_et}")
        return df

    print(f"Last {product_id} data point is from: {last_timestamp_et}")
    print(f"Fetching data up to: {now_et}...")

    url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
    granularity = 60  # 60 seconds = 1 minute candles
    all_new_candles = []
    start_time_loop = last_timestamp_et + timedelta(minutes=1)
    delta = timedelta(minutes=300)

    while start_time_loop < now_et:
        end_time_loop = start_time_loop + delta
        params = {
            "start": start_time_loop.isoformat(),
            "end": end_time_loop.isoformat(),
            "granularity": granularity
        }
        print(f"Requesting chunk for {product_id}: {start_time_loop.strftime('%Y-%m-%d %H:%M')} -> {end_time_loop.strftime('%Y-%m-%d %H:%M')}")
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            if data:
                data.reverse()
                all_new_candles.extend(data)
                print(f"  ...retrieved {len(data)} candles.")
            else:
                print("  ...no data in this interval.")
        except requests.exceptions.RequestException as e:
            print(f"An error occurred during API request for {product_id}: {e}")
            break
        start_time_loop = end_time_loop
        time.sleep(0.2)

    if all_new_candles:
        new_df = pd.DataFrame(
            all_new_candles,
            columns=['Timestamp', 'Low', 'High', 'Open', 'Close', 'Volume']
        )
        print(f"\nFetched a total of {len(new_df)} new candles for {product_id}.")
        old_rows = len(df)

        new_df['datetime'] = pd.to_datetime(new_df['timestamp'], unit='s', utc=True)
        new_df['datetime_et'] = new_df['datetime'].dt.tz_convert('America/New_York')

        updated_df = pd.concat([df, new_df], ignore_index=True, sort=False)
        
        updated_df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
        updated_df.sort_values('timestamp', inplace=True)
        updated_df.reset_index(drop=True, inplace=True)
        
        print(f"{product_id} df updated from {old_rows} to {len(updated_df)} rows.")
        print("\nLast 5 rows of the updated DataFrame:")
        print(updated_df.tail())
        
        updated_df.to_csv(file_path, index=False)
        return updated_df
    
    elif last_timestamp_et < now_et - timedelta(minutes=2):
        print(f"\nNo new data was fetched for {product_id}, though the dataset appears to be out of date.")
    
    return df

# --- Update ETH and BTC data ---
eth_df = update_crypto_data(eth_df, "ETH-USD", 'data/ethusd_1min_ohlc.csv')
btc_df = update_crypto_data(btc_df, "BTC-USD", 'data/btcusd_1-min_data.csv')
# %%
start_dt = pd.to_datetime("2025-07-05",utc=True)
start_dt = start_dt.tz_convert('America/New_York')
end_dt = pd.to_datetime("2026-03-13",utc=True)
end_dt = end_dt.tz_convert('America/New_York')
tmp_btc_df = btc_df[(btc_df['datetime_et'] >= start_dt) & (btc_df['datetime_et'] <= end_dt)]
# plot btc volume and price change at two axis
fig, ax1 = plt.subplots(figsize=(10,5))
ax2 = ax1.twinx()
ax1.plot(tmp_btc_df['datetime_et'], tmp_btc_df['close'], 'g-',label = "Close")
ax2.plot(tmp_btc_df['datetime_et'], tmp_btc_df['volume'], color='b', alpha=0.5, label = 'Volume')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.show()
# %%
# transform timestamp and plot the price
start_dt = pd.to_datetime("2025-07-05",utc=True)
start_dt = start_dt.tz_convert('America/New_York')
end_dt = pd.to_datetime("2026-03-13",utc=True)
end_dt = end_dt.tz_convert('America/New_York')
tmp_eth_df = eth_df[(eth_df['datetime_et'] >= start_dt) & (eth_df['datetime_et'] <= end_dt)]
# plot btc volume and price change at two axis
fig, ax1 = plt.subplots(figsize=(10,5))
ax2 = ax1.twinx()
ax1.plot(tmp_eth_df['datetime_et'], tmp_eth_df['close'], 'g-',label = 'close')
ax2.plot(tmp_eth_df['datetime_et'], tmp_eth_df['volume'], color='b', alpha=0.5, label = 'Volume')
# show legend
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
# show plot
plt.show()
#%%
btc_df['volume'].hist(bins=100)
