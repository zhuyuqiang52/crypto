#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
btc_df = pd.read_csv('data/btcusd_1-min_data.csv')
eth_df = pd.read_csv('data/ethusd_1min_ohlc.csv')
btc_df['datetime'] = pd.to_datetime(btc_df['Timestamp'], unit='s', utc=True)
btc_df['datetime_et'] = btc_df['datetime'].dt.tz_convert('America/New_York')
eth_df['datetime'] = pd.to_datetime(eth_df['timestamp'], unit='s', utc=True)
eth_df['datetime_et'] = eth_df['datetime'].dt.tz_convert('America/New_York')


# %%
# import requests
# import pandas as pd
# from datetime import datetime, timedelta
# import pytz
# import time

# # --- Complete eth_df with the latest minute data from Coinbase API ---

# # Ensure eth_df from the previous cell is available
# try:
#     eth_df
# except NameError:
#     raise NameError("eth_df not found. Please run the first cell to load the initial data.")

# # --- 1. Define Time Range for Data Fetching ---
# est = pytz.timezone('America/New_York')
# # Get the last timestamp from our existing data
# last_timestamp_et = eth_df['datetime_et'].max()
# now_et = datetime.now(est)

# # Check if data is already up-to-date
# if last_timestamp_et >= now_et - timedelta(minutes=2):
#     print("ETH data is already up-to-date.")
#     print(f"Last timestamp: {last_timestamp_et}")
# else:
#     print(f"Last data point is from: {last_timestamp_et}")
#     print(f"Fetching data up to: {now_et}...")

#     # --- 2. Setup API Request Parameters ---
#     product_id = "ETH-USD"
#     url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
#     granularity = 60  # 60 seconds = 1 minute candles
    
#     # --- 3. Fetch Data in Paginated Loop ---
#     all_new_candles = []
#     # Start fetching from the minute after our last data point
#     start_time_loop = last_timestamp_et + timedelta(minutes=1)
    
#     # Coinbase API returns a max of 300 candles per request. 300 mins = 5 hours. [10]
#     delta = timedelta(minutes=300)

#     while start_time_loop < now_et:
#         end_time_loop = start_time_loop + delta
        
#         params = {
#             "start": start_time_loop.isoformat(),
#             "end": end_time_loop.isoformat(),
#             "granularity": granularity
#         }

#         print(f"Requesting chunk: {start_time_loop.strftime('%Y-%m-%d %H:%M')} -> {end_time_loop.strftime('%Y-%m-%d %H:%M')}")
        
#         try:
#             response = requests.get(url, params=params)
#             response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
#             data = response.json()
            
#             if data:
#                 # API returns candles in descending order; reverse them to process chronologically
#                 data.reverse()
#                 all_new_candles.extend(data)
#                 print(f"  ...retrieved {len(data)} candles.")
#             else:
#                 print("  ...no data in this interval.")
        
#         except requests.exceptions.RequestException as e:
#             print(f"An error occurred during API request: {e}")
#             break  # Exit loop on failure
        
#         # Move to the next time window for the next iteration
#         start_time_loop = end_time_loop
        
#         # Be a good citizen and respect API rate limits
#         time.sleep(0.2)

#     # --- 4. Process and Append New Data ---
#     if all_new_candles:
#         # Convert list of lists to a DataFrame
#         # API columns: [time, low, high, open, close, volume]
#         new_df = pd.DataFrame(
#             all_new_candles,
#             columns=['timestamp', 'low', 'high', 'open', 'close', 'volume']
#         )
        
#         print(f"\nFetched a total of {len(new_df)} new candles.")
#         old_rows = len(eth_df)

#         # Add and convert datetime columns
#         new_df['datetime'] = pd.to_datetime(new_df['timestamp'], unit='s', utc=True)
#         new_df['datetime_et'] = new_df['datetime'].dt.tz_convert('America/New_York')
        
#         # Reorder new_df columns to match eth_df for clean concatenation
#         new_df = new_df[eth_df.columns]

#         # Append new data to the existing DataFrame
#         eth_df = pd.concat([eth_df, new_df], ignore_index=True)
        
#         # Clean up: remove duplicates, sort, and reset index
#         eth_df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
#         eth_df.sort_values('timestamp', inplace=True)
#         eth_df.reset_index(drop=True, inplace=True)
        
#         print(f"eth_df updated from {old_rows} to {len(eth_df)} rows.")
#         print("\nLast 5 rows of the updated DataFrame:")
#         print(eth_df.tail())
        
#     elif last_timestamp_et < now_et - timedelta(minutes=2):
#         print("\nNo new data was fetched, though the dataset appears to be out of date.")
# eth_df.to_csv('data/ethusd_1min_ohlc.csv', index=False)
# %%
eth_df.tail()
# %%
start_dt = pd.to_datetime("2026-01-10",utc=True)
start_dt = start_dt.tz_convert('America/New_York')
end_dt = pd.to_datetime("2026-01-13",utc=True)
end_dt = end_dt.tz_convert('America/New_York')
tmp_btc_df = btc_df[(btc_df['datetime_et'] >= start_dt) & (btc_df['datetime_et'] <= end_dt)]
# plot btc volume and price change at two axis
fig, ax1 = plt.subplots(figsize=(10,5))
ax2 = ax1.twinx()
ax1.plot(tmp_btc_df['datetime_et'], tmp_btc_df['Close'], 'g-')
ax2.plot(tmp_btc_df['datetime_et'], tmp_btc_df['Volume'], color='b', alpha=0.5)
plt.show()
    
# %%
start_dt = pd.to_datetime("2025-07-05",utc=True)
start_dt = start_dt.tz_convert('America/New_York')
end_dt = pd.to_datetime("2025-07-15",utc=True)
end_dt = end_dt.tz_convert('America/New_York')
tmp_btc_df = btc_df[(btc_df['datetime_et'] >= start_dt) & (btc_df['datetime_et'] <= end_dt)]
# plot btc volume and price change at two axis
fig, ax1 = plt.subplots(figsize=(10,5))
ax2 = ax1.twinx()
ax1.plot(tmp_btc_df['datetime_et'], tmp_btc_df['Close'], 'g-')
ax2.plot(tmp_btc_df['datetime_et'], tmp_btc_df['Volume'], color='b', alpha=0.5)
plt.show()
# %%
start_dt = pd.to_datetime("2025-04-17",utc=True)
start_dt = start_dt.tz_convert('America/New_York')
end_dt = pd.to_datetime("2025-05-17",utc=True)
end_dt = end_dt.tz_convert('America/New_York')
tmp_btc_df = btc_df[(btc_df['datetime_et'] >= start_dt) & (btc_df['datetime_et'] <= end_dt)]
# plot btc volume and price change at two axis
fig, ax1 = plt.subplots(figsize=(10,5))
ax2 = ax1.twinx()
ax1.plot(tmp_btc_df['datetime_et'], tmp_btc_df['Close'], 'g-')
ax2.plot(tmp_btc_df['datetime_et'], tmp_btc_df['Volume'], color='b', alpha=0.5)
plt.show()

#%%
btc_df['Volume'].hist(bins=100)