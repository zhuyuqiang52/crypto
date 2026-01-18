import pandas as pd
import numpy as np

class Factor:
    """Base class for all factors."""
    def __init__(self, name, description):
        self.name = name
        self.description = description

    def compute(self, df):
        """
        Compute the factor and return it as a pandas Series.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError

class FactorPool:
    """A collection of factors that can be computed on a given dataset."""
    def __init__(self):
        self.factors = {}
        self.btc_data = None
        self.eth_data = None

    def add_factor(self, factor):
        """Add a factor to the pool."""
        self.factors[factor.name] = factor

    def load_data(self, btc_path, eth_path):
        """Load BTC and ETH data from CSV files."""
        self.btc_data = pd.read_csv(btc_path)
        self.eth_data = pd.read_csv(eth_path)
        
        # Preprocessing
        for df in [self.btc_data, self.eth_data]:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
            df['datetime_et'] = df['datetime'].dt.tz_convert('America/New_York')
        
        self.btc_data = self.btc_data[self.btc_data['datetime_et'] >= '2024-01-01'].copy()
        self.eth_data = self.eth_data[self.eth_data['datetime_et'] >= '2024-01-01'].copy()

    def get_hourly_data(self):
        """Get hourly resampled data for BTC and ETH."""
        if self.btc_data is None or self.eth_data is None:
            raise ValueError("Data not loaded. Please call load_data() first.")

        btc_hourly_close = self.btc_data.set_index('datetime_et')['close'].resample('H').last().rename('btc_hour_close')
        eth_hourly_close = self.eth_data.set_index('datetime_et')['close'].resample('H').last().rename('eth_hour_close')
        
        btc_hourly_volume = self.btc_data.set_index('datetime_et')['volume'].resample('H').sum().rename('btc_hour_volume')
        
        hourly_data = pd.concat([btc_hourly_close, eth_hourly_close, btc_hourly_volume], axis=1).dropna()
        hourly_data.rename(columns={'btc_hour_close': 'Close', 'btc_hour_volume': 'Volume'}, inplace=True)
        return hourly_data

    def compute_all(self):
        """Compute all factors in the pool."""
        hourly_data = self.get_hourly_data()
        results = {}
        results['hour_close'] = hourly_data['Close']
        results['hour_volume'] = hourly_data['Volume']
        for name, factor in self.factors.items():
            results[name] = factor.compute(hourly_data)
        return pd.concat(results, axis=1)
    
    def test_all(self,results_df):
        """Compute all factors performance in the pool."""
        fut_1h_ret_ser = results_df['hour_close'].shift(-1) / results_df['hour_close'] - 1
        fut_12h_ret_ser = results_df['hour_close'].shift(-12) / results_df['hour_close'] - 1
        fut_24h_ret_ser = results_df['hour_close'].shift(-24) / results_df['hour_close'] - 1
        results_df = pd.concat([results_df, fut_1h_ret_ser.rename('fut_1h_ret'), fut_12h_ret_ser.rename('fut_12h_ret'), fut_24h_ret_ser.rename('fut_24h_ret')], axis=1)
        results_df.ffill(axis=0, inplace=True)
        return results.corr().iloc[:,-3:]

    
       
            
if __name__ == '__main__':
    # Example usage
    pool = FactorPool()
    pool.load_data('data/btcusd_1-min_data.csv', 'data/ethusd_1min_ohlc.csv')
    
    # Add factors to the pool
    pool.add_factor(WeeklyVolumeZScore())
    pool.add_factor(MonthlyVolumeZScore())
    pool.add_factor(RollingMaxVolumePercentage())
    pool.add_factor(PriceVolumeCorrelation(pool.btc_data))
    pool.add_factor(PriceVolumeConfirmedReturn(pool.btc_data))
    pool.add_factor(MinZScore())
    pool.add_factor(RSI())
    pool.add_factor(BTCETH24hCorrelation())
    pool.add_factor(BTCETHIntraHourCorrelation(pool.btc_data, pool.eth_data))

    results = pool.compute_all()
    print(results.head())

# Factor Implementations
class WeeklyVolumeZScore(Factor):
    """
    Calculates the z-score of the current hour's volume against the rolling average
    and standard deviation of the previous 7 days of hourly volumes. This factor
    helps identify significant deviations in trading volume from the recent weekly norm.
    """
    def __init__(self):
        super().__init__('vol_zscore_weekly', 'Z-score of hourly volume compared to rolling 7-day average')

    def compute(self, df):
        df_hourly = df[['Volume']].copy()
        z_scores = []
        for i in range(len(df_hourly)):
            window = df_hourly.iloc[max(0, i - 7*24):i]
            if len(window) < 2:
                z_scores.append(np.nan)
                continue
            mean_vol = window['Volume'].mean()
            std_vol = window['Volume'].std()
            if std_vol == 0:
                z_scores.append(0.0)
            else:
                z = (df_hourly.iloc[i]['Volume'] - mean_vol) / std_vol
                z_scores.append(z)
        return pd.Series(z_scores, index=df_hourly.index, name=self.name)

class MonthlyVolumeZScore(Factor):
    """
    Computes the z-score of the current hour's volume against the mean and standard
    deviation of volumes from the same hour and same day of the week over the past
    4 weeks. This captures weekly seasonal patterns in volume.
    """
    def __init__(self):
        super().__init__('vol_zscore_monthly', 'Z-score of the same hour of the same day over past 4 weeks')

    def compute(self, df):
        df_hourly = df[['Volume']].copy()
        df_hourly['hour_of_day'] = df_hourly.index.hour
        df_hourly['day_of_week'] = df_hourly.index.dayofweek
        z_scores = []
        for idx, row in df_hourly.iterrows():
            hour = row.name.hour
            day = row.name.dayofweek
            past_data = df_hourly[(df_hourly.index < idx) & (df_hourly.index.hour == hour) & (df_hourly.index.dayofweek == day)].tail(4)
            if len(past_data) < 2:
                z_scores.append(np.nan)
                continue
            mean_vol = past_data['Volume'].mean()
            std_vol = past_data['Volume'].std()
            if std_vol == 0:
                z_scores.append(0.0)
            else:
                z = (row['Volume'] - mean_vol) / std_vol
                z_scores.append(z)
        return pd.Series(z_scores, index=df_hourly.index, name=self.name)

class RollingMaxVolumePercentage(Factor):
    """
    Calculates the ratio of the current hour's volume to the maximum hourly volume
    observed over the previous 7-day rolling window. This factor indicates the
    current volume's significance relative to recent peak volumes.
    """
    def __init__(self):
        super().__init__('pct_of_rolling_max', 'Percentage of current volume to 7-day rolling max')

    def compute(self, df):
        rolling_max_7d = df['Volume'].shift(9).rolling(window=7*24).max()
        return df['Volume'] / rolling_max_7d

class PriceVolumeCorrelation(Factor):
    """
    Measures the correlation between minute-by-minute price returns and volume
    changes within each hour. A high positive correlation suggests that price
    movements are strongly supported by trading volume.
    """
    def __init__(self, minute_data):
        super().__init__('price_vol_corr', 'Hourly correlation between price return and volume change')
        self.minute_data = minute_data.set_index('datetime_et')

    def compute(self, df):
        def hourly_price_vol_corr(df_hour):
            close_ret = df_hour['close'].ffill().pct_change()
            vol_ret = df_hour['volume'].ffill().pct_change()
            if close_ret.dropna().shape[0] < 2 or vol_ret.dropna().shape[0] < 2:
                return np.nan
            return close_ret.corr(vol_ret)
        return self.minute_data[['close','volume']].resample('H').apply(hourly_price_vol_corr)

class PriceVolumeConfirmedReturn(Factor):
    """
    This factor captures the hourly price return, but only when the intra-hour
    price-volume correlation is positive. It is designed to isolate price movements
    that are confirmed by supportive volume, filtering out noise.
    """
    def __init__(self, minute_data):
        super().__init__('hour_return_confirmed_by_pv_corr', 'Hourly return confirmed by price-volume correlation')
        self.minute_data = minute_data.set_index('datetime_et')

    def compute(self, df):
        price_vol_corr = PriceVolumeCorrelation(self.minute_data).compute(df)
        hour_return = df['Close'].pct_change()
        return np.where((price_vol_corr > 0.0), hour_return, 0)

class MinZScore(Factor):
    """
    This factor returns the minimum value between the weekly and monthly volume z-scores.
    It provides a single, more conservative measure of volume deviation by considering
    both short-term and seasonal patterns.
    """
    def __init__(self):
        super().__init__('min_zscore', 'Minimum of weekly and monthly volume z-scores')

    def compute(self, df):
        weekly_z = WeeklyVolumeZScore().compute(df)
        monthly_z = MonthlyVolumeZScore().compute(df)
        return pd.concat([weekly_z, monthly_z], axis=1).min(axis=1)

class RSI(Factor):
    """
    Calculates the 14-hour Relative Strength Index (RSI). The RSI is a momentum
    oscillator that measures the speed and change of price movements, typically
    used to identify overbought or oversold conditions.
    """
    def __init__(self):
        super().__init__('rsi', '14-hour Relative Strength Index')

    def compute(self, df):
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14*24).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14*24).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

class BTCETH24hCorrelation(Factor):
    """
    Computes the 24-hour rolling correlation of hourly returns between BTC and ETH.
    This factor measures the degree to which the two assets' prices move together
    over the course of a day.
    """
    def __init__(self):
        super().__init__('btc_eth_24h_corr', '24-hour rolling correlation of BTC and ETH hourly returns')

    def compute(self, df):
        btc_hr_ret = df['Close'].pct_change()
        eth_hr_ret = df['eth_hour_close'].pct_change()
        return btc_hr_ret.rolling(window=24).corr(eth_hr_ret)

class BTCETHIntraHourCorrelation(Factor):
    """
    Calculates the correlation of minute-by-minute returns between BTC and ETH
    within each hour. This provides a high-frequency measure of their co-movement,
    capturing short-term market dynamics.
    """
    def __init__(self, btc_minute_data, eth_minute_data):
        super().__init__('btc_eth_intra_hour_corr', 'Intra-hour correlation of BTC and ETH minute returns')
        self.btc_minute_data = btc_minute_data.set_index('datetime_et')
        self.eth_minute_data = eth_minute_data.set_index('datetime_et')

    def compute(self, df):
        btc_min_ret = self.btc_minute_data['close'].pct_change().rename('btc_ret')
        eth_min_ret = self.eth_minute_data['close'].pct_change().rename('eth_ret')
        combined_min = pd.concat([btc_min_ret, eth_min_ret], axis=1).dropna()
        
        def calc_corr(df_min):
            return df_min['btc_ret'].corr(df_min['eth_ret']) if len(df_min) > 1 else np.nan
            
        return combined_min.resample('H').apply(calc_corr)

class CorrelationZScore(Factor):
    """
    Calculates the z-score of a given correlation factor over a specified rolling
    window. This is a meta-factor that can be applied to any correlation-based
    factor to normalize its values and identify statistically significant deviations
    from its recent average.
    """
    def __init__(self, corr_factor, window=24):
        self.corr_factor = corr_factor
        self.window = window
        name = f"{corr_factor.name}_zscore"
        description = f"Z-score of {corr_factor.name} over a {window}-hour window"
        super().__init__(name, description)

    def compute(self, df):
        corr_series = self.corr_factor.compute(df)
        return (corr_series - corr_series.rolling(self.window).mean()) / corr_series.rolling(self.window).std()

