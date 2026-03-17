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
            df['datetime'] = pd.to_datetime(df['Timestamp'], unit='s', utc=True)
            df['datetime_et'] = df['datetime'].dt.tz_convert('America/New_York')
        
        self.btc_data = self.btc_data[self.btc_data['datetime_et'] >= '2021-01-01'].copy()
        self.eth_data = self.eth_data[self.eth_data['datetime_et'] >= '2021-01-01'].copy()

    def get_hourly_data(self):
        """Get hourly resampled data for BTC and ETH."""
        if self.btc_data is None or self.eth_data is None:
            raise ValueError("Data not loaded. Please call load_data() first.")

        btc_hourly_open = self.btc_data.set_index('datetime_et')['open'].resample('H').first().rename('btc_hour_open')
        btc_hourly_high = self.btc_data.set_index('datetime_et')['high'].resample('H').max().rename('btc_hour_high')
        btc_hourly_low = self.btc_data.set_index('datetime_et')['low'].resample('H').min().rename('btc_hour_low')
        btc_hourly_close = self.btc_data.set_index('datetime_et')['close'].resample('H').last().rename('btc_hour_close')
        btc_hourly_volume = self.btc_data.set_index('datetime_et')['volume'].resample('H').sum().rename('btc_hour_volume')
        
        eth_hourly_close = self.eth_data.set_index('datetime_et')['close'].resample('H').last().rename('eth_hour_close')
        
        hourly_data = pd.concat([
            btc_hourly_open, btc_hourly_high, btc_hourly_low, btc_hourly_close, 
            btc_hourly_volume, eth_hourly_close
        ], axis=1).dropna()
        hourly_data.rename(columns={
            'btc_hour_open': 'Open', 'btc_hour_high': 'High', 'btc_hour_low': 'Low', 
            'btc_hour_close': 'Close', 'btc_hour_volume': 'Volume'}, inplace=True)
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
        return results_df.corr().iloc[:,-3:]

    
       
            
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
    pool.add_factor(IntraHourVolatility(pool.btc_data))
    pool.add_factor(IntraHourVWAP(pool.btc_data))
    pool.add_factor(IntraHourHighLowRange(pool.btc_data))
    pool.add_factor(IntraHourVolumeSpike(pool.btc_data))
    pool.add_factor(IntraHourPriceVolumeTrend(pool.btc_data))
    pool.add_factor(HourlyReturn())
    pool.add_factor(MACD())
    pool.add_factor(BollingerBandsWidth())
    pool.add_factor(Alpha1())
    pool.add_factor(Alpha101())
    pool.add_factor(ADX())
    pool.add_factor(FisherTransform())
    pool.add_factor(PercentageATR())
    pool.add_factor(MoneyFlowIndex())
    pool.add_factor(HilbertTransformPhase())

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

class IntraHourVolatility(Factor):
    """
    Standard deviation of minute returns within the hour.
    """
    def __init__(self, minute_data):
        super().__init__('intra_hour_volatility', 'Standard deviation of minute returns within the hour')
        self.minute_data = minute_data.set_index('datetime_et')

    def compute(self, df):
        min_ret = self.minute_data['close'].pct_change()
        return min_ret.resample('H').std().rename(self.name)

class IntraHourVWAP(Factor):
    """
    Volume Weighted Average Price (VWAP) within the hour.
    """
    def __init__(self, minute_data):
        super().__init__('intra_hour_vwap', 'Volume weighted average price within the hour')
        self.minute_data = minute_data.set_index('datetime_et')

    def compute(self, df):
        pv = self.minute_data['close'] * self.minute_data['volume']
        v = self.minute_data['volume']
        vwap = pv.resample('H').sum() / (v.resample('H').sum() + 1e-8)
        return vwap.rename(self.name)

class IntraHourHighLowRange(Factor):
    """
    High-Low range within the hour normalized by the opening price.
    """
    def __init__(self, minute_data):
        super().__init__('intra_hour_hl_range', 'High-Low range within the hour relative to open')
        self.minute_data = minute_data.set_index('datetime_et')

    def compute(self, df):
        h = self.minute_data['high'].resample('H').max()
        l = self.minute_data['low'].resample('H').min()
        o = self.minute_data['open'].resample('H').first()
        return ((h - l) / o).rename(self.name)

class IntraHourVolumeSpike(Factor):
    """
    Ratio of the maximum minute volume to the average minute volume within the hour.
    """
    def __init__(self, minute_data):
        super().__init__('intra_hour_vol_spike', 'Ratio of max minute volume to mean minute volume in the hour')
        self.minute_data = minute_data.set_index('datetime_et')

    def compute(self, df):
        v = self.minute_data['volume']
        return (v.resample('H').max() / (v.resample('H').mean() + 1e-8)).rename(self.name)

class IntraHourPriceVolumeTrend(Factor):
    """
    Inspired by Alpha 101 (e.g., Alpha 12).
    Intra-hour sum of sign(delta(volume)) * (-delta(close)).
    """
    def __init__(self, minute_data):
        super().__init__('intra_hour_pv_trend', 'Intra-hour sum of sign(delta(volume)) * (-delta(close))')
        self.minute_data = minute_data.set_index('datetime_et')

    def compute(self, df):
        delta_v = self.minute_data['volume'].diff()
        delta_c = self.minute_data['close'].diff()
        pv_trend = (np.sign(delta_v) * -delta_c).resample('H').sum()
        return pv_trend.rename(self.name)

class HourlyReturn(Factor):
    """
    Raw hourly close-to-close return.
    """
    def __init__(self):
        super().__init__('hourly_return', 'Hourly close-to-close return')

    def compute(self, df):
        return df['Close'].pct_change().rename(self.name)

class MACD(Factor):
    """
    Moving Average Convergence Divergence (MACD) on the hourly close price.
    """
    def __init__(self, fast=12, slow=26, signal=9):
        super().__init__(f'macd_{fast}_{slow}_{signal}', 'MACD indicator')
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def compute(self, df):
        ema_fast = df['Close'].ewm(span=self.fast, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=self.slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        return macd.rename(self.name)

class BollingerBandsWidth(Factor):
    """
    Bollinger Bands width on the hourly close price.
    """
    def __init__(self, window=20, num_std=2):
        super().__init__(f'bb_width_{window}', 'Bollinger Bands Width')
        self.window = window
        self.num_std = num_std

    def compute(self, df):
        rolling_mean = df['Close'].rolling(window=self.window).mean()
        rolling_std = df['Close'].rolling(window=self.window).std()
        upper_band = rolling_mean + (rolling_std * self.num_std)
        lower_band = rolling_mean - (rolling_std * self.num_std)
        return ((upper_band - lower_band) / rolling_mean).rename(self.name)

class Alpha1(Factor):
    """
    WorldQuant Alpha 1 (Adapted for single asset).
    Formula: rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5
    """
    def __init__(self):
        super().__init__('alpha_1', 'WorldQuant Alpha 1')

    def compute(self, df):
        ret = df['Close'].pct_change()
        stddev = ret.rolling(20).std()
        cond = ret < 0
        
        val = np.where(cond, stddev, df['Close'])
        signed_power = np.sign(val) * (np.abs(val) ** 2)
        
        # Ts_ArgMax over 5 periods (returns 1 to 5)
        ts_argmax = pd.Series(signed_power, index=df.index).rolling(5).apply(lambda x: np.argmax(x) + 1, raw=True)
        
        # Scaling to [-0.5, 0.5] as a substitute for cross-sectional rank
        return (ts_argmax / 5.0 - 0.5).rename(self.name)

class Alpha101(Factor):
    """
    WorldQuant Alpha 101.
    Formula: ((close - open) / ((high - low) + .001))
    """
    def __init__(self):
        super().__init__('alpha_101', 'WorldQuant Alpha 101')

    def compute(self, df):
        return ((df['Close'] - df['Open']) / ((df['High'] - df['Low']) + 0.001)).rename(self.name)

class ADX(Factor):
    """
    Average Directional Index (ADX).
    Measures trend strength.
    """
    def __init__(self, window=14):
        super().__init__(f'adx_{window}', 'Average Directional Index')
        self.window = window

    def compute(self, df):
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        pos_dm = pd.Series(pos_dm, index=df.index)
        neg_dm = pd.Series(neg_dm, index=df.index)
        
        atr = tr.ewm(alpha=1/self.window, adjust=False).mean()
        pos_di = 100 * pos_dm.ewm(alpha=1/self.window, adjust=False).mean() / atr
        neg_di = 100 * neg_dm.ewm(alpha=1/self.window, adjust=False).mean() / atr
        
        dx = 100 * (pos_di - neg_di).abs() / (pos_di + neg_di + 1e-8)
        adx = dx.ewm(alpha=1/self.window, adjust=False).mean()
        
        return adx.rename(self.name)

class FisherTransform(Factor):
    """
    Fisher Transform on price.
    Transforms price into a Gaussian distribution.
    """
    def __init__(self, window=9):
        super().__init__(f'fisher_transform_{window}', 'Fisher Transform')
        self.window = window

    def compute(self, df):
        hl2 = (df['High'] + df['Low']) / 2
        roll_min = hl2.rolling(window=self.window).min()
        roll_max = hl2.rolling(window=self.window).max()
        
        # Normalize to [-1, 1]
        x = 2 * ((hl2 - roll_min) / (roll_max - roll_min + 1e-8)) - 1
        x = x.clip(-0.999, 0.999)
        
        fisher = 0.5 * np.log((1 + x) / (1 - x))
        # Smoothed with previous value
        fisher = fisher.ewm(alpha=0.5, adjust=False).mean()
        
        return fisher.rename(self.name)

class PercentageATR(Factor):
    """
    Normalized ATR (ATR / Close) to make it price-agnostic.
    """
    def __init__(self, window=14):
        super().__init__(f'natr_{window}', 'Percentage Average True Range')
        self.window = window

    def compute(self, df):
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.ewm(alpha=1/self.window, adjust=False).mean()
        natr = (atr / close) * 100
        
        return natr.rename(self.name)

class MoneyFlowIndex(Factor):
    """
    Money Flow Index (MFI).
    Volume-weighted RSI.
    """
    def __init__(self, window=14):
        super().__init__(f'mfi_{window}', 'Money Flow Index')
        self.window = window

    def compute(self, df):
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        rmf = tp * df['Volume']
        
        diff = tp.diff()
        pos_mf = np.where(diff > 0, rmf, 0)
        neg_mf = np.where(diff < 0, rmf, 0)
        
        pos_mf = pd.Series(pos_mf, index=df.index).rolling(window=self.window).sum()
        neg_mf = pd.Series(neg_mf, index=df.index).rolling(window=self.window).sum()
        
        mfr = pos_mf / (neg_mf + 1e-8)
        mfi = 100 - (100 / (1 + mfr))
        
        return mfi.rename(self.name)

class HilbertTransformPhase(Factor):
    """
    Hilbert Transform dominant cycle phase approximation.
    """
    def __init__(self, window=24):
        super().__init__(f'hilbert_phase_{window}', 'Hilbert Transform Phase Approx')
        self.window = window

    def compute(self, df):
        # A simple approximation: subtract moving average to center,
        # then calculate phase using a quarter-cycle delayed signal
        delay = max(1, self.window // 4)
        centered = df['Close'] - df['Close'].rolling(window=self.window).mean()
        hilbert = centered.shift(delay)
        
        phase = np.arctan2(hilbert, centered)
        return phase.rename(self.name)
