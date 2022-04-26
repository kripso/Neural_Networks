from utils.enums import Intervals, Months
from plotly import figure_factory
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time


#
# Time and Timestamp Helping Functions
#
def timeStampToTime(unixtime: int) -> str:
    return time.strftime("%H:%M", time.gmtime(unixtime))


def timeToTimeStamp(year: int, month: Months, day: int, hour: int, minute: int, second: int) -> int:
    return int(datetime.timestamp(datetime(year=year, month=month.value, day=day, hour=hour, minute=minute, second=second)))


def timeStampToDateTime(unixtime: int) -> datetime:
    return datetime.fromtimestamp(unixtime)


#
# Load and return stored prices
#
def get_stored_prices(interval=Intervals.fifteenMin, since=timeToTimeStamp(2017, Months.january, 1, 1, 0, 0), pair='XBTEUR') -> pd.DataFrame:
    """Returng stored prices for curency pair and interval

    Args:
        interval (_type_, optional): _description_. Defaults to Intervals.fifteenMin.
        since (_type_, optional): _description_. Defaults to timeToTimeStamp(2017, Months.january, 1, 1, 0, 0).
        pair (str, optional): _description_. Defaults to 'XBTEUR'.

    Returns:
        pd.DataFrame: _description_
    """
    file = './data/{}_{}.csv'.format(pair, interval.value)

    df = pd.read_csv(file)
    df.columns = ['time', 'open', 'high', 'low', 'close', 'volume', 'count']

    for column in df.columns:
        df[column] = pd.to_numeric(df[column])

    dateTimeData = [timeStampToDateTime(data) for data in df.time]
    df["dateTime"] = dateTimeData
    # df = df.set_index(df.dateTime)

    return df.loc[df['time'] > since]


#
# Convert the interval of candle stick prices to n*interval
#
def n_times_interval(data: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Convert the interval of candle stick prices to n*interval

    Args:
        data (pd.DataFrame): _description_
        n (int, optional): _description_. Defaults to 3.

    Returns:
        pd.DataFrame: _description_
    """
    dataArr = []

    index = 0
    for row in range(data.time.count()):
        if data.time[row] % 3600 == 0:
            index = row
            break

    for row in range(index, data.time.count(), n):
        firstRow = row
        lastRow = (row + n - 1) if (row + n - 1) < data.time.count() else (data.time.count() - 1)

        dataArr.append([
            data.time[firstRow],
            data.open[firstRow],
            max(data.high[_row] for _row in range(row, lastRow + 1)),
            -max((-data.low[_row]) for _row in range(row, lastRow + 1)),
            data.close[lastRow],
            # if 'vwap' in data:
            # np.median([data.vwap[_row] for _row in range(row, lastRow + 1)]),
            sum(data.volume[_row] for _row in range(row, lastRow + 1)),
            sum(data['count'][_row] for _row in range(row, lastRow + 1)),
            data.dateTime[firstRow]
        ])

    tmpDf = pd.DataFrame(list(map(np.ravel, dataArr)), columns=data.columns)
    tmpDf.index = tmpDf.dateTime
    return tmpDf


#
# Average of sliced data over lenght
#
def _get_sma(data, length) -> int:
    """slices data to lengt and calculates its average

    Args:
        data (_type_): _description_
        length (_type_): _description_

    Returns:
        int: _description_
    """
    return sum(data[:length]) / length


#
# SMA - Simple Moving Average
#
def get_sma(data: pd.DataFrame, length) -> pd.DataFrame:
    """Getter for Simple Moving Average.

    Args:
        data (pd.DataFrame): _description_
        length (_type_): _description_

    Returns:
        pd.DataFrame: _description_
    """
    noneIndeces = data.size-data.count()
    our_range = range(len(data))[noneIndeces + length - 1:]

    sma = [np.mean(data[i - length + 1: i + 1]) for i in our_range]
    sma = np.array([None for _ in range((length+noneIndeces)-1)] + sma)
    return pd.DataFrame(sma, index=data.index.copy(), columns=['data'])


#
# EMA - Exponential Moving Average
#
def get_ema(data: pd.DataFrame, length, smoothing=2, com=None) -> pd.DataFrame:
    """Getter for Exponential Moving Average

    Args:
        data (pd.DataFrame): _description_
        length (_type_): _description_
        smoothing (int, optional): _description_. Defaults to 2.
        com (_type_, optional): _description_. Defaults to None.

    Returns:
        pd.DataFrame: _description_
    """
    noneIndeces = data.size-data.count()
    ema = [None for _ in range((length+noneIndeces)-1)]
    ema.append(_get_sma(data[noneIndeces:], length))

    alpha = (smoothing / (1 + length))
    if com:
        alpha = 1/(com)

    for price in data[length+noneIndeces:]:
        ema.append((price * alpha) + ema[-1] * (1 - alpha))

    return pd.DataFrame(ema, index=data.index.copy(), columns=['data'])


#
# MACD - Moving Average Convergence Divergence
#
def get_macd(data, fastperiod=12, slowperiod=26, signalperiod=9) -> list[pd.DataFrame, pd.DataFrame]:
    """Getter for Moving Average Convergence Divergence indicator

    Args:
        data (_type_): _description_
        fastperiod (int, optional): _description_. Defaults to 12.
        slowperiod (int, optional): _description_. Defaults to 26.
        signalperiod (int, optional): _description_. Defaults to 9.

    Returns:
        list[pd.DataFrame, pd.DataFrame]: _description_
    """
    fastEma = get_ema(data, fastperiod)
    slowEma = get_ema(data, slowperiod)

    macd = fastEma - slowEma

    macdsignal = get_ema(macd.data, signalperiod)
    return (macd, macdsignal)


#
# HMA -> Hull Moving Average
#
def get_hma(data: pd.DataFrame, length) -> list[int]:
    """Getter for Hull Moving Average from dataframe

    Args:
        data (pd.DataFrame): _description_
        length (_type_): _description_

    Returns:
        list[int]: _description_
    """
    noneIndeces = data.size-data.count()

    sma1 = get_sma(data[noneIndeces:], length)
    sma2 = get_sma(data[noneIndeces:], int(length/2))
    sma_diff = 2 * sma2 - sma1

    return get_sma(sma_diff.data, int(np.sqrt(length)))


#
# RSI -> Relative Strength Index
#
def get_rsi(df, lenght=14, ema=True):
    """Getter for relative strength index. Returns a pd.Series.

    Args:
        df (_type_): input dataframe from which rsi is calculated
        lenght (int, optional): _description_. Defaults to 14.
        ema (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    close_delta = df['close'].diff()

    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)

    if ema is True:
        # Use exponential moving average
        ma_up = up.ewm(com=lenght - 1, min_periods=lenght).mean()
        ma_down = down.ewm(com=lenght - 1, min_periods=lenght).mean()
    else:
        # Use simple moving average
        ma_up = up.rolling(window=lenght, adjust=False).mean()
        ma_down = down.rolling(window=lenght, adjust=False).mean()

    rsi = ma_up / ma_down
    rsi = 100 - (100/(1 + rsi))
    return rsi, ma_up, ma_down


#
# Scripting
#
if __name__ == '__main__':
    """TEST and Prototyping Location
    """

    #
    # EMA/SMA test
    #
    df = get_stored_prices(interval=Intervals.fifteenMin).tail(240)
    ema20 = get_ema(df['close'], 100)
    sma20 = get_sma(df['close'], 20)
    ema50 = get_ema(df['close'], 50)
    sma50 = get_sma(df['close'], 50)
    ema200 = get_ema(df['close'], 200)
    sma200 = get_sma(df['close'], 200)

    plt.figure(figsize=[24, 10])
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.plot(df['close'], label='Closing Prices')
    plt.plot(ema20, label='EMA 20')
    plt.plot(sma20, label='SMA 20')
    plt.plot(ema50, label='EMA 50')
    plt.plot(sma50, label='SMA 50')
    plt.plot(ema200, label='EMA 200')
    plt.plot(sma200, label='SMA 200')
    plt.legend()
    plt.show()

    #
    # RSI test
    #
    df = get_stored_prices(interval=Intervals.fifteenMin)

    rsi, up2, down2 = get_rsi(df, lenght=12)
    plt.figure(figsize=[24, 10])
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.plot(rsi[:100], label='rsi')
    plt.axhline(y=50, color='black', linestyle='--')
    plt.axhline(y=70, color='green', linestyle='--')
    plt.axhline(y=30, color='red', linestyle='--')
    plt.legend()
    plt.show()

    #
    # converting to larger interva Test
    #
    # Fifteen minute candlestick plot
    fifteenMinDF = get_stored_prices().tail(30)
    fig = figure_factory.create_candlestick(fifteenMinDF.open, fifteenMinDF.high, fifteenMinDF.low, fifteenMinDF.close, dates=fifteenMinDF.dateTime)
    fig.show()

    df = n_times_interval(get_stored_prices(since=timeToTimeStamp(year=2022, month=Months.march, day=30, hour=0, minute=0, second=0)), n=2)
    fig = figure_factory.create_candlestick(df.open, df.high, df.low, df.close, dates=df.dateTime)
    fig.show()

    #
    # MACD Test
    #
    df = get_stored_prices(interval=Intervals.fifteenMin).tail(240)

    macd, macdsignal = get_macd(df.close, fastperiod=12, slowperiod=26, signalperiod=9)
    plt.figure(figsize=[24, 10])
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.plot(macd, label='macd')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.plot(macdsignal, label='signal')
    plt.legend()
    plt.show()
