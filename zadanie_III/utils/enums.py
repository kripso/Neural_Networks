from enum import Enum


class PriceStates(Enum):
    hold = 'hold'
    rising = 'rising'
    falling = 'falling'
    risingPullBack = 'risingPullBack'
    fallingPullBack = 'fallingPullBack'


class Intervals(Enum):
    minute = 1
    fiveMin = 5
    fifteenMin = 15
    halfHour = 30
    hour = 60
    fourHours = 240
    day = 1440
    week = 10080
    halfMonth = 21600


class OrderTypes(Enum):
    market = "market"
    limit = "limit"
    stop_loss = "stop-loss"
    take_profit = "take-profit"
    stop_loss_limit = "stop-loss-limit"
    take_profit_limit = "take-profit-limit"
    settle_position = "settle-position"


class Type(Enum):
    buy = 'buy'
    sell = 'sell'


class Months(Enum):
    january = 1
    february = 2
    march = 3
    april = 4
    may = 5
    june = 6
    july = 7
    august = 8
    september = 9
    october = 10
    november = 11
    december = 12
