from agent.TradingAgent import TradingAgent
import pandas as pd
import numpy as np


class yshah72_dopamine(TradingAgent):
    """
    Simple Trading Agent that compares the past mid-price observations and places a
    buy limit order if the first window mid-price exponential average >= the second window mid-price exponential average or a
    sell limit order if the first window mid-price exponential average < the second window mid-price exponential average
    """

    def __init__(self, id, name, type, symbol, starting_cash,
                 min_size, max_size, wake_up_freq='60s',
                 log_orders=False, random_state=None):

        super().__init__(id, name, type, starting_cash=starting_cash, log_orders=log_orders, random_state=random_state)
        self.symbol = symbol
        self.min_size = min_size  # Minimum order size
        self.max_size = max_size  # Maximum order size
        self.size = self.random_state.randint(self.min_size, self.max_size)
        self.wake_up_freq = wake_up_freq
        self.log_orders = log_orders
        self.state = "AWAITING_WAKEUP"
        self.window1 = 20
        self.window2 = 5
        self.extended_stream_history = {}
        self.MinLength = 10
        self.history = []
        self.ordersMade = {}
        self.orderCount = 0
        self.yashWrite = open('/home/dopamine/yshah72_dopamine.txt', 'w')

    def kernelStarting(self, startTime):
        super().kernelStarting(startTime)

    def wakeup(self, currentTime):
        """ Agent wakeup is determined by self.wake_up_freq """
        can_trade = super().wakeup(currentTime)
        if not can_trade: return
        self.getCurrentSpread(self.symbol)

        self.state = 'AWAITING_SPREAD'

    def placeOrder(self, currentTime):
        # self.yashWrite.write(f" currentTime = {currentTime}, marketClose = {self.mkt_close} \n")
        if self.mkt_close - currentTime <= pd.Timedelta('5 min'):
            if self.symbol in self.holdings and self.holdings[self.symbol] > 0:
                limit_price = self.getLimitPrice(False)
                self.placeLimitOrder(self.symbol, quantity=self.holdings[self.symbol], is_buy_order=False,
                                     limit_price=limit_price)
            return

        if self.symbol in self.extended_stream_history and len(
                self.extended_stream_history[self.symbol]) < self.MinLength:
            if self.symbol in self.extended_stream_history:
                self.extended_stream_history[self.symbol] = (self.extended_stream_history[self.symbol] +
                                                             self.stream_history[self.symbol])
            else:
                self.extended_stream_history[self.symbol] = self.stream_history[self.symbol]
            return

        self.yashWrite.write(f"In PlaceOrder \n")
        # first determine whether to buy or sell
        bid, _, ask, _ = self.getKnownBidAsk(self.symbol)
        toBuy = None
        if bid and ask:
            self.history.append((bid + ask) / 2)
            # using Bollinger Bands to determine whether to buy or sell
            if len(self.history) > self.window1:
                s = pd.Series(self.history)
                upper = s.rolling(window=self.window1).mean() + 1.5 * s.rolling(window=self.window1).std()
                lower = s.rolling(window=self.window1).mean() - 1.5 * s.rolling(window=self.window1).std()
                if self.history[-1] >= upper.values[-1]: # sell order
                    toBuy = False
                elif self.history[-1] <= lower.values[-1]: # buy order
                    toBuy = True


        if toBuy is None:
            return
        # Now determine at what price
        # iterate over the stream history and 
        limit_price = self.getLimitPrice(toBuy)

        self.placeLimitOrder(self.symbol, quantity=self.size, is_buy_order=toBuy, limit_price=limit_price)

        self.orderCount += 1
        self.ordersMade[self.orderCount] = {'limit_price': limit_price, 'is_buy_order': toBuy, 'quantity': self.size}

        self.yashWrite.write(f"Place order at {'buy' if toBuy else 'sell'} {limit_price}  \n")

    def getLimitPrice(self, toBuy):
        min_price = np.inf
        max_price = -np.inf

        for history in self.stream_history[self.symbol]:
            for _, order in history.items():
                if order['limit_price'] < min_price:
                    min_price = order['limit_price']
                if order['limit_price'] > max_price:
                    max_price = order['limit_price']

        # now figure out what price-orders have not been fulfilled yet
        nd = np.zeros((max_price - min_price + 1, 2))
        for history in self.stream_history[self.symbol]:
            for _, order in history.items():
                if order['is_buy_order']:
                    nd[order['limit_price'] - min_price, 0] += order['quantity']
                else:
                    nd[order['limit_price'] - min_price, 1] += order['quantity']

        # now subtract both columns to figure out how many orders at what prices are left 
        nd = nd[:, 0] - nd[:, 1]
        # if the columns are negative, that means there are more sell orders than buy orders
        # if the columns are positive, that means there are more buy orders than sell orders
        buy_nd = nd[nd > 0]
        sell_nd = nd[nd < 0]

        bid, _, ask, _ = self.getKnownBidAsk(self.symbol)

        s = pd.Series(self.history).ewm(span=self.window1).mean()

        if toBuy:
            # buy at the lowest price
            if buy_nd.shape[0] == 0 and sell_nd.shape[0] != 0:
                idx = np.where(nd < 0)[0]
                idx = 0 if len(idx) == 0 else idx[0]
                # limit_price = np.argmin(sell_nd) + min_price
                limit_price = min_price + idx
            #     self.yashWrite.write(f"Setting Limit Price for Buy by First Condition {limit_price} \n")
            #
            elif buy_nd.shape[0] != 0:
                limit_price = np.argmin(buy_nd) + min_price
            else:
                limit_price = (min_price + max_price) // 2

        else:
            # sell at the highest price
            if sell_nd.shape[0] == 0 and buy_nd.shape[0] != 0:
                idx = np.where(nd > 0)[0]
                idx = 0 if len(idx) == 0 else idx[-1]
                # limit_price = np.argmax(buy_nd) + min_price
                limit_price = min_price + idx
                self.yashWrite.write(f"Setting Limit Price for Sell by First Condition {limit_price} \n")
            elif sell_nd.shape[0] != 0:
                limit_price = np.argmax(sell_nd) + min_price
            else:
                limit_price = (min_price + max_price) // 2
                # limit_price = s.iloc[-1]
            # limit_price = max_price - 10


        return limit_price

    def receiveMessage(self, currentTime, msg):
        """ Momentum agent actions are determined after obtaining the best bid and ask in the LOB """
        super().receiveMessage(currentTime, msg)
        if self.state == 'AWAITING_SPREAD' and msg.body['msg'] == 'QUERY_SPREAD':
            # self.yashWrite.write(f"Received spread \n")
            self.getOrderStream(self.symbol)
            self.state = 'AWAITING_ORDER_STREAM'

        elif self.state == 'AWAITING_ORDER_STREAM' and msg.body['msg'] == 'QUERY_ORDER_STREAM':
            # self.yashWrite.write(f"Received order stream \n")
            self.placeOrder(currentTime)
            self.setWakeup(currentTime + self.getWakeFrequency())
            self.state = 'AWAITING_WAKEUP'

    def getWakeFrequency(self):
        return pd.Timedelta(self.wake_up_freq)

    def author(self):
        return 'yshah72'

    def agentname(self):
        return 'yshah72_dopamine'

    def number_of_counting(self):
        return self.orderCount
