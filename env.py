import gym
import pandas as pd
import numpy as np
from gym import spaces
from sklearn import preprocessing

# Trade Environment class
class TradeEnv(gym.Env):
    # init
    def __init__(
        self,
        df,
        commission=0.0,
        margin_req=1.00,
        closeout_req=0.5,
        slippage=0.0,
        initial=100000,
        serial=False,
        max_steps=200,
    ):
        super(TradeEnv, self).__init__()
        self.lookback_sz = 50
        self.commission = commission
        self.margin_req = margin_req
        self.closeout_req = closeout_req
        self.slippage = slippage
        self.initial = initial
        self.serial = serial
        self.max_steps = max_steps
        self.colTime = "time"
        self.cols = {
            "volume": "volume",
            "ask": "ask.c",
            "bid": "bid.c",
            "open": "mid.o",
            "high": "mid.h",
            "low": "mid.l",
            "close": "mid.c",
        }
        self.ac_cols = ["nav", "unit_long", "value_long", "unit_short", "value_short"]
        self.df = df.dropna().sort_values(self.colTime)
        self.action_space = spaces.MultiDiscrete([3, 10])
        self.df_sz = self.df.shape[1] - 1
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.df_sz + len(self.ac_cols), self.lookback_sz + 1),
            dtype=np.float16,
        )
        self.scaler_std = preprocessing.StandardScaler()
        self.scaler_mixmax = preprocessing.MinMaxScaler()

    # reset
    def reset(self):
        self.balance = self.initial
        self.units_open = []  # FIFO queue
        self.prices_open = []
        self._reset_pos()

        # account history df
        ac_array = np.repeat([[self.initial, 0, 0, 0, 0]], self.lookback_sz + 1, axis=0)
        self.ac_hist = pd.DataFrame(ac_array, columns=self.ac_cols)

        # price history df
        pr_cols = [self.cols[k] for k in self.cols.keys()]
        self.pr_hist = self.df.loc[
            self.start_pos - self.lookback_sz : self.start_pos, pr_cols
        ]
        self.pr_hist.columns = [k for k in self.cols.keys()]

        # trade history df
        tr_cols = ["step", "type", "unit", "total"]
        self.tr_hist = pd.DataFrame(None, columns=tr_cols)

        return self._next_obs()

    # reset starting position to zero, random pick position if `serial == False`
    def _reset_pos(self):
        self.current_step = 0
        if self.serial:
            self.remaining_steps = len(self.df) - self.lookback_sz - 1
            self.start_pos = self.lookback_sz
        else:
            self.remaining_steps = np.random.randint(1, self.max_steps)
            self.start_pos = np.random.randint(
                self.lookback_sz, len(self.df) - self.remaining_steps
            )
        self.active_df = self.df[
            self.start_pos - self.lookback_sz : self.start_pos + self.remaining_steps
        ]

    # get next obs
    def _next_obs(self):
        # OHLCV info
        end = self.current_step + self.lookback_sz + 1
        obs = (
            self.active_df.iloc[self.current_step : end]
            .drop(self.colTime, axis=1)
            .values.T
        )

        # append scaled history
        self.scaler_mixmax.fit(self.ac_hist.T)
        scaled_hist = self.scaler_mixmax.fit_transform(self.ac_hist.T)
        obs = np.append(obs, scaled_hist[:, -(self.lookback_sz + 1) :], axis=0)
        return obs

    # getter current price row
    def _get_pr_row(self):
        pr_cols = [self.cols[k] for k in self.cols.keys()]
        pr_row = self.df.loc[self.start_pos + self.current_step, pr_cols]
        pr_row.index = [k for k in self.cols.keys()]
        return pr_row

    # getter current position value. position = unit * current bid (when long) or ask (when short)
    def _get_position_value(self):
        pr_row = self._get_pr_row()
        unit = sum(self.units_open)
        return unit * (pr_row["bid"] if unit > 0 else pr_row["ask"])

    # getter margin used. margin_used = current position * margin req
    def _get_margin_used(self):
        return abs(self._get_position_value()) * self.margin_req

    # getter current nav. nav = balance + unrealized p/l using current bid or ask
    def _get_nav(self):
        value_open = sum(
            [unit * price for unit, price in zip(self.units_open, self.prices_open)]
        )
        return self.balance + self._get_position_value() - value_open

    # getter margin closeout value. mcv = balance + unrealized p/l using current mid
    def _get_mcv(self):
        pr_row = self._get_pr_row()
        value_open = sum(
            [unit * price for unit, price in zip(self.units_open, self.prices_open)]
        )
        value_current = sum(self.units_open) * pr_row["close"]
        return self.balance + value_current - value_open

    # open position. if unit = -1 means short
    def _open_pos(self, unit, price):
        unit_filled = 0
        # try closing first if possible
        unit_filled += self._close_pos(unit, price)
        if unit != unit_filled:
            # append to queue
            unit_to_fill = unit - unit_filled
            self.units_open.append(unit_to_fill)
            self.prices_open.append(price)
            # pay commission
            self.balance -= abs(unit_to_fill) * price * self.commission
            unit_filled += unit_to_fill
        return unit_filled

    # close position. if unit = -1 means take short to close a long. return unit requested that left unfilled
    def _close_pos(self, unit, price):
        unit_filled = 0
        unit_to_fill = unit - unit_filled
        # close only if signs are opposite
        if unit * sum(self.units_open) < 0:
            # go to each order, close as many as possible
            while unit_to_fill != 0 and self.units_open:
                u = self.units_open[0]
                p = self.prices_open[0]
                # if need to close only partial
                if abs(u) > abs(unit_to_fill):
                    value_open = -unit_to_fill * p
                    value_current = -unit_to_fill * price
                    self.balance += value_current - value_open  # adjust balance
                    self.balance -= abs(unit_to_fill) * price * self.commission
                    self.units_open[0] + unit_to_fill  # adjust inventory
                    unit_filled += unit_to_fill
                    unit_to_fill -= unit_to_fill
                    break
                # if close all
                else:
                    value_open = u * p
                    value_current = u * price
                    self.balance += value_current - value_open  # adjust balance
                    self.balance -= abs(u) * price * self.commission
                    self.units_open.pop(0)  # remove
                    self.prices_open.pop(0)  # remove
                    unit_filled += -u
                    unit_to_fill -= -u
        return unit_filled

    # check whether margin closeout or not
    def _is_closeout(self):
        if self._get_mcv() < (self.closeout_req * self._get_margin_used()):
            return True
        else:
            return False

    # take a step forward
    def step(self, action):
        pr_row = self._get_pr_row()
        self._take_action(action, pr_row)
        self.remaining_steps -= 1
        self.current_step += 1
        # end of episode, force close
        if self.remaining_steps == 0:
            unit_to_close = -sum(self.units_open)  # reverse to close
            if unit_to_close > 0:
                price = pr_row["ask"]
                self._close_pos(unit_to_close, price)
            elif unit_to_close < 0:
                price = pr_row["bid"]
                self._close_pos(unit_to_close, price)
            self._reset_pos()
        obs = self._next_obs()
        reward = self._get_nav()
        done = self._is_closeout()

        # append next price to pr_hist
        pr_cols = [self.cols[k] for k in self.cols.keys()]
        pr_next = self.df.loc[self.start_pos + self.current_step, pr_cols]
        pr_next.index = [k for k in self.cols.keys()]
        self.pr_hist = self.pr_hist.append(pd.DataFrame(pr_next).T)

        return obs, reward, done, {}

    # action
    def _take_action(self, action, pr_row):
        action_type = action[0]
        proportion = action[1] / 10
        avail = self._get_nav() - self._get_margin_used()  # margin available to use
        unit_filled = 0
        price = 0
        # take pos
        if avail > 0:
            # long
            if action_type == 0:
                price = pr_row["ask"] * (1 + self.slippage)
                unit = int(avail / price * proportion)  # unit to long
                unit_filled = self._open_pos(unit, price)
            # short
            elif action_type == 1:
                price = pr_row["bid"] * (1 - self.slippage)
                unit = -int(avail / price * proportion)  # unit to short
                unit_filled = self._open_pos(unit, price)

        # record trade history
        if unit_filled != 0:
            tr = pd.DataFrame(
                [
                    [
                        self.start_pos + self.current_step,
                        "long" if unit_filled > 0 else "short",
                        unit_filled,
                        unit_filled * price,
                    ]
                ],
                columns=self.tr_hist.columns,
            )
            self.tr_hist = self.tr_hist.append(tr, ignore_index=True)

        # record account history
        if unit_filled > 0:
            unit_long = abs(unit_filled)
            unit_short = 0
        elif unit_filled < 0:
            unit_long = abs(unit_filled)
            unit_short = 0
        else:
            unit_long = 0
            unit_short = 0
        df_ = pd.DataFrame(
            [
                [
                    self._get_nav(),
                    unit_long,
                    unit_long * price,
                    unit_short,
                    unit_short * price,
                ]
            ],
            columns=self.ac_hist.columns,
        )
        self.ac_hist = pd.concat([self.ac_hist, df_], ignore_index=True)

    # get summary
    def get_summary(self):
        accounts = self.ac_hist.reset_index(drop=True)
        prices = self.pr_hist.reset_index(drop=True)
        accounts["gl"] = accounts["nav"].diff().fillna(0)
        accounts["ret"] = (accounts["gl"] / accounts["nav"].shift(1)).fillna(0)
        return pd.concat([prices, accounts], axis=1)
