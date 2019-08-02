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
        slippage=0.0,
        initial=100000,
        serial=False,
        max_steps=200,
    ):
        super(TradeEnv, self).__init__()
        self.lookback_sz = 50
        self.commission = commission
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
        self.df = df.dropna().sort_values(self.colTime)
        self.action_space = spaces.MultiDiscrete([3, 10])
        self.df_sz = self.df.shape[1] - 1
        self.account_history_sz = 5
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.df_sz + self.account_history_sz, self.lookback_sz + 1),
            dtype=np.float16,
        )
        self.scaler_std = preprocessing.StandardScaler()
        self.scaler_mixmax = preprocessing.MinMaxScaler()

    # reset
    def reset(self):
        self.balance = self.initial
        self.net_worth = self.initial
        self.unit = 0
        self._reset_pos()

        # account history df
        ac_cols = ["net_worth", "unit_bought", "cost", "unit_sold", "sales"]
        ac_array = np.repeat(
            [[self.net_worth, 0, 0, 0, 0]], self.lookback_sz + 1, axis=0
        )
        self.ac_hist = pd.DataFrame(ac_array, columns=ac_cols)

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

    # take a step forward
    def step(self, action):
        pr_cols = [self.cols[k] for k in self.cols.keys()]
        pr_row = self.df.loc[self.start_pos + self.current_step, pr_cols]
        pr_row.index = [k for k in self.cols.keys()]
        self._take_action(action, pr_row)
        self.remaining_steps -= 1
        self.current_step += 1
        # end of episode, force sell
        if self.remaining_steps == 0:
            self.balance += (
                self.unit * pr_row["bid"] * (1 - self.commission) * (1 - self.slippage)
            )
            self.unit = 0
            self._reset_pos()
        obs = self._next_obs()
        reward = self.net_worth
        done = self.net_worth <= 0

        # append next price to pr_hist
        pr_next = self.df.loc[self.start_pos + self.current_step, pr_cols]
        pr_next.index = [k for k in self.cols.keys()]
        self.pr_hist = self.pr_hist.append(pd.DataFrame(pr_next).T)

        return obs, reward, done, {}

    # action
    def _take_action(self, action, pr_row):
        action_type = action[0]
        amount = action[1] / 10
        unit_bought = 0
        unit_sold = 0
        cost = 0
        sales = 0
        # buy
        if action_type == 0:
            unit_bought = int(self.balance / pr_row["ask"] * amount)
            cost = (
                unit_bought
                * pr_row["ask"]
                * (1 + self.commission)
                * (1 + self.slippage)
            )
            self.unit += unit_bought
            self.balance -= cost
        # sell
        elif action_type == 1:
            unit_sold = int(self.unit * amount)
            sales = (
                unit_sold * pr_row["bid"] * (1 - self.commission) * (1 - self.slippage)
            )
            self.unit -= unit_sold
            self.balance += sales
        # record trade history
        if unit_sold > 0 or unit_bought > 0:
            tr = pd.DataFrame(
                [
                    [
                        self.start_pos + self.current_step,
                        "sell" if unit_sold > 0 else "buy",
                        unit_sold if unit_sold > 0 else unit_bought,
                        sales if unit_sold > 0 else cost,
                    ]
                ],
                columns=self.tr_hist.columns,
            )
            self.tr_hist = self.tr_hist.append(tr, ignore_index=True)
        # record account history
        self.net_worth = self.balance + self.unit * pr_row["bid"]
        df_ = pd.DataFrame(
            [[self.net_worth, unit_bought, cost, unit_sold, sales]],
            columns=self.ac_hist.columns,
        )
        self.ac_hist = pd.concat([self.ac_hist, df_], ignore_index=True)

    # get summary
    def get_summary(self):
        accounts = self.ac_hist.reset_index(drop=True)
        prices = self.pr_hist.reset_index(drop=True)
        accounts["gl"] = accounts["net_worth"].diff().fillna(0)
        accounts["ret"] = (accounts["gl"] / accounts["net_worth"].shift(1)).fillna(0)
        return pd.concat([prices, accounts], axis=1)
