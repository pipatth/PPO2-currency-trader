from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from env import TradeEnv
import pandas as pd
import argparse


# Loading historical tick data, model, env
def load_env_model(asset, test=True):

    # if testing, using serial mode
    if test:
        data_file = asset + "_test.tsv"
        serial = True
    # if training using random mode
    else:
        data_file = asset + "_train.tsv"
        serial = False

    # load data
    try:
        df = pd.read_csv(data_file, sep="\t")
    except FileNotFoundError:
        print("No data for " + asset)
        exit()
    df = df[
        [
            "time",
            "ask.c",
            "ask.h",
            "ask.l",
            "ask.o",
            "bid.c",
            "bid.h",
            "bid.l",
            "bid.o",
            "mid.c",
            "mid.h",
            "mid.l",
            "mid.o",
            "volume",
        ]
    ]

    # load environment
    env = DummyVecEnv([lambda: TradeEnv(df, serial=serial)])

    # load existing model
    try:
        model = PPO2.load(asset, env, verbose=1)
        print("Loaded existing model " + asset)
        return env, model
    except ValueError:
        print("No model for " + asset)
        if test:
            exit()
        else:
            print("Starting new model")
            model = PPO2(MlpPolicy, env, verbose=1)
            return env, model


# train model
def train(asset, n):
    # create model
    _, model = load_env_model(asset, test=False)

    # train model
    model.learn(total_timesteps=n)

    # save model
    model.save(asset)
    print("Training done, saved to " + asset)


# run
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", help="asset symbol e.g. USD_CAD", default="USD_CAD")
    parser.add_argument(
        "--n", help="number of sessions e.g. 1000", default=1000, type=int
    )
    args = vars(parser.parse_args())
    train(args["symbol"], n=args["n"])

