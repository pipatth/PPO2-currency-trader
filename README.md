# PPO2-currency-trader

Build reinforcement learning agents to trade USD_CAD and USD_CHF using data from OANDA v20 API.

After training, you can visualize the agents' trading activities (using testing data) on Dash app.

![Alt text](assets/dash_example.gif?raw=true "Title")

Credits: 
https://github.com/notadamking/RLTrader for gym environment codes, tutorials, and idea of using RL in trading
https://dash-gallery.plotly.host/Portal/ for Dash template

1. Install dependencies
- Make sure you have libopenmpi-dev 
```
$ sudo apt install libopenmpi-dev
```
- If using GPU
```
$ pip install -r requirements-gpu.txt
```
- If not using GPU
```
$ pip install -r requirements-nogpu.txt
```
2. Create a config file in ~/.v20.conf using information from your OANDA account. For example, 
```
hostname: api-fxpractice.oanda.com
streaming_hostname: stream-fxpractice.oanda.com
port: 443
ssl: true
token: XXXXX
username: XXXXX
datetime_format: RFC3339
accounts:
- XXX-XXX-XXXXXXX-XXX
active_account: XXX-XXX-XXXXXXX-XXX
```
3. Download data from OANDA
```
$ python data.py
```
4. Train models using PPO2. For example,
```
$ python model.py --symbol EUR_USD --n 10000
```
5. Visualize agent training
```
$ python app.py
```