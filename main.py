#%%
from sklearn.model_selection import TimeSeriesSplit, train_test_split
import numpy as np
import plotly.io as pio
pio.renderers.default = "browser"
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
# import data_fetcher
import plotly.express as px
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
#%%
cdf = data_fetcher.CryptoDataFetcher()
#%%
# df = cdf.get_data(start = '2017-08-01',end='2024-01-16',market='XRM/USDT')[-1000:].reset_index(drop=True)

path = r'/Users/clas/Documents/python/Learning to trade with direct RL/csvs/2017-08-01_2024-01-16_BTC-USDT_1h.csv'
df =pd.read_csv(path)[-30000:].reset_index(drop=True)
#%%

class TradingModel(nn.Module):
    def __init__(self, lookback=100, fees=0.00025, use_mlp=False):

        super(TradingModel, self).__init__()
        self.lookback = lookback
        self.fees = fees
        self.theta = torch.rand(lookback+2, requires_grad=True, dtype=torch.float32) # +1 for bias, +1 for previous position
        self.P = torch.zeros(1, dtype=torch.float32)  # Positions as 
        self.use_mlp = use_mlp
        self.build_mlp()

    def build_mlp(self):
        input_size = self.lookback + 2
        hidden_size = int(self.lookback / 2)
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, 1)
        )
    
    def build_gru(self):
        input_size = self.lookback + 2
        hidden_size = int(self.lookback / 2)
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.gru_model = nn.Sequential(
            self.gru,
            self.fc
        )

    def positions(self, features):
        len_time_series = len(features)
        P_list = [torch.rand(1) for x in range(3)]  # Initialize with ones = start out long. minus one to account for the the first position
        # P_list = [torch.rand(1) for x in range(self.lookback-1)]  # Initialize with ones = start out long. minus one to account for the the first position

        # for t in range(self.lookback, len_time_series+1):
        #     feature_inputs = features[t - self.lookback:t] # i.e. last n returns
        for t in range(1, len_time_series):
            feature_inputs = features[t].flatten() # i.e. last n returns

            feature_inputs_norm = (feature_inputs - feature_inputs.mean()) / feature_inputs.std()

            bias = torch.tensor([1.0], dtype=torch.float32) 
            
            n_last_positions = torch.cat(P_list)

            P_list_norm = ((n_last_positions - n_last_positions.mean()) / n_last_positions.std()).flatten()

            last_position = P_list_norm[-1].unsqueeze(0)
            
            # xt = torch.cat([feature_inputs_norm]).type(torch.float32)
            xt = torch.cat([feature_inputs_norm, last_position, bias]).type(torch.float32)
            xt = (xt - xt.mean()) / xt.std()
            if self.use_mlp:
                new_position = torch.sigmoid(self.mlp(xt))
                P_list.append(new_position)
            else:
                new_position = (torch.sigmoid(torch.dot(self.theta,xt)))
                P_list.append(new_position.unsqueeze(0))

        P = torch.cat(P_list).type(torch.float32)
        return P

    def get_weighted_returns(self, P, asset_returns, penalty_factor=0.1):
        '''alternative to the get_bot_return function without the need for a for loop'''

        if not len(asset_returns) == len(P):
            raise ValueError("The lengths of returns and positions must be the same.")

        log_returns = torch.log(asset_returns)
        position_changes = torch.abs(torch.diff(P, prepend=torch.tensor([0.0])))  # Prepend zero for the initial position change
        position_changes = torch.cat((torch.zeros(1), position_changes[:-1]))  # Prepend zero to keep penalties in line with returns
        penalties = penalty_factor * position_changes

        mult_P = torch.cat((torch.ones(1), P[:-1]))  # prepend one to shift the returns forward

        weighted_returns = log_returns * mult_P 

        net_returns = weighted_returns - penalties

        return net_returns

    def returns(self, P, asset_returns):
        if not (len(asset_returns) == len(P)):
            raise ValueError("The lengths of returns, and positions must be the same.")

        # Initialize variables
        portfolio_values = [torch.tensor(1,dtype=torch.float32)]
        investment_value = portfolio_values[-1] * P[0] # start out with some percentage invested at day 0
        current_cash = 1 - investment_value
        self.fees_paid = [0]

        # Lets say lookback is 3. Positions 0 to lookback-1 are the initial random positions. So we have positions for day 0, 1 and 2
        # We start at the end of day 2 (or beginning of day 3), use the returns of day 0,1,2 to calculate the positions for day 3.
        # at the end of day 2 we have and investment value of P[2]*portfolio_value.
        # 1. the amount invested at the end of day 2 (investment_value) experiences the return of day 3 (asset_returns[i] for i in range(1, len(asset_returns)))
        # 2. at the end of day 3 we compare the new investment value with the desired investment value (portfolio_value * P[3] ) and trade the difference
        # 3. update the investment, cash and portfolio value

        for i in range(1, len(asset_returns)):
            # Calculate asset value change
            investment_value = investment_value * asset_returns[i] # the amount invested at day 0 times the return of day 1
            portfolio_value = investment_value + current_cash 

            desired_investment_value = portfolio_value * P[i] # Desired investment value on day 1
            transaction_amount = abs(desired_investment_value - investment_value)

            # Adjust investment and cash to maintain target weight
            if desired_investment_value > investment_value:
                # Buy more of the asset
                investment_value += transaction_amount * (1 - self.fees)
                current_cash -= transaction_amount

            elif desired_investment_value < investment_value:
                # Sell some of the asset
                investment_value -= transaction_amount
                current_cash += transaction_amount * (1 - self.fees)
            
            self.fees_paid.append(transaction_amount * self.fees)

            portfolio_values.append(investment_value + current_cash)
        
        # Convert portfolio values to percentage changes
        portfolio_values = torch.stack(portfolio_values)
        bot_returns = portfolio_values[1:] / portfolio_values[:-1]
        bot_returns = torch.cat((torch.ones(1, dtype=torch.float32), bot_returns))

        return bot_returns

    def sharpe_ratio(self, returns):
        if returns.mean() > 0.5:
            print("Are you sure you are using log returns or percentage returns? The mean of the returns is very high")
        return returns.mean() / returns.std()

    def sortino_ratio(self, returns, target_return=0):
        if returns.mean() > 0.5:
            print("Are you sure you are using log returns or percentage returns? The mean of the returns is very high")
        downside_returns = torch.where(returns < target_return, returns, torch.zeros_like(returns))
        return (returns.mean() - target_return) / downside_returns.std()
    
    def forward(self, asset_returns, features):
        
        P = self.positions(features)
        bot_returns = self.returns(P, asset_returns)

        return bot_returns

    def train_model(self, asset_returns, features, epochs=2000, learning_rate=0.3, optimization_target="sharpe"):

        if self.use_mlp:
            parameters_to_optimize = self.parameters()
        else:
            parameters_to_optimize = [self.theta]

        optimizer = torch.optim.Adam(parameters_to_optimize, lr=learning_rate)

        self.model_wandb = []
        for i in range(epochs):
            optimizer.zero_grad()
            bot_returns = self.forward(asset_returns,features)
            
            if optimization_target == "sharpe":
                ratio = self.sharpe_ratio(bot_returns-1)
            elif optimization_target == "sortino":
                ratio = self.sortino_ratio(bot_returns-1)
            elif optimization_target == "sum_return":
                ratio = (bot_returns).sum()
            elif optimization_target == "prod_return":
                ratio = (bot_returns).prod()
            elif optimization_target == "mean_return":
                ratio = (bot_returns).mean()
            else:
                raise ValueError("Invalid optimization target. Choose sharpe, sortino, sum_return, prod_return or mean_return")
            
            (-ratio).backward()  # Maximizing the ratio

            optimizer.step()
            self.model_wandb.append(parameters_to_optimize)
            if i % 10 == 0 or i == epochs-1 or i <=5:
                print(f'Epoch {i} ratio: {np.round(ratio.item(),10)}, mean returns: {np.round(bot_returns.mean().item(),10)}')
        print("finished training")
        self.ratio = ratio.item()
        return self, ratio.item()
    
    def walk_forward(self, asset_returns, features, epochs=200, learning_rate=0.3, optimization_target="sharpe", num_splits=10, max_train_size=None):
        tss = TimeSeriesSplit(n_splits=num_splits, max_train_size=max_train_size)

        self.walk_forward_positions = []
        for i, (train_index, test_index) in enumerate(tss.split(asset_returns)):
            print(f"Split {i+1} of {num_splits}, {train_index[0]} - {train_index[-1]} and {test_index[0]} - {test_index[-1]}")
            len_test = len(test_index)

            train_test_index = np.concatenate((train_index, test_index))
            self.theta = torch.rand(self.lookback+2, requires_grad=True, dtype=torch.float32) # reset weights
            self.build_mlp()

            self.train_model(asset_returns[train_index], features[train_index], epochs, learning_rate, optimization_target)
            # run on train and test set to avoid the "warm up"/lookback period of the model.
            P = self.positions(features[test_index]) 
            # just take the test period
            positions_test = P#[-len_test:]
            self.walk_forward_positions.append(positions_test)

        
        self.walk_forward_positions = torch.cat(self.walk_forward_positions)
        len_mising = len(asset_returns)-len(self.walk_forward_positions) # missing due to the first train set

        self.walk_forward_positions = torch.cat((torch.ones(len_mising, dtype=torch.float32), self.walk_forward_positions))

        assert len(self.walk_forward_positions) == len(asset_returns), "The length of the walk forward positions should be the same as the asset returns"

    def plot(self, asset_returns, features, use_matplotlib=False):
        def ar_positions(n, alpha=0.99, sigma=0.01):
            ar = np.zeros(n)
            ar[0] = 0
            for i in range(1, n):
                ar[i] = alpha * ar[i-1] + np.random.normal(0, sigma)
            
            min_val = np.min(ar)
            max_val = np.max(ar)
            ar = (ar - min_val) / (max_val - min_val)
            return ar
        random_ar_positions = np.array([0.5 for x in range(len(asset_returns))]) #ar_positions(len(asset_returns))
        random_positions = torch.from_numpy(random_ar_positions)
        untrained_bot_returns = self.returns(random_positions, asset_returns).detach().numpy().cumprod()
        P = self.walk_forward_positions
        trained_bot_returns = self.returns(P, asset_returns).detach().numpy().cumprod()
        asset_returns = asset_returns.detach().numpy().cumprod()
        x_range = list(range(len(untrained_bot_returns)))
        data = {
            'Untrained Bot Returns': untrained_bot_returns,
            'Trained Bot Returns': trained_bot_returns,
            'Asset Returns': asset_returns
        }

        df = pd.DataFrame(data, index=x_range)
    
        if use_matplotlib:
            fig, axs = plt.subplots(2, figsize=(10,12))
            for column in df.columns:
                axs[0].plot(df.index, df[column], label=column)
            axs[0].set_xlabel('Time')
            axs[0].set_ylabel('Cumulative Returns')
            axs[0].set_title('Combined Returns Over Time')
            axs[0].legend()
            axs[0].set_yscale('log')
        
            axs[1].plot(P.detach().numpy(), label='Position')
            axs[1].set_xlabel('Time')
            axs[1].set_ylabel('Position')
            axs[1].set_title('Position Over Time')
            axs[1].legend()
        
            plt.tight_layout()
            plt.show()
        else:
            fig_combined = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
            
            for variable in df.columns:
                fig_combined.add_trace(
                    go.Scatter(x=df.index,y=df[variable],name=variable,mode='lines'),
                    row=1,col=1)

            fig_combined.add_trace(
                go.Scatter(x=x_range,y=P.detach().numpy(),name='P',mode='lines'), 
                row=2,col=1)
            
            fig_combined.add_trace(
                go.Scatter(x=x_range,y=random_positions,name='Random Positions',mode='lines'), 
                row=2,col=1)
            
            fig_combined.update_layout(title='Combined Returns Over Time')
            fig_combined.update_yaxes(type="log", row=1)
            fig_combined.update_yaxes(title_text="P", row=2)
            
            fig_combined.show()


#%%
def create_zig_zag_target(df,col, threshold, return_df=False):
    mode = "up"
    lowest = df.loc[0, col]
    highest = df.loc[0, col]

    lowest_ix = 0
    highest_ix = 0

    for ix, row in df.iterrows():
        if mode == "up":
            if row[col] > highest:
                highest = row[col]
                highest_ix = ix

            if highest / row[col] > threshold:
                mode = "down"
                df.loc[highest_ix, "pivot"] = highest
                lowest = row[col]
                highest = row[col]
                lowest_ix = ix
                highest_ix = ix

        elif mode == "down":
            if row[col] < lowest:
                lowest = row[col]
                lowest_ix = ix

            if row[col] / lowest > threshold:
                mode = "up"
                df.loc[lowest_ix, "pivot"] = lowest
                highest = row[col]
                lowest = row[col]
                lowest_ix = ix
                highest_ix = ix

    df["pivot_fill"] = df["pivot"].bfill()
    df["pivot_fill_shift"] = df["pivot_fill"].shift(-1)
    df["next_pivot"] = np.where(
        df["pivot_fill"].shift(-1) != df["pivot_fill"],
        df["pivot_fill"].shift(-1),
        np.nan,
    )
    df["middle_level"] = ((df["pivot"] + df["next_pivot"]) / 2).ffill()

    df["target"] = np.where(df[col] > df["middle_level"], 0, 1)
    # df["target"] = df["close"] > df["middle_level"]

    # df.drop(['pivot_fill','pivot_fill_shift','next_pivot','middle_level','maximum','pivot','minimum'],axis=1,inplace=True)
    if return_df:
        return df
    else:
        return df["target"]

# %%
from sklearn.model_selection import train_test_split
import numpy as np

def sample_array(input_array, probability=0.59):
    mask = np.random.choice([True, False], size=input_array.shape, p=[probability, 1-probability])
    return np.where(mask, input_array, 1-input_array),mask

lb = 10
# signal = np.where(df['close'].rolling(lb,0).mean().shift(-lb) > df['close'], 1, 0)
signal = create_zig_zag_target(df,'close',1.05)
#%%
sampled_array,mask = sample_array(signal)
#%%
asset_returns = torch.from_numpy(df['asset_return'].values).type(torch.float32)
features = torch.from_numpy(
    ((df['close'] - df['close'].rolling(50,0).mean())/df['close'].rolling(50).std()).values
    ).type(torch.float32)

features2 = torch.from_numpy(
    ((df['close'] - df['close'].rolling(10,0).mean())/df['close'].rolling(10).std()).values
    ).type(torch.float32)
preds = torch.from_numpy(sampled_array).type(torch.float32)
features = torch.stack([features,features2,asset_returns,preds],dim=1)
#%%
pd.Series(features.detach().numpy()).tail(2000).head(10000).hist()

#%%
tm = TradingModel(use_mlp=True,lookback=4,fees=0.00025)
tm.walk_forward(asset_returns, 
                features, 
                epochs = 10, 
                learning_rate = 0.1, 
                optimization_target = "mean_return", 
                num_splits = 10, 
                max_train_size = 2000)
                
tm.plot(asset_returns, features, use_matplotlib=True)
#%%
# Split the data into training and testing sets
features_train, features_test, asset_returns_train, asset_returns_test = train_test_split(features, asset_returns, test_size=0.2, random_state=42,shuffle=False)
tm = TradingModel(use_mlp=True,lookback=50,fees=0.00025)

tm.train_model(asset_returns_train, features_train, epochs=10, learning_rate=0.1, optimization_target="mean_return")
tm.plot(asset_returns_test, features_test, use_matplotlib=True)
tm.plot(asset_returns, features, use_matplotlib=True)
#%%
br = tm.forward(asset_returns, features)
#%%
pd.Series(br.detach().numpy()).tail(2000).head(10000).cumprod().plot()
pd.Series(asset_returns.detach().numpy()).tail(2000).head(10000).cumprod().plot()
#%%

P = tm.positions(features)
pd.Series(P.detach().numpy()).hist(bins=100)

 # %%
torch.save(tm.state_dict(), 'model.pth')  
#%%

#%%
import boto3

s3 = boto3.client('s3')
#%%
bucket_name = 'your-s3-bucket-name'  
key = 'models/model.pth'  # Path within the bucket

s3.upload_file('model.pth', bucket_name, key)
#%%

input_size = 10
hidden_size = 5
gru = nn.GRU(input_size, hidden_size, batch_first=True)
fc = nn.Linear(hidden_size, 1)
gru_model = nn.Sequential(gru)
gru_model(torch.rand(1,10,10))[0].shape
# %%
gru_model(torch.rand(1,10,10))[1].shape
