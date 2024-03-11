
#%%
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import plotly.io as pio
pio.renderers.default = "browser"
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
#%%

path = r'/Users/clas/Documents/python/Learning to trade with direct RL/csvs/2017-08-01_2024-01-16_BTC-USDT_1h.csv'
df =pd.read_csv(path)
#%%

class TradingModel(nn.Module):
    def __init__(self, lookback=100, fees=0.00025, use_mlp=False):

        super(TradingModel, self).__init__()
        self.lookback = lookback
        self.fees = fees
        self.theta = torch.rand(lookback+2, requires_grad=True, dtype=torch.float32) # +1 for bias, +1 for previous position
        self.P = torch.zeros(1, dtype=torch.float32)  # Initialize with ones = start out long
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

    def positions(self, features):
        len_time_series = len(features)
        P_list = [torch.rand(1) for x in range(self.lookback-1)]  # Initialize with ones = start out long. minus one to account for the the first position

        for t in range(self.lookback, len_time_series+1):
            feature_inputs = features[t - self.lookback:t] # i.e. last n returns

            feature_inputs_norm = (feature_inputs - feature_inputs.mean()) / feature_inputs.std()

            bias = torch.tensor([1.0], dtype=torch.float32) 
            
            n_last_positions = torch.cat(P_list)

            P_list_norm = ((n_last_positions - n_last_positions.mean()) / n_last_positions.std()).flatten()

            last_position = P_list_norm[-1].unsqueeze(0)
            
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

    def returns(self, P, asset_returns):
        if not (len(asset_returns) == len(P)):
            raise ValueError("The lengths of returns, and positions must be the same.")

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

            self.train_model(asset_returns[train_test_index], features[train_test_index], epochs, learning_rate, optimization_target)
            # run on train and test set to avoid the "warm up"/lookback period of the model.
            P = self.positions(features[test_index]) 
            # just take the test period
            positions_test = P[-len_test:]
            self.walk_forward_positions.append(positions_test)

        
        self.walk_forward_positions = torch.cat(self.walk_forward_positions)
        len_mising = len(asset_returns)-len(self.walk_forward_positions) # missing due to the first train set

        self.walk_forward_positions = torch.cat((torch.ones(len_mising, dtype=torch.float32), self.walk_forward_positions))

        assert len(self.walk_forward_positions) == len(asset_returns), "The length of the walk forward positions should be the same as the asset returns"

    def plot(self, asset_returns, use_matplotlib=False):
        fifty_fifty_positions = np.array([0.5 for x in range(len(asset_returns))])
        fifty_fifty_positions = torch.from_numpy(fifty_fifty_positions)
        fifty_fifty_returns = self.returns(fifty_fifty_positions, asset_returns).detach().numpy().cumprod()
        P = self.walk_forward_positions
        trained_bot_returns = self.returns(P, asset_returns).detach().numpy().cumprod()
        asset_returns = asset_returns.detach().numpy().cumprod()
        x_range = list(range(len(fifty_fifty_returns)))
        data = {
            'Fifty-Fifty Bot Returns': fifty_fifty_returns,
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
            fig_combined = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01)
            
            for variable in df.columns:
                fig_combined.add_trace(
                    go.Scatter(x=df.index,y=df[variable],name=variable,mode='lines'),
                    row=1,col=1)

            fig_combined.add_trace(
                go.Scatter(x=x_range,y=P.detach().numpy(),name='P',mode='lines'), 
                row=2,col=1)
                        
            fig_combined.update_layout(title='Combined Returns Over Time')
            fig_combined.update_yaxes(type="log", row=1)
            fig_combined.update_yaxes(title_text="P", row=2)
            
            fig_combined.show()

#%%
asset_returns = torch.from_numpy(df['asset_return'].values).type(torch.float32)
features = torch.from_numpy(df['asset_return'].values).type(torch.float32)
tm = TradingModel(use_mlp=False,lookback=10,fees=0.00025)
#%%
tm.walk_forward(asset_returns, 
                features, 
                epochs = 1, 
                learning_rate = 0.1, 
                optimization_target = "mean_return", 
                num_splits = 4, 
                max_train_size = 2000)
#%%                
tm.plot(asset_returns, use_matplotlib=True)
