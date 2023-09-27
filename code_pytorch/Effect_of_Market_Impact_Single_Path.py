import datetime as dt
import math
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import Utils_general
import DeepAgent as DeepAgent
import DeepAgentTransformer
import DeepAgentLSTM as DeepAgentLSTM
import DeepAgentGRU as DeepAgentGRU
from scipy.stats import ttest_ind

nbs_point_traj = 13
batch_size = 256
train_size = 100000
test_size = 100000
epochs = 100
r_borrow = 0
r_lend = 0
stock_dyn = "BSM" 
params_vect = [0.1, 0.1898]
S_0 = 1000
T = 252/252
alpha = 1.01
beta = 0.99
loss_type = "RSMSE"
option_type = "call"
position_type = "short"
strike = 1000
nbs_layers = 4
nbs_units = 256
lr = 0.0001
prepro_stock = "log-moneyness"
nbs_shares = 1
lambdas = [-1, -1]

Ts = [252/252, 30/252, 1/252]
for T in Ts:
    if T == 252/252:
        time_period = "year"
        nbs_point_traj = 13
    elif T == 30/252:
        time_period = "month"
        nbs_point_traj = 31
    else:
        time_period = "day"
        nbs_point_traj = 9

    if (option_type == 'Call'):
        V_0 = Utils_general.BlackScholes_price(S_0, T, r_borrow, params_vect[1], strike, 1)
    else:
        V_0 = Utils_general.BlackScholes_price(S_0, T, r_borrow, params_vect[1], strike, -1)

    # Creating test dataset
    mu, sigma = params_vect
    N = nbs_point_traj - 1
    dt = T / N
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_set = S_0 * torch.ones(int(test_size/batch_size), nbs_point_traj, batch_size, device=device)
    for i in range(int(test_size/batch_size)):
        S_t = S_0 * torch.ones(batch_size, device=device)
        for j in range(N):
            Z = torch.randn(batch_size, device=device)
            S_t = S_t * torch.exp((mu - sigma ** 2 / 2) * dt + sigma * math.sqrt(dt) * Z)
            test_set[i, j+1, :] = S_t

    loss_type = "RSMSE"
    impact_values = [0.0, 0.0018, 0.025]
    global_hedge_deltas = []
    for i, impact in enumerate(impact_values):
        alpha = 1.0+impact
        beta = 1.0-impact

        name = "code_pytorch/effect_of_market_impact_single_path/{loss_type}_hedging_{impact:.4f}_market_impact_single_path_{time_period}".format(loss_type="quadratic" if loss_type=="RMSE" else "semi_quadratic", impact=impact, time_period=time_period)
        agent = DeepAgent.DeepAgent(nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect, S_0, T, alpha, beta,
                        loss_type, option_type, position_type, strike, V_0, nbs_layers, nbs_units, lr, prepro_stock,
                        nbs_shares, lambdas, name=name)

        print("START FFNN {loss_type} - {impact:.4f} IMPACT".format(loss_type=loss_type, impact=impact))
        all_losses, epoch_losses = agent.train(train_size = train_size, epochs=epochs)

        print("DONE FFNN {loss_type} - {impact:.4f} IMPACT".format(loss_type=loss_type, impact=impact))
        agent.model = torch.load("/home/a_eagu/Deep-Hedging-with-Market-Impact/" + name)
        deltas, hedging_err, S_t, V_t, A_t, B_t, = agent.test(test_size=test_size, test_set=test_set)
        global_hedge_deltas.append(deltas[:,0:1,0])
        semi_square_hedging_err = np.square(np.where(hedging_err > 0, hedging_err, 0))
        rsmse = np.sqrt(np.mean(semi_square_hedging_err))

        print(" ----------------- ")
        print(" Deep Hedging {loss_type} FFNN Results".format(loss_type=loss_type))
        print(" ----------------- ")
        Utils_general.print_stats(hedging_err, deltas, loss_type, "Deep hedge - FFNN - {}".format(loss_type), V_0)

        def count_parameters(agent):
            return sum(p.numel() for p in agent.model.parameters() if p.requires_grad)

    print("FFNN PARAMETERS: ", count_parameters(agent))

    deltas_DH, hedging_err_DH = Utils_general.delta_hedge_res(S_t[:,0:1], r_borrow, r_lend, params_vect[1], T, alpha, beta, option_type="Call", position_type="Short", strike=strike, V_0=V_0, nbs_shares=nbs_shares, hab=lambdas)

    fig = plt.figure(figsize=(10, 5))
    plt.plot(S_t[:,0:1], label="Stock prices path")
    plt.xlabel('Time Steps')
    plt.ylabel('S_t (stock price)')
    plt.legend()
    plt.title("Path of stock prices - Time Period: {time_period} - mu = {mu:.4f} and sigma = {sigma:.4f}".format(time_period=time_period, mu=params_vect[0], sigma=params_vect[1]))
    plt.savefig("code_pytorch/effect_of_market_impact_single_path/path_of_stock_prices_effect_of_market_impact_{}".format(time_period))

    fig = plt.figure(figsize=(10, 5))
    plt.plot(deltas_DH, label="Delta hedge")
    for i, global_deltas in enumerate(global_hedge_deltas):
        plt.plot(global_deltas, label="Global {loss} - {impact}".format(loss=loss_type, impact="no impact" if alpha==1.0 and beta==1.0 else "alpha = {:.4f}, beta = {:.4f}".format(1.0-impact_values[i], 1.0+impact_values[i])))
    plt.xlabel('Time Steps')
    plt.ylabel('Share of stock (delta_{t+1})')
    plt.grid()
    plt.legend()
    plt.title("Path of deltas - Time Period: {time_period} - only RSMSE - shares = 1".format(time_period=time_period))
    plt.savefig("code_pytorch/effect_of_market_impact_single_path/path_of_deltas_effect_of_market_impact_{}".format(time_period))

    diff_deltas_DH = np.ones((deltas_DH.shape[0]-1, deltas_DH.shape[1]))
    for i in range(deltas_DH.shape[0]-1):
        diff_deltas_DH[i] = deltas_DH[i+1] - deltas_DH[i]

    diff_global_deltas = []
    for i in range(len(global_hedge_deltas)):
        diff_global_deltas.append(np.ones((global_hedge_deltas[i].shape[0]-1, global_hedge_deltas[i].shape[1])))
    for i in range(len(global_hedge_deltas)):
        for j in range(global_hedge_deltas[i].shape[0]-1):
            diff_global_deltas[i][j] = global_hedge_deltas[i][j+1] - global_hedge_deltas[i][j] 

    fig = plt.figure(figsize=(10, 5))
    plt.plot(diff_deltas_DH, label="Delta hedge")
    for i, diff_deltas in enumerate(diff_global_deltas):
        plt.plot(diff_deltas, label="Global {loss} - {impact}".format(loss=loss_type, impact="no impact" if alpha==1.0 and beta==1.0 else "alpha = {:.4f}, beta = {:.4f}".format(1.0-impact_values[i], 1.0+impact_values[i])))
    plt.xlabel('Time Steps')
    plt.ylabel('Difference share of stock (diff delta_{t+1})')
    plt.grid()
    plt.legend()
    plt.title("Path of diff deltas - Time Period: {time_period} - only RSMSE - shares = 1".format(time_period=time_period))
    plt.savefig("code_pytorch/effect_of_market_impact_single_path/path_of_diff_deltas_effect_of_market_impact_{}".format(time_period))

    print()