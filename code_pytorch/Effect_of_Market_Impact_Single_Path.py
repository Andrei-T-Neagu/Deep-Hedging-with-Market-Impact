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

nbs_point_traj = 31
batch_size = 256
train_size = 100000
test_size = 100000
epochs = 10
r_borrow = 0
r_lend = 0
stock_dyn = "BSM" 
params_vect = [0.1, 0.1898]
S_0 = 1000
T = 30/252
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

Ts = [1/252]
#Ts = [252/252, 30/252, 1/252]
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
    impact_values = [0.002, 0.001, 0.0]
    impact_persistence = [0, -math.log(0.75)/dt, -math.log(0.5)/dt, -math.log(0.25)/dt, -1] # impact persistence of {1, 0.75, 0.5, 0.25, 0}

    deltas_results = np.zeros((len(impact_persistence), len(impact_values), N))
    S_t_bids = np.zeros((len(impact_persistence), len(impact_values), nbs_point_traj))
    diff_deltas_results = np.zeros((len(impact_persistence), len(impact_persistence), N))

    for p, persistence in enumerate(impact_persistence):
        lambdas = [persistence, persistence]
        for i, impact in enumerate(impact_values):
            alpha = 1.0+impact
            beta = 1.0-impact
            
            name = "code_pytorch/effect_of_market_impact_single_path/{time_period_folder}/{loss_type}_hedging_{impact:.4f}_market_impact_single_path_{time_period}_{persistence:.1f}_persistence".format(time_period_folder=time_period, loss_type="quadratic" if loss_type=="RMSE" else "semi_quadratic", impact=impact, time_period=time_period, persistence=persistence)
            agent = DeepAgent.DeepAgent(nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect, S_0, T, alpha, beta,
                            loss_type, option_type, position_type, strike, V_0, nbs_layers, nbs_units, lr, prepro_stock,
                            nbs_shares, lambdas, name=name)

            print("START FFNN {loss_type} - {impact:.4f} IMPACT - {persistence:.1f} Persistence".format(loss_type=loss_type, impact=impact, persistence=persistence))
            all_losses, epoch_losses = agent.train(train_size = train_size, epochs=epochs)

            print("DONE FFNN {loss_type} - {impact:.4f} IMPACT - {persistence:.1f} Persistence".format(loss_type=loss_type, impact=impact, persistence=persistence))
            agent.model = torch.load("/home/a_eagu/Deep-Hedging-with-Market-Impact/{name}".format(name=name))
            deltas, hedging_err, S_t, V_t, A_t, B_t, = agent.test(test_size=test_size, test_set=test_set)
            if p == 0 and i == 0:
                max_error_index = np.argmax(hedging_err)
            
            deltas_results[p, i] = deltas[:,max_error_index,0]
            for k in range(N):
                if k == 0:
                    diff_deltas_results[p, i, k] = deltas_results[p, i, k]
                else:
                    diff_deltas_results[p, i, k] = deltas_results[p, i, k].item() - deltas_results[p, i, k-1].item()
            
            # sells = np.where(diff_deltas_results < 0, -diff_deltas_results, 0)
            # sells = np.concatenate((sells[:, :, :], np.ones((sells.shape[0], sells.shape[1], 1))), axis=2)

            # print("deltas.shape: ", deltas.shape)
            # print("deltas_results[p, i].shape: ", deltas_results[p, i].shape)
            # print("deltas_results[p, i]: ", deltas_results[p, i])
            # print("sells.shape: ", sells.shape)
            # print("sells[p, i]: ", sells[p, i])
            # print("diff_deltas_results[p, i].shape: ", diff_deltas_results[p, i].shape)
            # print("diff_deltas_results[p, i]: ", diff_deltas_results[p, i])
            # print("B_t[:, max_error_index].shape: ", B_t[:, max_error_index].shape)
            # print("B_t[:, max_error_index]: ", B_t[:, max_error_index])

            S_t_bids[p, i] = S_t[:, max_error_index] * ((1 + 1 + B_t[:, max_error_index]) ** beta - (1 + B_t[:, max_error_index]) ** beta)

            # print("S_t_bids[p, i].shape: ", S_t_bids[p, i].shape)
            # print("S_t_bids[p, i]: ", S_t_bids[p, i])

            semi_square_hedging_err = np.square(np.where(hedging_err > 0, hedging_err, 0))
            rsmse = np.sqrt(np.mean(semi_square_hedging_err))

            print(" ----------------- ")
            print(" Deep Hedging {loss_type} FFNN Results".format(loss_type=loss_type))
            print(" ----------------- ")
            Utils_general.print_stats(hedging_err, deltas, loss_type, "Deep hedge - FFNN - {}".format(loss_type), V_0)

            def count_parameters(agent):
                return sum(p.numel() for p in agent.model.parameters() if p.requires_grad)

        deltas_DH, hedging_err_DH = Utils_general.delta_hedge_res(S_t[:,max_error_index:max_error_index+1], r_borrow, r_lend, params_vect[1], T, alpha, beta, option_type="Call", position_type="Short", strike=strike, V_0=V_0, nbs_shares=nbs_shares, hab=lambdas)

        fig = plt.figure(figsize=(10, 5))
        plt.plot(S_t[:,max_error_index], label="Stock prices path")
        for j in range(len(impact_values)):
            plt.plot(S_t_bids[p, j, :], label="Bid prices path with alpha = {:.4f}, beta = {:.4f}".format(1.0-impact_values[j], 1.0+impact_values[j]))
        plt.xlabel('Time Steps')
        plt.ylabel('S_t (stock price)')
        plt.grid()
        plt.legend()
        plt.title("Path of stock prices - Time Period: {time_period} - mu = {mu:.4f} and sigma = {sigma:.4f} - lambdas = {persistence:.1f}".format(time_period=time_period, mu=params_vect[0], sigma=params_vect[1], persistence=persistence))
        plt.savefig("code_pytorch/effect_of_market_impact_single_path/{time_period_folder}/same_persistence/path_of_stock_prices_effect_of_market_impact_{time_period}_persistence_{persistence:.1f}.jpg".format(time_period_folder=time_period, time_period=time_period, persistence=persistence))

        fig = plt.figure(figsize=(10, 5))
        plt.plot(deltas_DH, label="Delta hedge")
        for j in range(len(impact_values)):
            plt.plot(deltas_results[p, j, :], label="Global {loss} - {impact}".format(loss=loss_type, impact="no impact" if impact_values[j] == 0.0 and impact_values[j] == 0.0 else "alpha = {:.4f}, beta = {:.4f}".format(1.0-impact_values[j], 1.0+impact_values[j])))
        plt.xlabel('Time Steps')
        plt.ylabel('Share of stock (delta_{t+1})')
        plt.grid()
        plt.legend()
        plt.title("Path of deltas - Time Period: {time_period} - only RSMSE - shares = 1 - lambdas = {persistence:.1f}".format(time_period=time_period, persistence=persistence))
        plt.savefig("code_pytorch/effect_of_market_impact_single_path/{time_period_folder}/same_persistence/path_of_deltas_effect_of_market_impact_{time_period}_persistence_{persistence:.1f}.jpg".format(time_period_folder=time_period, time_period=time_period, persistence=persistence))

        diff_deltas_DH = np.ones((deltas_DH.shape[0]-1, deltas_DH.shape[1]))
        for j in range(deltas_DH.shape[0]-1):
            diff_deltas_DH[j] = deltas_DH[j+1] - deltas_DH[j]

        fig = plt.figure(figsize=(10, 5))
        plt.plot(diff_deltas_DH, label="Delta hedge")
        for j in range(len(impact_values)):
            plt.plot(diff_deltas_results[p, j], label="Global {loss} - {impact}".format(loss=loss_type, impact="no impact" if impact_values[j] == 0.0 and impact_values[j] == 0.0 else "alpha = {:.4f}, beta = {:.4f}".format(1.0-impact_values[j], 1.0+impact_values[j])))
        plt.xlabel('Time Steps')
        plt.ylabel('Difference share of stock (diff delta_{t+1})')
        plt.grid()
        plt.legend()
        plt.title("Path of diff deltas - Time Period: {time_period} - only RSMSE - shares = 1 - lambdas = {persistence:.1f}".format(time_period=time_period, persistence=persistence))
        plt.savefig("code_pytorch/effect_of_market_impact_single_path/{time_period_folder}/same_persistence/path_of_diff_deltas_effect_of_market_impact_{time_period}_persistence_{persistence:.1f}.jpg".format(time_period_folder=time_period, time_period=time_period, persistence=persistence))

        print()

    for i, impact in enumerate(impact_values):
        alpha = 1.0+impact
        beta = 1.0-impact
        fig = plt.figure(figsize=(10, 5))
        plt.plot(S_t[:,max_error_index], label="Stock prices path")
        for p, persistence in enumerate(impact_persistence):
            lambdas = [persistence, persistence]
            plt.plot(S_t_bids[p, i, :], label="Bid prices path with lambdas = {persistence:.1f}".format(persistence=persistence))
        plt.xlabel('Time Steps')
        plt.ylabel('S_t (stock price)')
        plt.grid()
        plt.legend()
        plt.title("Path of stock prices - Time Period: {time_period} - mu = {mu:.4f} and sigma = {sigma:.4f} - alpha = {alpha:.4f}, beta = {beta:.4f}".format(time_period=time_period, mu=params_vect[0], sigma=params_vect[1], alpha=alpha, beta=beta))
        plt.savefig("code_pytorch/effect_of_market_impact_single_path/{time_period_folder}/same_impact/path_of_stock_prices_effect_of_market_impact_{time_period}_alpha_{alpha:.4f}_beta_{beta:.4f}.jpg".format(time_period_folder=time_period, time_period=time_period, alpha=alpha, beta=beta))
    
    for i, impact in enumerate(impact_values):
        alpha = 1.0+impact
        beta = 1.0-impact
        fig = plt.figure(figsize=(10, 5))
        plt.plot(deltas_DH, label="Delta hedge")
        for p, persistence in enumerate(impact_persistence):
            plt.plot(deltas_results[p, i, :], label="Global {loss} - lambdas = {persistence:.1f}".format(loss=loss_type, persistence=persistence))
        plt.xlabel('Time Steps')
        plt.ylabel('Share of stock (delta_{t+1})')
        plt.grid()
        plt.legend()
        plt.title("Path of deltas - Time Period: {time_period} - only RSMSE - shares = 1 - alpha = {alpha:.4f}, beta = {beta:.4f}".format(time_period=time_period, alpha=alpha, beta=beta))
        plt.savefig("code_pytorch/effect_of_market_impact_single_path/{time_period_folder}/same_impact/path_of_deltas_effect_of_market_impact_{time_period}_alpha_{alpha:.4f}_beta_{beta:.4f}.jpg".format(time_period_folder=time_period, time_period=time_period, alpha=alpha, beta=beta))

    for i, impact in enumerate(impact_values):
        alpha = 1.0+impact
        beta = 1.0-impact
        fig = plt.figure(figsize=(10, 5))
        plt.plot(diff_deltas_DH, label="Delta hedge")
        for p, persistence in enumerate(impact_persistence):
            plt.plot(diff_deltas_results[p, i], label="Global {loss} - lambdas = {persistence:.1f}".format(loss=loss_type, persistence=persistence))
        plt.xlabel('Time Steps')
        plt.ylabel('Difference share of stock (diff delta_{t+1})')
        plt.grid()
        plt.legend()
        plt.title("Path of diff deltas - Time Period: {time_period} - only RSMSE - shares = 1 - alpha = {alpha:.4f}, beta = {beta:.4f}".format(time_period=time_period, alpha=alpha, beta=beta))
        plt.savefig("code_pytorch/effect_of_market_impact_single_path/{time_period_folder}/same_impact/path_of_diff_deltas_effect_of_market_impact_{time_period}_alpha_{alpha:.4f}_beta_{beta:.4f}.jpg".format(time_period_folder=time_period, time_period=time_period, alpha=alpha, beta=beta))

    
