import datetime as dt
import math
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import Utils_general
import DeepAgent
import DeepAgentTransformer
import DeepAgentLSTM
import DeepAgentGRU
from scipy.stats import ttest_ind

nbs_point_traj = 9
batch_size = 256
train_size = 100000
test_size = 100000
epochs = 10
r_borrow = 0
r_lend = 0
stock_dyn = "BSM" 
params_vect = [0.1, 0.1898]
S_0 = 1000
T = 1/252
alpha = 1.05
beta = 0.95
loss_type = "RSMSE"
option_type = "call"
position_type = "short"
strike = 1000
nbs_layers = 5
nbs_units = 256
lr = 0.0001
prepro_stock = "log-moneyness"
nbs_shares = 1
lambdas = [1, 1]

name_ffnn = 'ffnn_model'
name_lstm = 'lstm_model'
name_gru = 'gru_model'
name_transformer = 'transformer_model'

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
        test_set[i, j, :] = S_t



agent_trans = DeepAgentTransformer.DeepAgent(nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect, S_0, T, alpha, beta,
                 loss_type, option_type, position_type, strike, V_0, nbs_layers, nbs_units, lr, prepro_stock,
                 nbs_shares, lambdas, name=name_transformer)

print("START TRANSFORMER")
all_losses_trans, trans_losses = agent_trans.train(train_size = train_size, epochs=epochs)
print("DONE TRANSFORMER")
agent_trans.model = torch.load("/home/a_eagu/Deep-Hedging-with-Market-Impact/" + name_transformer)
deltas_trans, hedging_err_trans, S_t_trans, V_t_trans, A_t_trans, B_t_trans, = agent_trans.test(test_size=test_size, test_set=test_set)
semi_square_hedging_err_trans = np.square(np.where(hedging_err_trans > 0, hedging_err_trans, 0))
rsmse_trans = np.sqrt(np.mean(semi_square_hedging_err_trans))

agent_lstm = DeepAgentLSTM.DeepAgent(nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect, S_0, T, alpha, beta,
                 loss_type, option_type, position_type, strike, V_0, nbs_layers, nbs_units, lr, prepro_stock,
                 nbs_shares, lambdas, name=name_lstm)

print("START LSTM")
all_losses_lstm, lstm_losses = agent_lstm.train(train_size = train_size, epochs=epochs)
print("DONE LSTM")
agent_lstm.model = torch.load("/home/a_eagu/Deep-Hedging-with-Market-Impact/" + name_lstm)
deltas_lstm, hedging_err_lstm, S_t_lstm, V_t_lstm, A_t_lstm, B_t_lstm, = agent_lstm.test(test_size=test_size, test_set=test_set)
semi_square_hedging_err_lstm = np.square(np.where(hedging_err_lstm > 0, hedging_err_lstm, 0))
rsmse_lstm = np.sqrt(np.mean(semi_square_hedging_err_lstm))

agent_gru = DeepAgentGRU.DeepAgent(nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect, S_0, T, alpha, beta,
                 loss_type, option_type, position_type, strike, V_0, nbs_layers, nbs_units, lr, prepro_stock,
                 nbs_shares, lambdas, name=name_gru)

print("START GRU")
all_losses_gru, gru_losses = agent_gru.train(train_size = train_size, epochs=epochs)
print("DONE GRU")
agent_gru.model = torch.load("/home/a_eagu/Deep-Hedging-with-Market-Impact/" + name_gru)
deltas_gru, hedging_err_gru, S_t_gru, V_t_gru, A_t_gru, B_t_gru, = agent_gru.test(test_size=test_size, test_set=test_set)
semi_square_hedging_err_gru = np.square(np.where(hedging_err_gru > 0, hedging_err_gru, 0))
rsmse_gru = np.sqrt(np.mean(semi_square_hedging_err_gru))

agent = DeepAgent.DeepAgent(nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect, S_0, T, alpha, beta,
                 loss_type, option_type, position_type, strike, V_0, nbs_layers, nbs_units, lr, prepro_stock,
                 nbs_shares, lambdas, name=name_ffnn)

print("START FFNN")
all_losses_ffnn, ffnn_losses = agent.train(train_size = train_size, epochs=epochs)
print("DONE FFNN")
agent.model = torch.load("/home/a_eagu/Deep-Hedging-with-Market-Impact/" + name_ffnn)
deltas_ffnn, hedging_err_ffnn, S_t_ffnn, V_t_ffnn, A_t_ffnn, B_t_ffnn, = agent.test(test_size=test_size, test_set=test_set)
semi_square_hedging_err_ffnn = np.square(np.where(hedging_err_ffnn > 0, hedging_err_ffnn, 0))
rsmse_ffnn = np.sqrt(np.mean(semi_square_hedging_err_ffnn))

print(" ----------------- ")
print(" Deep Hedging %s TRANSFORMER Results" % (loss_type))
print(" ----------------- ")
Utils_general.print_stats(hedging_err_trans, deltas_trans, loss_type, "Deep hedge - TRANSFORMER - %s" % (loss_type), V_0)

print(" ----------------- ")
print(" Deep Hedging %s GRU Results" % (loss_type))
print(" ----------------- ")
Utils_general.print_stats(hedging_err_gru, deltas_gru, loss_type, "Deep hedge - GRU - %s" % (loss_type), V_0)

print(" ----------------- ")
print(" Deep Hedging %s LSTM Results" % (loss_type))
print(" ----------------- ")
Utils_general.print_stats(hedging_err_lstm, deltas_lstm, loss_type, "Deep hedge - LSTM - %s" % (loss_type), V_0)

print(" ----------------- ")
print(" Deep Hedging %s FFNN Results" % (loss_type))
print(" ----------------- ")
Utils_general.print_stats(hedging_err_ffnn, deltas_ffnn, loss_type, "Deep hedge - FFNN - %s" % (loss_type), V_0)

print(" ----------------- ")
print(" Delta Hedging Results")
print(" ----------------- ")
deltas_DH, hedging_err_DH = Utils_general.delta_hedge_res(S_t_ffnn, r_borrow, r_lend, params_vect[1], T, alpha, beta, option_type="Call", position_type="Short", strike=strike, V_0=V_0, nbs_shares=nbs_shares, hab=lambdas)
Utils_general.print_stats(hedging_err_DH, deltas_DH, "Delta hedge", "Delta hedge", V_0)
semi_square_hedging_err_DH = np.square(np.where(hedging_err_DH > 0, hedging_err_DH, 0))
rsmse_DH = np.sqrt(np.mean(semi_square_hedging_err_DH))

print()

print("|--------------------------------------Comparison of RSMSE--------------------------------------|")
print("|\tTransformer\t|\tGRU\t|\tLSTM\t|\tFFNN\t|\tDelta Hedge\t|")
print("|-----------------------|---------------|---------------|---------------|-----------------------|")
print("|\t{:.4f}\t\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t\t|".format(rsmse_trans, rsmse_gru, rsmse_lstm, rsmse_ffnn, rsmse_DH))
print("|-----------------------|---------------|---------------|---------------|-----------------------|")

with open("Comparison_RSMSE_" + str(nbs_point_traj) + ".txt", "w") as rsmse_file:
    # Writing data to a file
    rsmse_file.write("|--------------------------------------Comparison of RSMSE--------------------------------------|\n")
    rsmse_file.write("|\t\tTransformer\t\t|\t\tGRU\t\t|\t\tLSTM\t|\t\tFFNN\t|\t\tDelta Hedge\t\t|\n")
    rsmse_file.write("|-----------------------|---------------|---------------|---------------|-----------------------|\n")
    rsmse_file.write("|\t\t{:.4f}\t\t\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t\t\t|\n".format(rsmse_trans, rsmse_gru, rsmse_lstm, rsmse_ffnn, rsmse_DH))
    rsmse_file.write("|-----------------------|---------------|---------------|---------------|-----------------------|\n")

print()

smse_trans_trans = ttest_ind(semi_square_hedging_err_trans, semi_square_hedging_err_trans, equal_var=False, alternative="less").pvalue
smse_trans_gru = ttest_ind(semi_square_hedging_err_trans, semi_square_hedging_err_gru, equal_var=False, alternative="less").pvalue
smse_trans_lstm = ttest_ind(semi_square_hedging_err_trans, semi_square_hedging_err_lstm, equal_var=False, alternative="less").pvalue
smse_trans_ffnn = ttest_ind(semi_square_hedging_err_trans, semi_square_hedging_err_ffnn, equal_var=False, alternative="less").pvalue
smse_trans_DH = ttest_ind(semi_square_hedging_err_trans, semi_square_hedging_err_DH, equal_var=False, alternative="less").pvalue

smse_gru_trans = ttest_ind(semi_square_hedging_err_gru, semi_square_hedging_err_trans, equal_var=False, alternative="less").pvalue
smse_gru_gru = ttest_ind(semi_square_hedging_err_gru, semi_square_hedging_err_gru, equal_var=False, alternative="less").pvalue
smse_gru_lstm = ttest_ind(semi_square_hedging_err_gru, semi_square_hedging_err_lstm, equal_var=False, alternative="less").pvalue
smse_gru_ffnn = ttest_ind(semi_square_hedging_err_gru, semi_square_hedging_err_ffnn, equal_var=False, alternative="less").pvalue
smse_gru_DH = ttest_ind(semi_square_hedging_err_gru, semi_square_hedging_err_DH, equal_var=False, alternative="less").pvalue

smse_lstm_trans = ttest_ind(semi_square_hedging_err_lstm, semi_square_hedging_err_trans, equal_var=False, alternative="less").pvalue
smse_lstm_gru = ttest_ind(semi_square_hedging_err_lstm, semi_square_hedging_err_gru, equal_var=False, alternative="less").pvalue
smse_lstm_lstm = ttest_ind(semi_square_hedging_err_lstm, semi_square_hedging_err_lstm, equal_var=False, alternative="less").pvalue
smse_lstm_ffnn = ttest_ind(semi_square_hedging_err_lstm, semi_square_hedging_err_ffnn, equal_var=False, alternative="less").pvalue
smse_lstm_DH = ttest_ind(semi_square_hedging_err_lstm, semi_square_hedging_err_DH, equal_var=False, alternative="less").pvalue

smse_ffnn_trans = ttest_ind(semi_square_hedging_err_ffnn, semi_square_hedging_err_trans, equal_var=False, alternative="less").pvalue
smse_ffnn_gru = ttest_ind(semi_square_hedging_err_ffnn, semi_square_hedging_err_gru, equal_var=False, alternative="less").pvalue
smse_ffnn_lstm = ttest_ind(semi_square_hedging_err_ffnn, semi_square_hedging_err_lstm, equal_var=False, alternative="less").pvalue
smse_ffnn_ffnn = ttest_ind(semi_square_hedging_err_ffnn, semi_square_hedging_err_ffnn, equal_var=False, alternative="less").pvalue
smse_ffnn_DH = ttest_ind(semi_square_hedging_err_ffnn, semi_square_hedging_err_DH, equal_var=False, alternative="less").pvalue

smse_DH_trans = ttest_ind(semi_square_hedging_err_DH, semi_square_hedging_err_trans, equal_var=False, alternative="less").pvalue
smse_DH_gru = ttest_ind(semi_square_hedging_err_DH, semi_square_hedging_err_gru, equal_var=False, alternative="less").pvalue
smse_DH_lstm = ttest_ind(semi_square_hedging_err_DH, semi_square_hedging_err_lstm, equal_var=False, alternative="less").pvalue
smse_DH_ffnn = ttest_ind(semi_square_hedging_err_DH, semi_square_hedging_err_ffnn, equal_var=False, alternative="less").pvalue
smse_DH_DH = ttest_ind(semi_square_hedging_err_DH, semi_square_hedging_err_DH, equal_var=False, alternative="less").pvalue

print("|------------------------------------------------T-Test for Smaller SMSE------------------------------------------------|")
print("|\t\t\t|\tTransformer\t|\tGRU\t|\tLSTM\t|\tFFNN\t|\tDelta Hedge\t|")
print("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|")
print("|\tTransformer\t|\t{:.4f}\t\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t\t|".format(smse_trans_trans, smse_trans_gru, smse_trans_lstm, smse_trans_ffnn, smse_trans_DH))
print("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|")
print("|\tGRU\t\t|\t{:.4f}\t\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t\t|".format(smse_gru_trans, smse_gru_gru, smse_gru_lstm, smse_gru_ffnn, smse_gru_DH))
print("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|")
print("|\tLSTM\t\t|\t{:.4f}\t\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t\t|".format(smse_lstm_trans, smse_lstm_gru, smse_lstm_lstm, smse_lstm_ffnn, smse_lstm_DH))
print("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|")
print("|\tFFNN\t\t|\t{:.4f}\t\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t\t|".format(smse_ffnn_trans, smse_ffnn_gru, smse_ffnn_lstm, smse_ffnn_ffnn, smse_ffnn_DH))
print("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|")
print("|\tDelta Hedge\t|\t{:.4f}\t\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t\t|".format(smse_DH_trans, smse_DH_gru, smse_DH_lstm, smse_DH_ffnn, smse_DH_DH))
print("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|")

with open("SMSE_T_TEST_" + str(nbs_point_traj) + ".txt", "w") as smse_file:
    # Writing data to a file
    smse_file.write("|------------------------------------------------T-Test for Smaller SMSE------------------------------------------------|\n")
    smse_file.write("|\t\t\t\t\t\t|\t\tTransformer\t\t|\t\tGRU\t\t|\t\tLSTM\t|\t\tFFNN\t|\t\tDelta Hedge\t\t|\n")
    smse_file.write("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|\n")
    smse_file.write("|\t\tTransformer\t\t|\t\t{:.4f}\t\t\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t\t\t|\n".format(smse_trans_trans, smse_trans_gru, smse_trans_lstm, smse_trans_ffnn, smse_trans_DH))
    smse_file.write("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|\n")
    smse_file.write("|\t\tGRU\t\t\t\t|\t\t{:.4f}\t\t\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t\t\t|\n".format(smse_gru_trans, smse_gru_gru, smse_gru_lstm, smse_gru_ffnn, smse_gru_DH))
    smse_file.write("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|\n")
    smse_file.write("|\t\tLSTM\t\t\t|\t\t{:.4f}\t\t\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t\t\t|\n".format(smse_lstm_trans, smse_lstm_gru, smse_lstm_lstm, smse_lstm_ffnn, smse_lstm_DH))
    smse_file.write("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|\n")
    smse_file.write("|\t\tFFNN\t\t\t|\t\t{:.4f}\t\t\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t\t\t|\n".format(smse_ffnn_trans, smse_ffnn_gru, smse_ffnn_lstm, smse_ffnn_ffnn, smse_ffnn_DH))
    smse_file.write("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|\n")
    smse_file.write("|\t\tDelta Hedge\t\t|\t\t{:.4f}\t\t\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t\t\t|\n".format(smse_DH_trans, smse_DH_gru, smse_DH_lstm, smse_DH_ffnn, smse_DH_DH))
    smse_file.write("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|\n")

print()

print("|-----------------------------------------------------------------------Comparison of Mean Hedging Error------------------------------------------------------------------------|")
print("|\t\tTransformer\t\t|\t\tGRU\t\t|\t\tLSTM\t\t|\t\tFFNN\t\t|\t\tDelta Hedge\t\t|")
print("|---------------------------------------|-------------------------------|-------------------------------|-------------------------------|---------------------------------------|")
print("|\t{:.4f} +- {:.4f}\t\t|\t{:.4f} +- {:.4f}\t|\t{:.4f} +- {:.4f}\t|\t{:.4f} +- {:.4f}\t|\t{:.4f} +- {:.4f}\t\t|".format(np.mean(hedging_err_trans), np.std(hedging_err_trans, ddof=1), np.mean(hedging_err_gru), np.std(hedging_err_gru, ddof=1), np.mean(hedging_err_lstm), np.std(hedging_err_lstm, ddof=1), np.mean(hedging_err_ffnn), np.std(hedging_err_ffnn, ddof=1), np.mean(hedging_err_DH), np.std(hedging_err_DH, ddof=1)))
print("|---------------------------------------|-------------------------------|-------------------------------|-------------------------------|---------------------------------------|")

with open("mean_hedging_error_" + str(nbs_point_traj) + ".txt", "w") as mean_file:
    # Writing data to a file
    mean_file.write("|-----------------------------------------------------------------------Comparison of Mean Hedging Error------------------------------------------------------------------------|\n")
    mean_file.write("|\t\t\t\tTransformer\t\t\t\t|\t\t\tGRU\t\t\t\t\t|\t\t\tLSTM\t\t\t\t|\t\t\tFFNN\t\t\t\t|\t\t\t\tDelta Hedge\t\t\t\t|\n")
    mean_file.write("|---------------------------------------|-------------------------------|-------------------------------|-------------------------------|---------------------------------------|\n")
    mean_file.write("|\t\t\t{:.4f} +- {:.4f}\t\t\t|\t\t{:.4f} +- {:.4f}\t\t|\t\t{:.4f} +- {:.4f}\t\t|\t\t{:.4f} +- {:.4f}\t\t|\t\t\t{:.4f} +- {:.4f}\t\t\t|\n".format(np.mean(hedging_err_trans), np.std(hedging_err_trans, ddof=1), np.mean(hedging_err_gru), np.std(hedging_err_gru, ddof=1), np.mean(hedging_err_lstm), np.std(hedging_err_lstm, ddof=1), np.mean(hedging_err_ffnn), np.std(hedging_err_ffnn, ddof=1), np.mean(hedging_err_DH), np.std(hedging_err_DH, ddof=1)))
    mean_file.write("|---------------------------------------|-------------------------------|-------------------------------|-------------------------------|---------------------------------------|\n")

print()

mean_trans_trans = ttest_ind(hedging_err_trans, hedging_err_trans, equal_var=False, alternative="less").pvalue
mean_trans_gru = ttest_ind(hedging_err_trans, hedging_err_gru, equal_var=False, alternative="less").pvalue
mean_trans_lstm = ttest_ind(hedging_err_trans, hedging_err_lstm, equal_var=False, alternative="less").pvalue
mean_trans_ffnn = ttest_ind(hedging_err_trans, hedging_err_ffnn, equal_var=False, alternative="less").pvalue
mean_trans_DH = ttest_ind(hedging_err_trans, hedging_err_DH, equal_var=False, alternative="less").pvalue

mean_gru_trans = ttest_ind(hedging_err_gru, hedging_err_trans, equal_var=False, alternative="less").pvalue
mean_gru_gru = ttest_ind(hedging_err_gru, hedging_err_gru, equal_var=False, alternative="less").pvalue
mean_gru_lstm = ttest_ind(hedging_err_gru, hedging_err_lstm, equal_var=False, alternative="less").pvalue
mean_gru_ffnn = ttest_ind(hedging_err_gru, hedging_err_ffnn, equal_var=False, alternative="less").pvalue
mean_gru_DH = ttest_ind(hedging_err_gru, hedging_err_DH, equal_var=False, alternative="less").pvalue

mean_lstm_trans = ttest_ind(hedging_err_lstm, hedging_err_trans, equal_var=False, alternative="less").pvalue
mean_lstm_gru = ttest_ind(hedging_err_lstm, hedging_err_gru, equal_var=False, alternative="less").pvalue
mean_lstm_lstm = ttest_ind(hedging_err_lstm, hedging_err_lstm, equal_var=False, alternative="less").pvalue
mean_lstm_ffnn = ttest_ind(hedging_err_lstm, hedging_err_ffnn, equal_var=False, alternative="less").pvalue
mean_lstm_DH = ttest_ind(hedging_err_lstm, hedging_err_DH, equal_var=False, alternative="less").pvalue

mean_ffnn_trans = ttest_ind(hedging_err_ffnn, hedging_err_trans, equal_var=False, alternative="less").pvalue
mean_ffnn_gru = ttest_ind(hedging_err_ffnn, hedging_err_gru, equal_var=False, alternative="less").pvalue
mean_ffnn_lstm = ttest_ind(hedging_err_ffnn, hedging_err_lstm, equal_var=False, alternative="less").pvalue
mean_ffnn_ffnn = ttest_ind(hedging_err_ffnn, hedging_err_ffnn, equal_var=False, alternative="less").pvalue
mean_ffnn_DH = ttest_ind(hedging_err_ffnn, hedging_err_DH, equal_var=False, alternative="less").pvalue

mean_DH_trans = ttest_ind(hedging_err_DH, hedging_err_trans, equal_var=False, alternative="less").pvalue
mean_DH_gru = ttest_ind(hedging_err_DH, hedging_err_gru, equal_var=False, alternative="less").pvalue
mean_DH_lstm = ttest_ind(hedging_err_DH, hedging_err_lstm, equal_var=False, alternative="less").pvalue
mean_DH_ffnn = ttest_ind(hedging_err_DH, hedging_err_ffnn, equal_var=False, alternative="less").pvalue
mean_DH_DH = ttest_ind(hedging_err_DH, hedging_err_DH, equal_var=False, alternative="less").pvalue

print("|-----------------------------------------T-Test for Smaller Mean Hedging Error-----------------------------------------|")
print("|\t\t\t|\tTransformer\t|\tGRU\t|\tLSTM\t|\tFFNN\t|\tDelta Hedge\t|")
print("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|")
print("|\tTransformer\t|\t{:.4f}\t\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t\t|".format(mean_trans_trans, mean_trans_gru, mean_trans_lstm, mean_trans_ffnn, mean_trans_DH))
print("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|")
print("|\tGRU\t\t|\t{:.4f}\t\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t\t|".format(mean_gru_trans, mean_gru_gru, mean_gru_lstm, mean_gru_ffnn, mean_gru_DH))
print("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|")
print("|\tLSTM\t\t|\t{:.4f}\t\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t\t|".format(mean_lstm_trans, mean_lstm_gru, mean_lstm_lstm, mean_lstm_ffnn, mean_lstm_DH))
print("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|")
print("|\tFFNN\t\t|\t{:.4f}\t\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t\t|".format(mean_ffnn_trans, mean_ffnn_gru, mean_ffnn_lstm, mean_ffnn_ffnn, mean_ffnn_DH))
print("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|")
print("|\tDelta Hedge\t|\t{:.4f}\t\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t|\t{:.4f}\t\t|".format(mean_DH_trans, mean_DH_gru, mean_DH_lstm, mean_DH_ffnn, mean_DH_DH))
print("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|")

with open("Mean_T_TEST_" + str(nbs_point_traj) + ".txt", "w") as mean_test_file:
    # Writing data to a file
    mean_test_file.write("|-----------------------------------------T-Test for Smaller Mean Hedging Error-----------------------------------------|\n")
    mean_test_file.write("|\t\t\t\t\t\t|\t\tTransformer\t\t|\t\tGRU\t\t|\t\tLSTM\t|\t\tFFNN\t|\t\tDelta Hedge\t\t|\n")
    mean_test_file.write("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|\n")
    mean_test_file.write("|\t\tTransformer\t\t|\t\t{:.4f}\t\t\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t\t\t|\n".format(mean_trans_trans, mean_trans_gru, mean_trans_lstm, mean_trans_ffnn, mean_trans_DH))
    mean_test_file.write("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|\n")
    mean_test_file.write("|\t\tGRU\t\t\t\t|\t\t{:.4f}\t\t\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t\t\t|\n".format(mean_gru_trans, mean_gru_gru, mean_gru_lstm, mean_gru_ffnn, mean_gru_DH))
    mean_test_file.write("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|\n")
    mean_test_file.write("|\t\tLSTM\t\t\t|\t\t{:.4f}\t\t\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t\t\t|\n".format(mean_lstm_trans, mean_lstm_gru, mean_lstm_lstm, mean_lstm_ffnn, mean_lstm_DH))
    mean_test_file.write("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|\n")
    mean_test_file.write("|\t\tFFNN\t\t\t|\t\t{:.4f}\t\t\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t\t\t|\n".format(mean_ffnn_trans, mean_ffnn_gru, mean_ffnn_lstm, mean_ffnn_ffnn, mean_ffnn_DH))
    mean_test_file.write("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|\n")
    mean_test_file.write("|\t\tDelta Hedge\t\t|\t\t{:.4f}\t\t\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t|\t\t{:.4f}\t\t\t|\n".format(mean_DH_trans, mean_DH_gru, mean_DH_lstm, mean_DH_ffnn, mean_DH_DH))
    mean_test_file.write("|-----------------------|-----------------------|---------------|---------------|---------------|-----------------------|\n")

print()

def count_parameters(agent):
    return sum(p.numel() for p in agent.model.parameters() if p.requires_grad)

print("TRANSFORMER PARAMETERS: ", count_parameters(agent_trans))
print("GRU PARAMETERS: ", count_parameters(agent_gru))
print("LSTM PARAMETERS: ", count_parameters(agent_lstm))
print("FFNN PARAMETERS: ", count_parameters(agent))

all_losses_fig = plt.figure(figsize=(10, 5))
plt.plot(all_losses_lstm, label="LSTM")
plt.plot(all_losses_ffnn, label="FFNN")
plt.plot(all_losses_trans, label="Transformer")
plt.plot(all_losses_gru, label="GRU")
plt.legend()
plt.savefig("all_losses" + str(nbs_point_traj) + ".png")

epoch_losses_fig = plt.figure(figsize=(10, 5))
plt.plot(lstm_losses, label="LSTM")
plt.plot(ffnn_losses, label="FFNN")
plt.plot(trans_losses, label="Transformer")
plt.plot(gru_losses, label="GRU")
plt.legend()
plt.savefig("epoch_losses" + str(nbs_point_traj) + ".png")

log_epoch_losses_fig = plt.figure(figsize=(10, 5))
plt.plot(np.log(lstm_losses), label="LSTM")
plt.plot(np.log(ffnn_losses), label="FFNN")
plt.plot(np.log(trans_losses), label="Transformer")
plt.plot(np.log(gru_losses), label="GRU")
plt.legend()
plt.savefig("log_epoch_losses" + str(nbs_point_traj) + ".png")

fig = plt.figure(figsize=(10, 5))
plt.hist([hedging_err_gru, hedging_err_trans], bins=50, label=["GRU", "Transformer"])
plt.xlabel('Hedging error')
plt.ylabel('Frequency')
plt.legend()
plt.title("Hedging errors for GRU / Transformer - " + str(nbs_point_traj))
plt.savefig("Hedging_Errors_GRU_Transformer" + str(nbs_point_traj) + ".png")

fig = plt.figure(figsize=(10, 5))
plt.hist([hedging_err_gru, hedging_err_lstm], bins=50, label=["GRU", "LSTM"])
plt.xlabel('Hedging error')
plt.ylabel('Frequency')
plt.legend()
plt.title("Hedging errors for GRU / LSTM - " + str(nbs_point_traj))
plt.savefig("Hedging_Errors_GRU_LSTM" + str(nbs_point_traj) + ".png")

fig = plt.figure(figsize=(10, 5))
plt.hist([hedging_err_gru, hedging_err_ffnn], bins=50, label=["GRU", "FFNN"])
plt.xlabel('Hedging error')
plt.ylabel('Frequency')
plt.legend()
plt.title("Hedging errors for GRU / FFNN - " + str(nbs_point_traj))
plt.savefig("Hedging_Errors_GRU_FFNN" + str(nbs_point_traj) + ".png")

fig = plt.figure(figsize=(10, 5))
plt.hist([hedging_err_lstm, hedging_err_trans], bins=50, label=["LSTM", "Transformer"])
plt.xlabel('Hedging error')
plt.ylabel('Frequency')
plt.legend()
plt.title("Hedging errors for LSTM / Transformer - " + str(nbs_point_traj))
plt.savefig("Hedging_Errors_LSTM_Transformer" + str(nbs_point_traj) + ".png")

fig = plt.figure(figsize=(10, 5))
plt.hist([hedging_err_ffnn, hedging_err_trans], bins=50, label=["ffnn", "Transformer"])
plt.xlabel('Hedging error')
plt.ylabel('Frequency')
plt.legend()
plt.title("Hedging errors for FFNN / Transformer - " + str(nbs_point_traj))
plt.savefig("Hedging_Errors_FFNN_Transformer" + str(nbs_point_traj) + ".png")

fig = plt.figure(figsize=(10, 5))
plt.hist([hedging_err_ffnn, hedging_err_lstm], bins=50, label=["ffnn", "LSTM"])
plt.xlabel('Hedging error')
plt.ylabel('Frequency')
plt.legend()
plt.title("Hedging errors for FFNN / LSTM - " + str(nbs_point_traj))
plt.savefig("Hedging_Errors_FFNN_LSTM" + str(nbs_point_traj) + ".png")

fig = plt.figure(figsize=(10, 5))
plt.hist([hedging_err_ffnn, hedging_err_DH], bins=50, label=["FFNN", "Delta-Hedge"])
plt.xlabel('Hedging error')
plt.ylabel('Frequency')
plt.legend()
plt.title("Hedging errors for FFNN vs Delta-Hedge - " + str(nbs_point_traj))
plt.savefig("Hedging_Errors_FFNN_DH" + str(nbs_point_traj) + ".png")

# Does not work with Transformers
# point_pred = agent.point_predict(t=6, S_t=1800, V_t=1, A_t=0, B_t=0, delta_t=0.0)
# print("Point Pred with (t=6, S_t=1800, V_t=1, A_t=0, B_t=0, delta_t=0.0): ", point_pred)
# point_pred = agent.point_predict(t=1, S_t=1800, V_t=1, A_t=0, B_t=0, delta_t=0.0)
# print("Point Pred with (t=1, S_t=1800, V_t=1, A_t=0, B_t=0, delta_t=0.0): ", point_pred)
# point_pred = agent.point_predict(t=6, S_t=600, V_t=1, A_t=0, B_t=0, delta_t=0.0)
# print("Point Pred with (t=6, S_t=600, V_t=1, A_t=0, B_t=0, delta_t=0.0): ", point_pred)
# point_pred = agent.point_predict(t=1, S_t=600, V_t=1, A_t=0, B_t=0, delta_t=0.0)
# print("Point Pred with (t=1, S_t=600, V_t=1, A_t=0, B_t=0, delta_t=0.0): ", point_pred)