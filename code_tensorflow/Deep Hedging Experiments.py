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

nbs_point_traj = 31
batch_size = 256
train_size = 100000
test_size = 200000
epochs = 1
r_borrow = 0
r_lend = 0
stock_dyn = "BSM" 
params_vect = [0.1, 0.1898]
S_0 = 1000
T = 30/252
alpha = 1.05
beta = 0.95
loss_type = "RSMSE"
option_type = "call"
position_type = "short"
strike = 1000
nbs_layers = 3
nbs_units = 256
lr = 0.0001
prepro_stock = "log-moneyness"
nbs_shares = 1
lambdas = [1, 1]
seed = 0
name_ffnn = 'ffnn_model'
name_lstm = 'lstm_model'
name_transformer = 'transformer_model'

if (option_type == 'Call'):
    V_0 = Utils_general.BlackScholes_price(S_0, T, r_borrow, params_vect[1], strike, 1)
else:
    V_0 = Utils_general.BlackScholes_price(S_0, T, r_borrow, params_vect[1], strike, -1)


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

agent_lstm = DeepAgentLSTM.DeepAgent(nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect, S_0, T, alpha, beta,
                 loss_type, option_type, position_type, strike, V_0, nbs_layers, nbs_units, lr, prepro_stock,
                 nbs_shares, lambdas, name=name_lstm)

print("START LSTM")
all_losses_lstm, lstm_losses = agent_lstm.train(train_size = train_size, epochs=epochs)
print("DONE LSTM")
agent_lstm.model = torch.load("/home/a_eagu/Deep-Hedging-with-Market-Impact/" + name_lstm)
deltas_lstm, hedging_err_lstm, S_t_lstm, V_t_lstm, A_t_lstm, B_t_lstm, = agent_lstm.test(test_size=test_size, test_set=test_set)
semi_square_hedging_err_lstm = np.square(np.where(hedging_err_lstm > 0, hedging_err_lstm, 0))

agent = DeepAgent.DeepAgent(nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect, S_0, T, alpha, beta,
                 loss_type, option_type, position_type, strike, V_0, nbs_layers, nbs_units, lr, prepro_stock,
                 nbs_shares, lambdas, name=name_ffnn)

print("START FFNN")
all_losses_ffnn, ffnn_losses = agent.train(train_size = train_size, epochs=epochs)
print("DONE FFNN")
agent.model = torch.load("/home/a_eagu/Deep-Hedging-with-Market-Impact/" + name_ffnn)
deltas_ffnn, hedging_err_ffnn, S_t_ffnn, V_t_ffnn, A_t_ffnn, B_t_ffnn, = agent.test(test_size=test_size, test_set=test_set)
semi_square_hedging_err_ffnn = np.square(np.where(hedging_err_ffnn > 0, hedging_err_ffnn, 0))

print(" ----------------- ")
print(" Deep Hedging %s TRANSFORMER Results" % (loss_type))
print(" ----------------- ")
Utils_general.print_stats(hedging_err_trans, deltas_trans, loss_type, "Deep hedge - TRANSFORMER - %s" % (loss_type), V_0)

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

print("TRANSFORMER S_T: ", S_t_trans[-1])
print("LSTM S_T: ", S_t_lstm[-1])
print("FFNN S_T: ", S_t_ffnn[-1])

print(" ----------------- ")
print(" t-test for (SMSE Transformer < SMSE lstm)")
print(" ----------------- ")
print(ttest_ind(semi_square_hedging_err_trans, semi_square_hedging_err_lstm, equal_var=False, alternative="less"))

print(" ----------------- ")
print(" t-test for (mean Transformer < mean lstm)")
print(" ----------------- ")
print(ttest_ind(hedging_err_trans, hedging_err_lstm, equal_var=False, alternative="less"))

print(" ----------------- ")
print(" t-test for (SMSE Transformer < SMSE ffnn)")
print(" ----------------- ")
print(ttest_ind(semi_square_hedging_err_trans, semi_square_hedging_err_ffnn, equal_var=False, alternative="less"))

print(" ----------------- ")
print(" t-test for (mean Transformer < mean ffnn)")
print(" ----------------- ")
print(ttest_ind(hedging_err_trans, hedging_err_ffnn, equal_var=False, alternative="less"))

print(" ----------------- ")
print(" t-test for (SMSE lstm < SMSE FFNN)")
print(" ----------------- ")
print(ttest_ind(semi_square_hedging_err_lstm, semi_square_hedging_err_ffnn, equal_var=False, alternative="less"))

print(" ----------------- ")
print(" t-test for (mean lstm < mean FFNN)")
print(" ----------------- ")
print(ttest_ind(hedging_err_lstm, hedging_err_ffnn, equal_var=False, alternative="less"))

print(" ----------------- ")
print(" t-test for (SMSE transformer < SMSE delta-hedge)")
print(" ----------------- ")
print(ttest_ind(semi_square_hedging_err_trans, semi_square_hedging_err_DH, equal_var=False, alternative="less"))

print(" ----------------- ")
print(" t-test for (mean transformer < mean delta-hedge)")
print(" ----------------- ")
print(ttest_ind(hedging_err_trans, hedging_err_DH, equal_var=False, alternative="less"))

print(" ----------------- ")
print(" t-test for (SMSE lstm < SMSE delta-hedge)")
print(" ----------------- ")
print(ttest_ind(semi_square_hedging_err_lstm, semi_square_hedging_err_DH, equal_var=False, alternative="less"))

print(" ----------------- ")
print(" t-test for (mean lstm < mean delta-hedge)")
print(" ----------------- ")
print(ttest_ind(hedging_err_lstm, hedging_err_DH, equal_var=False, alternative="less"))

print(" ----------------- ")
print(" t-test for (SMSE FFNN < SMSE delta-hedge)")
print(" ----------------- ")
print(ttest_ind(semi_square_hedging_err_ffnn, semi_square_hedging_err_DH, equal_var=False, alternative="less"))

print(" ----------------- ")
print(" t-test for (mean FFNN < mean delta-hedge)")
print(" ----------------- ")
print(ttest_ind(hedging_err_ffnn, hedging_err_DH, equal_var=False, alternative="less"))

all_losses_fig = plt.figure(figsize=(10, 5))
plt.plot(all_losses_lstm, label="LSTM")
plt.plot(all_losses_ffnn, label="FFNN")
plt.plot(all_losses_trans, label="Transformer")
plt.legend()
plt.savefig("all_losses" + str(nbs_point_traj) + ".png")

epoch_losses_fig = plt.figure(figsize=(10, 5))
plt.plot(lstm_losses, label="LSTM")
plt.plot(ffnn_losses, label="FFNN")
plt.plot(trans_losses, label="Transformer")
plt.legend()
plt.savefig("epoch_losses" + str(nbs_point_traj) + ".png")

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

# Only works for FFNN
# point_pred = agent.point_predict(t=6, S_t=1800, V_t=1, A_t=0, B_t=0, delta_t=0.0)
# print("Point Pred with (t=6, S_t=1800, V_t=1, A_t=0, B_t=0, delta_t=0.0): ", point_pred)
# point_pred = agent.point_predict(t=1, S_t=1800, V_t=1, A_t=0, B_t=0, delta_t=0.0)
# print("Point Pred with (t=1, S_t=1800, V_t=1, A_t=0, B_t=0, delta_t=0.0): ", point_pred)
# point_pred = agent.point_predict(t=6, S_t=600, V_t=1, A_t=0, B_t=0, delta_t=0.0)
# print("Point Pred with (t=6, S_t=600, V_t=1, A_t=0, B_t=0, delta_t=0.0): ", point_pred)
# point_pred = agent.point_predict(t=1, S_t=600, V_t=1, A_t=0, B_t=0, delta_t=0.0)
# print("Point Pred with (t=1, S_t=600, V_t=1, A_t=0, B_t=0, delta_t=0.0): ", point_pred)