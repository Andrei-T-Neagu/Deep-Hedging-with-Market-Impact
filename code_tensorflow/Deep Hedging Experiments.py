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
from scipy.stats import ttest_ind

nbs_point_traj = 13
batch_size = 256
train_size = 100000
test_size = 100000
epochs = 20
r_borrow = 0
r_lend = 0
stock_dyn = "BSM" 
params_vect = [0.1, 0.1898]
S_0 = 1000
T = 252/252
alpha = 1.00
beta = 1.00
loss_type = "RSMSE"
option_type = "call"
position_type = "short"
strike = 1000
nbs_layers = 2
nbs_units = 256
lr = 0.0001
prepro_stock = "log-moneyness"
nbs_shares = 1
lambdas = [-1, -1]
name_ffnn = 'ffnn_model'
name_lstm = 'lstm_model'

if (option_type == 'Call'):
    V_0 = Utils_general.BlackScholes_price(S_0, T, r_borrow, params_vect[1], strike, 1)
else:
    V_0 = Utils_general.BlackScholes_price(S_0, T, r_borrow, params_vect[1], strike, -1)

agent_t = DeepAgentLSTM.DeepAgent(nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect, S_0, T, alpha, beta,
                 loss_type, option_type, position_type, strike, V_0, nbs_layers, nbs_units, lr, prepro_stock,
                 nbs_shares, lambdas, name=name_lstm)

print("START LSTM")
all_losses_lstm, losses = agent_t.train(train_size = train_size, epochs=epochs)
print("DONE LSTM")
agent_t.model = torch.load("/home/a_eagu/Deep-Hedging-with-Market-Impact/" + name_lstm)
deltas_deep, hedging_err_deep, S_t_deep, V_t_deep, A_t_deep, B_t_deep, = agent_t.test(test_size=test_size)

agent = DeepAgent.DeepAgent(nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect, S_0, T, alpha, beta,
                 loss_type, option_type, position_type, strike, V_0, nbs_layers, nbs_units, lr, prepro_stock,
                 nbs_shares, lambdas, name=name_ffnn)

print("START FFNN")
all_losses_ffnn = agent.train(train_size = train_size, epochs=epochs)
print("DONE FFNN")
agent.model = torch.load("/home/a_eagu/Deep-Hedging-with-Market-Impact/" + name_ffnn)
deltas_ffnn, hedging_err_ffnn, S_t_ffnn, V_t_ffnn, A_t_ffnn, B_t_ffnn, = agent.test(test_size=test_size)

print(" ----------------- ")
print(" Deep Hedging %s LSTM Results" % (loss_type))
print(" ----------------- ")
Utils_general.print_stats(hedging_err_deep, deltas_deep, loss_type, "Deep hedge - %s" % (loss_type), V_0)

print(" ----------------- ")
print(" Deep Hedging %s FFNN Results" % (loss_type))
print(" ----------------- ")
Utils_general.print_stats(hedging_err_ffnn, deltas_ffnn, loss_type, "Deep hedge - %s" % (loss_type), V_0)

print(" ----------------- ")
print(" Delta Hedging Results")
print(" ----------------- ")
deltas_DH, hedging_err_DH = Utils_general.delta_hedge_res(S_t_ffnn, r_borrow, r_lend, params_vect[1], T, alpha, beta, option_type="Call", position_type="Short", strike=strike, V_0=V_0, nbs_shares=nbs_shares, hab=lambdas)
Utils_general.print_stats(hedging_err_DH, deltas_DH, "Delta hedge", "Delta hedge", V_0)

print(" ----------------- ")
print(" t-test for (mean lstm < mean FFNN)")
print(" ----------------- ")
print(ttest_ind(hedging_err_deep, hedging_err_ffnn, equal_var=False, alternative="less"))

print(" ----------------- ")
print(" t-test for (mean lstm < mean delta-hedge)")
print(" ----------------- ")
print(ttest_ind(hedging_err_deep, hedging_err_DH, equal_var=False, alternative="less"))

print(" ----------------- ")
print(" t-test for (mean FFNN < mean delta-hedge)")
print(" ----------------- ")
print(ttest_ind(hedging_err_ffnn, hedging_err_DH, equal_var=False, alternative="less"))

plt.plot(all_losses_lstm[0])
plt.show()
plt.savefig("all_losses_lstm.png")
plt.plot(all_losses_ffnn[0])
plt.savefig("all_losses_ffnn.png")

# Only works for FFNN
# point_pred = agent.point_predict(t=6, S_t=1800, V_t=1, A_t=0, B_t=0, delta_t=0.0)
# print("Point Pred with (t=6, S_t=1800, V_t=1, A_t=0, B_t=0, delta_t=0.0): ", point_pred)
# point_pred = agent.point_predict(t=1, S_t=1800, V_t=1, A_t=0, B_t=0, delta_t=0.0)
# print("Point Pred with (t=1, S_t=1800, V_t=1, A_t=0, B_t=0, delta_t=0.0): ", point_pred)
# point_pred = agent.point_predict(t=6, S_t=600, V_t=1, A_t=0, B_t=0, delta_t=0.0)
# print("Point Pred with (t=6, S_t=600, V_t=1, A_t=0, B_t=0, delta_t=0.0): ", point_pred)
# point_pred = agent.point_predict(t=1, S_t=600, V_t=1, A_t=0, B_t=0, delta_t=0.0)
# print("Point Pred with (t=1, S_t=600, V_t=1, A_t=0, B_t=0, delta_t=0.0): ", point_pred)