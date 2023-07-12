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

nbs_point_traj = 13
batch_size = 256
train_size = 50000
test_size = 10000
epochs = 10
r_borrow = 0
r_lend = 0 
stock_dyn = "BSM" 
params_vect = [0.1, 0.1898]
S_0 = 1000
T = 252/252
alpha = 1.0
beta = 1.0
loss_type = "RSMSE"
option_type = "call"
position_type = "short"
strike = 1000
nbs_layers = 2
nbs_units = 64
lr = 0.0001
prepro_stock = "log-moneyness"
nbs_shares = 1
lambdas = [-1, -1]
name='model'

if (option_type == 'Call'):
    V_0 = Utils_general.BlackScholes_price(S_0, T, r_borrow, params_vect[1], strike, 1)
else:
    V_0 = Utils_general.BlackScholes_price(S_0, T, r_borrow, params_vect[1], strike, -1)

agent = DeepAgentTransformer.DeepAgent(nbs_point_traj, batch_size, r_borrow, r_lend, stock_dyn, params_vect, S_0, T, alpha, beta,
                 loss_type, option_type, position_type, strike, V_0, nbs_layers, nbs_units, lr, prepro_stock,
                 nbs_shares, lambdas, name='model')

# print("START")
# losses = agent.train(train_size = train_size, epochs=epochs)
# print("DONE")
agent.model = torch.load("C:\\Users\\andrei\\Documents\\school\\grad\\y2\\code_tensorflow\\" + name)
deltas_deep, hedging_err_deep, S_t_deep, V_t_deep, A_t_deep, B_t_deep, = agent.test(test_size=test_size)

print(" ----------------- ")
print(" Deep Hedging %s Results" % (loss_type))
print(" ----------------- ")
Utils_general.print_stats(hedging_err_deep, deltas_deep, loss_type, "Deep hedge - %s" % (loss_type), V_0)

print(" ----------------- ")
print(" Delta Hedging Results")
print(" ----------------- ")
deltas_DH, hedging_err_DH = Utils_general.delta_hedge_res(S_t_deep, r_borrow, r_lend, params_vect[1], T, alpha, beta, option_type="Call", position_type="Short", strike=strike, V_0=V_0, nbs_shares=nbs_shares, hab=lambdas)
Utils_general.print_stats(hedging_err_DH, deltas_DH, "Delta hedge", "Delta hedge", V_0)

# Only works for FFNN
point_pred = agent.point_predict(t=6, S_t=1800, V_t=1, A_t=0, B_t=0, delta_t=0.0)
print("Point Pred with (t=6, S_t=1800, V_t=1, A_t=0, B_t=0, delta_t=0.0): ", point_pred)
point_pred = agent.point_predict(t=1, S_t=1800, V_t=1, A_t=0, B_t=0, delta_t=0.0)
print("Point Pred with (t=1, S_t=1800, V_t=1, A_t=0, B_t=0, delta_t=0.0): ", point_pred)
point_pred = agent.point_predict(t=6, S_t=600, V_t=1, A_t=0, B_t=0, delta_t=0.0)
print("Point Pred with (t=6, S_t=600, V_t=1, A_t=0, B_t=0, delta_t=0.0): ", point_pred)
point_pred = agent.point_predict(t=1, S_t=600, V_t=1, A_t=0, B_t=0, delta_t=0.0)
print("Point Pred with (t=1, S_t=600, V_t=1, A_t=0, B_t=0, delta_t=0.0): ", point_pred)