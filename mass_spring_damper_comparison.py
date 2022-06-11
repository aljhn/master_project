import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq.torchdiffeq import odeint_adjoint as odeint


seed = 42069

import random
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




class Lambda(nn.Module):

    def __init__(self):
        super(Lambda, self).__init__()
        self.m = 1
        self.k = 1
        self.d = 1

        self.a = self.k / self.m
        self.b = self.d / self.m

    def forward(self, t, y):
        y_ddot = - self.a * y[:, 0] - self.b * y[:, 1]
        return torch.stack((y[:, 1], y_ddot), dim=1)


system = Lambda()


with torch.no_grad():
    dt = 0.1
    dt_val = 0.01
    T = 1
    batch_t = torch.arange(0, T, dt)
    batch_t_val = torch.arange(0, T, dt_val)

    batch_size = 20

    batch_y0_train = torch.rand((batch_size, 2)) * 20 - 10
    batch_y0_val = torch.rand((batch_size, 2)) * 20 - 10
    
    batch_y_train = odeint(system, batch_y0_train, batch_t, method="dopri5")
    batch_y_val = odeint(system, batch_y0_val, batch_t_val, method="dopri5")

    batch_dy_train = torch.zeros_like(batch_y_train)
    batch_dy_val = torch.zeros_like(batch_y_val)

    for i in range(batch_dy_train.shape[0] - 1):
        batch_dy_train[i, :, :] = (batch_y_train[i + 1, :, :] - batch_y_train[i, :, :]) / dt
        batch_dy_val[i, :, :] = (batch_y_val[i + 1, :, :] - batch_y_val[i, :, :]) / dt_val
    batch_dy_train = batch_dy_train[:-1, :, :]
    batch_dy_val = batch_dy_val[:-1, :, :]

    batch_y_train = batch_y_train[1:, :, :]
    batch_y_val = batch_y_val[1:, :, :]
    batch_t = batch_t[1:]
    batch_t_val = batch_t_val[1:]

    batch_t = batch_t.to(device)
    batch_t_val = batch_t_val.to(device)
    batch_y0_train = batch_y0_train.to(device)
    batch_y0_val = batch_y0_val.to(device)
    batch_y_train = batch_y_train.to(device)
    batch_y_val = batch_y_val.to(device)
    batch_dy_train = batch_dy_train.to(device)
    batch_dy_val = batch_dy_val.to(device)


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 100),
            nn.Tanh(),
            #nn.Linear(100, 200),
            #nn.Tanh(),
            #nn.Linear(200, 100),
            #nn.Tanh(),
            nn.Linear(100, 50),
            nn.Tanh(),
            nn.Linear(50, 2),
        )

    def forward(self, t, y):
        return self.net(y)


func_node =  ODEFunc().to(device)
func_regressor = ODEFunc().to(device)

criterion = nn.MSELoss(reduction="mean")
optimizer_node = optim.Adam(func_node.parameters(), lr=1e-3)
optimizer_regressor = optim.Adam(func_regressor.parameters(), lr=1e-3)

epochs = 100

losses = np.zeros((epochs, 2))

for epoch in range(1, epochs + 1):
    try:
        optimizer_node.zero_grad()
        optimizer_regressor.zero_grad()

        pred_y = odeint(func_node, batch_y0_train, batch_t, method="dopri5")
        loss_node = criterion(pred_y, batch_y_train)

        dy = func_regressor(batch_t, batch_y_train)
        loss_regressor = criterion(dy, batch_dy_train)

        with torch.no_grad():
            pred_y = odeint(func_node, batch_y0_val, batch_t_val, method="dopri5")
            loss_node_val = criterion(pred_y, batch_y_val)

            pred_y = odeint(func_regressor, batch_y0_val, batch_t_val, method="dopri5")
            loss_regressor_val = criterion(pred_y, batch_y_val)
    
        loss_node.backward()
        loss_regressor.backward()

        optimizer_node.step()
        optimizer_regressor.step()

        print("Epoch {:04d} | NODE L2 Val {:.4f} | Regressor L2 Val {:.4f}".format(epoch, loss_node_val.item(), loss_regressor_val.item()))

        losses[epoch - 1, 1] = loss_node_val.item()
        losses[epoch - 1, 0] = loss_regressor_val.item()

    except KeyboardInterrupt:
        break


batch_t = torch.arange(0, T, 0.01)
y_true = odeint(system, batch_y0_train, batch_t, method="dopri5")

with torch.no_grad():
    y_node = odeint(func_node, batch_y0_train, batch_t, method="dopri5")
    y_regressor = odeint(func_regressor, batch_y0_train, batch_t, method="dopri5")

x = np.arange(-10, 10, 0.1)
n = len(x)

X = torch.zeros((n * n, 2))
for i in range(n):
    for j in range(n):
        X[i + n * j, 0] = x[j]
        X[i + n * j, 1] = x[i]

with torch.no_grad():
    Y = system(None, X)

X1 = np.zeros((n, n))
X2 = np.zeros((n, n))
Y1 = np.zeros((n, n))
Y2 = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        X1[i, j] = X[i + n * j, 0]
        X2[i, j] = X[i + n * j, 1]
        Y1[i, j] = Y[i + n * j, 0]
        Y2[i, j] = Y[i + n * j, 1]


plt.figure()

trajectory_index = 0
plt.subplot(2, 2, 1)
plt.plot(y_true[:, trajectory_index, 0], y_true[:, trajectory_index, 1], linewidth=3)
plt.plot(y_regressor[:, trajectory_index, 0], y_regressor[:, trajectory_index, 1], linewidth=3)
plt.plot(y_node[:, trajectory_index, 0], y_node[:, trajectory_index, 1], linewidth=3)
plt.streamplot(X1, X2, Y1, Y2, density=1.0, linewidth=None, color="#A23BEC")
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.xlim([-10, 10])
plt.ylim([-10, 10])

trajectory_index = 1
plt.subplot(2, 2, 2)
plt.plot(y_true[:, trajectory_index, 0], y_true[:, trajectory_index, 1], linewidth=3)
plt.plot(y_regressor[:, trajectory_index, 0], y_regressor[:, trajectory_index, 1], linewidth=3)
plt.plot(y_node[:, trajectory_index, 0], y_node[:, trajectory_index, 1], linewidth=3)
plt.streamplot(X1, X2, Y1, Y2, density=1.0, linewidth=None, color="#A23BEC")
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.xlim([-10, 10])
plt.ylim([-10, 10])

trajectory_index = 2
plt.subplot(2, 2, 3)
plt.plot(y_true[:, trajectory_index, 0], y_true[:, trajectory_index, 1], linewidth=3)
plt.plot(y_regressor[:, trajectory_index, 0], y_regressor[:, trajectory_index, 1], linewidth=3)
plt.plot(y_node[:, trajectory_index, 0], y_node[:, trajectory_index, 1], linewidth=3)
plt.streamplot(X1, X2, Y1, Y2, density=1.0, linewidth=None, color="#A23BEC")
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.xlim([-10, 10])
plt.ylim([-10, 10])

trajectory_index = 3
plt.subplot(2, 2, 4)
plt.plot(y_true[:, trajectory_index, 0], y_true[:, trajectory_index, 1], linewidth=3)
plt.plot(y_regressor[:, trajectory_index, 0], y_regressor[:, trajectory_index, 1], linewidth=3)
plt.plot(y_node[:, trajectory_index, 0], y_node[:, trajectory_index, 1], linewidth=3)
plt.streamplot(X1, X2, Y1, Y2, density=1.0, linewidth=None, color="#A23BEC")
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.xlim([-10, 10])
plt.ylim([-10, 10])

plt.figlegend(["True System", "Regression Model", "Neural ODE"])

plt.tight_layout()

plt.savefig("plots/mass_spring_damper_trajectories_h01_T1.pdf")
plt.show()



plt.figure()
plt.plot(np.arange(1, epochs + 1, 1), losses)
plt.legend(["Regression Model", "Neural ODE"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Comparable Validation Losses")
plt.yscale("log")
plt.yticks(np.round(np.linspace(np.min(losses), np.max(losses), 5), decimals=2), np.round(np.linspace(np.min(losses), np.max(losses), 5), decimals=2))
plt.grid()
plt.savefig("plots/mass_spring_damper_loss_comparison_h01_T1.pdf")
plt.show()
