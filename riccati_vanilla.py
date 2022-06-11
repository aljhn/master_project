import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq.torchdiffeq import odeint_adjoint as odeint


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Lambda(nn.Module):

    def forward(self, t, y):
        return y**2 - t

batch_t = torch.linspace(0, 10, 100)
#batch_y0 = torch.tensor([[-10.0], [-5.0], [-1.0], [0.0], [0.1], [0.5]])
#batch_y0 = torch.reshape(torch.arange(-10.0, 0.5, 0.1), (-1, 1))
batch_y0 = torch.tensor([[-2.0]])

with torch.no_grad():
    batch_y = odeint(Lambda(), batch_y0, batch_t, method='dopri5')

batch_y = batch_y[1:, :, :]
batch_t = batch_t[1:]


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 100),
            nn.Tanh(),
            #nn.Linear(100, 200),
            #nn.Tanh(),
            #nn.Linear(200, 100),
            #nn.Tanh(),
            nn.Linear(100, 50),
            nn.Tanh(),
            nn.Linear(50, 1),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)


if __name__ == "__main__":

    func = ODEFunc().to(device)
    
    optimizer = optim.Adam(func.parameters(), lr=1e-3)

    for itr in range(1, 2001):
        try:
            optimizer.zero_grad()
            pred_y = odeint(func, batch_y0, batch_t).to(device)
            loss = torch.mean(torch.abs(pred_y - batch_y))
            loss.backward()
            optimizer.step()

            if itr % 1 == 0:
                with torch.no_grad():
                    pred_y = odeint(func, batch_y0, batch_t)
                    loss = torch.mean(torch.abs(pred_y - batch_y))
                    print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
        except KeyboardInterrupt:
            break

    
    x = np.arange(-5, 5, 0.5)
    X = torch.reshape(torch.from_numpy(x), (-1, 1)).float().to(device)

    tt = np.arange(0, 10, 0.5)
    t = torch.from_numpy(tt).float().to(device)
    Y = torch.zeros((X.shape[0], t.shape[0]))
    with torch.no_grad():
        for i in range(t.shape[0]):
            Y[:, i] = func(t[i], X)[:, 0]
    
    X1, X2 = np.meshgrid(tt, x)

    ones = np.ones(Y.shape)

    plt.streamplot(tt, x, ones, Y, density=1.4, color="#A23BEC")


    x = torch.tensor([[-2.0]])

    with torch.no_grad():
        y = odeint(func, x, t).cpu().numpy()
    
    #plt.plot(tt, y[:, 0, 0], linewidth=3)
    #plt.plot(tt, y[:, 1, 0], linewidth=3)
    #plt.plot(batch_t, batch_y[:, 0, 0], linewidth=3)

    plt.xlabel("t")
    plt.ylabel("x")
    plt.tight_layout()
    plt.savefig("plots/riccati_node.pdf")
    plt.show()
