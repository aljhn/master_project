import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq.torchdiffeq import odeint_adjoint as odeint


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 2),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)


batch_t = torch.linspace(0, 1, 10)
batch_y = torch.reshape(torch.stack([torch.linspace(1, -1, 10), torch.linspace(-1, 1, 10)], 1), (10, 2, 1))
batch_y0 = torch.concat([batch_y[0, :, :], torch.tensor([[0.0], [0.0]])], 1)
batch_y = batch_y[1:, :, :]
batch_t = batch_t[1:]

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

            if itr % 50 == 0:
                with torch.no_grad():
                    pred_y = odeint(func, batch_y0, batch_t)
                    loss = torch.mean(torch.abs(pred_y - batch_y))
                    print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
        except KeyboardInterrupt:
            break

    
    x = torch.reshape(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]), (2, 2))
    t = torch.linspace(0, 1, 10)

    with torch.no_grad():
        y = odeint(func, x, t).cpu().numpy()
    
    plt.plot(t.cpu().numpy(), y[:, 0, 0])
    plt.plot(t.cpu().numpy(), y[:, 1, 0])
    plt.xlabel("t")
    plt.ylabel("x")
    plt.tight_layout()
    plt.savefig("plots/nonhomeomorphicflows_anode.pdf")
    plt.show()
