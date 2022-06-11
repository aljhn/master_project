import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq.torchdiffeq import odeint_adjoint as odeint
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

seed = 42069

import random
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

m1 = 1
m2 = 1
m = m2 / m1

l1 = 1
l2 = 1
l = l2 / l1

class Lambda(nn.Module):

    def __init__(self, constrained=False):
        super(Lambda, self).__init__()
        self.constrained = constrained

    def forward(self, t, y):
        n = y.shape[0]

        M = torch.zeros((n, 2, 2))
        M[:, 0, 0] = 1 + m
        M[:, 0, 1] = m * l * torch.cos(y[:, 0] - y[:, 1])
        M[:, 1, 0] = m * l * torch.cos(y[:, 0] - y[:, 1])
        M[:, 1, 1] = m * (l**2)

        M_det = M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]

        M_inv = torch.zeros((n, 2, 2))
        M_inv[:, 0, 0] = M[:, 1, 1] / M_det
        M_inv[:, 0, 1] = - M[:, 0, 1] / M_det
        M_inv[:, 1, 0] = - M[:, 1, 0] / M_det
        M_inv[:, 1, 1] = M[:, 0, 0] / M_det
        

        #ff = torch.zeros((n, 2, 1))
        #ff[:, 0, 0] = - (1 + m) * torch.sin(y[:, 0]) - m * l * (y[:, 3]**2) * torch.sin(y[:, 0] - y[:, 1])
        #ff[:, 1, 0] = - m * l * torch.sin(y[:, 1]) + m * l * (y[:, 2]**2) * torch.sin(y[:, 0] - y[:, 1])

        ff = torch.stack([- (1 + m) * torch.sin(y[:, 0]) - m * l * (y[:, 3]**2) * torch.sin(y[:, 0] - y[:, 1]), - m * l * torch.sin(y[:, 1]) + m * l * (y[:, 2]**2) * torch.sin(y[:, 0] - y[:, 1])], dim=1)
        ff = torch.reshape(ff, (*ff.shape, 1))

        yy = torch.matmul(M_inv, ff)

        u1 = 0
        u2 = 0

        if self.constrained:
            u1 = -0.1 * y[:, 0] - 0.01 * y[:, 2]
            u2 = -2.0 * (y[:, 1] - y[:, 0]) - 0.5 * y[:, 3]

        return torch.stack([y[:, 2], y[:, 3], yy[:, 0, 0] + u1, yy[:, 1, 0] + u2], dim=1)

"""m1 = 1
m2 = 1
l1 = 1
l2 = 1
g = 9.81

class Lambda(nn.Module):

    def __init__(self, constrained=False):
        super(Lambda, self).__init__()
        self.constrained = constrained

    def forward(self, t, y):
        M = torch.zeros((y.shape[0], 2, 2))
        M[:, 0, 0] = (m1 + m2) * (l1**2)
        M[:, 0, 1] = m2 * l1 * l2 * np.cos(y[:, 0] - y[:, 1])
        M[:, 1, 0] = m2 * l1 * l2 * np.cos(y[:, 0] - y[:, 1])
        M[:, 1, 1] = m2 * (l2**2)

        M_det = M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]

        M_inv = torch.zeros((y.shape[0], 2, 2))
        M_inv[:, 0, 0] = M[:, 1, 1] / M_det
        M_inv[:, 0, 1] = - M[:, 0, 1] / M_det
        M_inv[:, 1, 0] = - M[:, 1, 0] / M_det
        M_inv[:, 1, 1] = M[:, 0, 0] / M_det

        ff = torch.stack((- (m1 + m2) * g * l1 * np.sin(y[:, 0]) - m2 * l1 * l2 * (y[:, 3]**2) * np.sin(y[:, 0] - y[:, 1]), - m2 * l2 * g * np.sin(y[:, 1]) + m2 * l1 * l2 * (y[:, 2]**2) * np.sin(y[:, 0] - y[:, 1])), dim=1).unsqueeze(-1)

        u = torch.zeros_like(ff)
        if self.constrained:
            u[:, 0, 0] = -0.001 * y[:, 0] - 0.001 * y[:, 2]
            u[:, 1, 0] = -1.0 * (y[:, 1] - y[:, 0]) - 0.5 * y[:, 3]

        yy = torch.matmul(M_inv, ff + u)

        return torch.stack([y[:, 2], y[:, 3], yy[:, 0, 0], yy[:, 1, 0]], dim=1)"""


lambda_unconstrained = Lambda(constrained=False)
lambda_constrained = Lambda(constrained=True)


with torch.no_grad():
    dt = 0.1
    T = 10
    batch_t = torch.arange(0, T, dt)

    batch_size = 200

    batch_y0_train = torch.zeros((batch_size, 4))
    batch_y0_train[:, 0] = torch.rand(batch_size) * 6 - 3
    batch_y0_train[:, 1] = batch_y0_train[:, 0] + torch.rand(batch_size) - 0.5
    batch_y0_train[:, 2] = torch.rand(batch_size) * 4 - 2
    batch_y0_train[:, 3] = torch.rand(batch_size) * 2 - 1

    batch_y0_val = torch.zeros((batch_size, 4))
    batch_y0_val[:, 0] = torch.rand(batch_size) * 6 - 3
    batch_y0_val[:, 1] = batch_y0_val[:, 0] + torch.rand(batch_size) - 0.5
    batch_y0_val[:, 2] = torch.rand(batch_size) * 4 - 2
    batch_y0_val[:, 3] = torch.rand(batch_size) * 2 - 1


    batch_y_unconstrained_train = odeint(lambda_unconstrained, batch_y0_train, batch_t, method="dopri5")
    batch_y_constrained_train = odeint(lambda_constrained, batch_y0_train, batch_t, method="dopri5")
    batch_y_unconstrained_val = odeint(lambda_unconstrained, batch_y0_val, batch_t, method="dopri5")
    batch_y_constrained_val = odeint(lambda_constrained, batch_y0_val, batch_t, method="dopri5")

    batch_dy_unconstrained_train = torch.zeros_like(batch_y_unconstrained_train)
    batch_dy_constrained_train = torch.zeros_like(batch_y_constrained_train)
    batch_dy_unconstrained_val = torch.zeros_like(batch_y_unconstrained_val)
    batch_dy_constrained_val = torch.zeros_like(batch_y_constrained_val)

    for i in range(batch_dy_unconstrained_train.shape[0] - 1):
        batch_dy_unconstrained_train[i, :, :] = (batch_y_unconstrained_train[i + 1, :, :] - batch_y_unconstrained_train[i, :, :]) / dt
        batch_dy_constrained_train[i, :, :] = (batch_y_constrained_train[i + 1, :, :] - batch_y_constrained_train[i, :, :]) / dt
        batch_dy_unconstrained_val[i, :, :] = (batch_y_unconstrained_val[i + 1, :, :] - batch_y_unconstrained_val[i, :, :]) / dt
        batch_dy_constrained_val[i, :, :] = (batch_y_constrained_val[i + 1, :, :] - batch_y_constrained_val[i, :, :]) / dt
    batch_dy_unconstrained_train = batch_dy_unconstrained_train[:-1, :, :]
    batch_dy_constrained_train = batch_dy_constrained_train[:-1, :, :]
    batch_dy_unconstrained_val = batch_dy_unconstrained_val[:-1, :, :]
    batch_dy_constrained_val = batch_dy_constrained_val[:-1, :, :]

    batch_y_unconstrained_train = batch_y_unconstrained_train[1:, :, :]
    batch_y_constrained_train = batch_y_constrained_train[1:, :, :]
    batch_y_unconstrained_val = batch_y_unconstrained_val[1:, :, :]
    batch_y_constrained_val = batch_y_constrained_val[1:, :, :]
    batch_t = batch_t[1:]



"""
class LSTMClassifier(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.linear1 = nn.Linear(hidden_size, hidden_size // 2)
        self.linear2 = nn.Linear(hidden_size // 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x, (hn, cn) = self.lstm(x)
        x = hn.reshape(2 * batch_size, self.hidden_size)
        x = self.linear1(x)
        x = self.sigmoid(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        x = x.squeeze()
        return x

batch_x_train = torch.concat((torch.concat((batch_y0_train.unsqueeze(0), batch_y_unconstrained_train), dim=0), torch.concat((batch_y0_train.unsqueeze(0), batch_y_constrained_train), dim=0)), dim=1)
batch_x_val = torch.concat((torch.concat((batch_y0_val.unsqueeze(0), batch_y_unconstrained_val), dim=0), torch.concat((batch_y0_val.unsqueeze(0), batch_y_constrained_val), dim=0)), dim=1)
batch_y = torch.concat((torch.ones(batch_size), torch.zeros(batch_size)), dim=0)

hidden_size = 300
lstm_classifier = LSTMClassifier(batch_x_train.shape[2], hidden_size)
lstm_optimizer = optim.Adam(lstm_classifier.parameters(), lr=1e-3)
criterion = nn.BCELoss()

epochs = 200

accs = np.zeros(epochs)

for epoch in range(1, epochs + 1):
    lstm_optimizer.zero_grad()
    y_pred = lstm_classifier(batch_x_train)
    loss = criterion(y_pred, batch_y)
    loss.backward()
    lstm_optimizer.step()

    with torch.no_grad():
        y_pred = lstm_classifier(batch_x_val)
        y_pred = y_pred.numpy().round()
        accuracy = int(accuracy_score(batch_y, y_pred) * 100)

    print("Epoch: {:02d} | Loss: {:.6f} | Val Accuracy: {:2d}%".format(epoch, loss.item(), accuracy))
    accs[epoch - 1] = accuracy

plt.figure()
plt.plot(np.arange(1, epochs + 1, 1), accs)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("LSTM Validation Accuracy")
plt.grid()
plt.tight_layout()
plt.savefig("plots/lstm_acc_.pdf")
plt.show()

exit()


FX: 89% acc
OG: 98% acc
"""




class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(4, 50),
            nn.Tanh(),
            nn.Linear(50, 100),
            nn.Tanh(),
            nn.Linear(100, 200),
            nn.Tanh(),
            nn.Linear(200, 100),
            nn.Tanh(),
            nn.Linear(100, 50),
            nn.Tanh(),
            nn.Linear(50, 4),
        )

    def forward(self, t, y):
        return self.net(y)


func1 = ODEFunc() # Trained on unconstrained
func2 = ODEFunc() # Trained on constrained

optimizer1 = optim.Adam(func1.parameters(), lr=1e-3)
optimizer2 = optim.Adam(func2.parameters(), lr=1e-3)
criterion = nn.MSELoss(reduction="mean")

params1 = tuple(func1.parameters())
params2 = tuple(func2.parameters())

epochs = 100

accs = np.zeros((epochs, 2))

for epoch in range(1, epochs + 1):
    try:
        #dy1 = func1(batch_t, batch_y_unconstrained_train)
        #dy2 = func2(batch_t, batch_y_constrained_train)
        #loss1 = criterion(dy1, batch_dy_unconstrained_train)
        #loss2 = criterion(dy2, batch_dy_constrained_train)

        #pred_y1 = odeint(func1, batch_y0_train, batch_t)
        #pred_y2 = odeint(func2, batch_y0_train, batch_t)
        #loss1 = criterion(pred_y1, batch_y_unconstrained_train)
        #loss2 = criterion(pred_y2, batch_y_constrained_train)

        #optimizer1.zero_grad()
        #optimizer2.zero_grad()
        #loss1.backward()
        #loss2.backward()
        #optimizer1.step()
        #optimizer2.step()


        """gradients = [torch.zeros_like(param, requires_grad=False) for param in params1]
        
        normalization_jacobian = torch.zeros((batch_size, batch_y_unconstrained_train.shape[2], batch_y_unconstrained_train.shape[2]))
        
        F = func1(batch_t[0], batch_y_unconstrained_train[0, :, :])
        F_norm = torch.linalg.norm(F, dim=1)
        for b in range(batch_size):
            normalization_jacobian[b, :, :] = torch.diag(torch.ones(4)) / F_norm[b] - torch.outer(F[b, :], F[b, :]) / (F_norm[b] ** 3)
        
        dr = batch_dy_unconstrained_train[0, :, :]
        dr = torch.nn.functional.normalize(dr, dim=1)
        vjp_prev = torch.autograd.grad(F, params1, torch.bmm(normalization_jacobian, dr.unsqueeze(-1)).squeeze())

        for i in range(1, batch_y_unconstrained_train.shape[0]):
            F = func1(batch_t[i], batch_y_unconstrained_train[i, :, :])
            F_norm = torch.linalg.norm(F, dim=1)
            for b in range(batch_size):
                normalization_jacobian[b, :, :] = torch.diag(torch.ones(4)) / F_norm[b] - torch.outer(F[b, :], F[b, :]) / (F_norm[b] ** 3)
            
            dr = batch_dy_unconstrained_train[i, :, :]
            dr = torch.nn.functional.normalize(dr, dim=1)
            vjp_next = torch.autograd.grad(F, params1, torch.bmm(normalization_jacobian, dr.unsqueeze(-1)).squeeze())

            for j in range(len(params1)):
                gradients[j] += dt * (vjp_next[j] + vjp_prev[j]) / 2 # Trapezoid integration
            
            vjp_prev = vjp_next

        optimizer1.zero_grad()
        i = 0
        for param in func1.parameters():
            param.grad = - gradients[i] / batch_size / T
            i += 1
        optimizer1.step()



        gradients = [torch.zeros_like(param, requires_grad=False) for param in params2]
        
        normalization_jacobian = torch.zeros((batch_size, batch_y_constrained_train.shape[2], batch_y_constrained_train.shape[2]))
        
        F = func2(batch_t[0], batch_y_constrained_train[0, :, :])
        F_norm = torch.linalg.norm(F, dim=1)
        for b in range(batch_size):
            normalization_jacobian[b, :, :] = torch.diag(torch.ones(4)) / F_norm[b] - torch.outer(F[b, :], F[b, :]) / (F_norm[b] ** 3)
        
        dr = batch_dy_constrained_train[0, :, :]
        dr = torch.nn.functional.normalize(dr, dim=1)
        vjp_prev = torch.autograd.grad(F, params2, torch.bmm(normalization_jacobian, dr.unsqueeze(-1)).squeeze())

        for i in range(1, batch_y_constrained_train.shape[0]):
            F = func2(batch_t[i], batch_y_constrained_train[i, :, :])
            F_norm = torch.linalg.norm(F, dim=1)
            for b in range(batch_size):
                normalization_jacobian[b, :, :] = torch.diag(torch.ones(4)) / F_norm[b] - torch.outer(F[b, :], F[b, :]) / (F_norm[b] ** 3)
            
            dr = batch_dy_constrained_train[i, :, :]
            dr = torch.nn.functional.normalize(dr, dim=1)
            vjp_next = torch.autograd.grad(F, params2, torch.bmm(normalization_jacobian, dr.unsqueeze(-1)).squeeze())

            for j in range(len(params2)):
                gradients[j] += dt * (vjp_next[j] + vjp_prev[j]) / 2 # Trapezoid integration
            
            vjp_prev = vjp_next

        optimizer2.zero_grad()
        i = 0
        for param in func2.parameters():
            param.grad = - gradients[i] / batch_size / T
            i += 1
        optimizer2.step()"""

        loss1 = torch.zeros(1)
        loss2 = torch.zeros(1)


            
        
        with torch.no_grad():
            unconstrained_line_integral1 = torch.zeros(batch_size)

            F = lambda_unconstrained(batch_t[0], batch_y_unconstrained_val[0, :, :])
            F = torch.nn.functional.normalize(F, dim=1)
            dr = batch_dy_unconstrained_val[0, :, :]
            dr = torch.nn.functional.normalize(dr, dim=1)
            dot_prev = torch.sum(F * dr, dim=1)

            for i in range(1, batch_t.shape[0]):
                F = lambda_unconstrained(batch_t[i], batch_y_unconstrained_val[i, :, :])
                F = torch.nn.functional.normalize(F, dim=1)
                dr = batch_dy_unconstrained_val[i, :, :]
                dr = torch.nn.functional.normalize(dr, dim=1)
                dot_next = torch.sum(F * dr, dim=1)

                unconstrained_line_integral1 += dt * (dot_next + dot_prev) / 2
                dot_prev = dot_next

            unconstrained_line_integral1 /= T
            unconstrained_line_integral_loss1 = torch.mean(unconstrained_line_integral1)

            constrained_line_integral1 = torch.zeros(batch_size)

            F = lambda_unconstrained(batch_t[0], batch_y_constrained_val[0, :, :])
            F = torch.nn.functional.normalize(F, dim=1)
            dr = batch_dy_constrained_val[0, :, :]
            dr = torch.nn.functional.normalize(dr, dim=1)
            dot_prev = torch.sum(F * dr, dim=1)

            for i in range(1, batch_t.shape[0]):
                F = lambda_unconstrained(batch_t[i], batch_y_constrained_val[i, :, :])
                F = torch.nn.functional.normalize(F, dim=1)
                dr = batch_dy_constrained_val[i, :, :]
                dr = torch.nn.functional.normalize(dr, dim=1)
                dot_next = torch.sum(F * dr, dim=1)

                constrained_line_integral1 += dt * (dot_next + dot_prev) / 2
                dot_prev = dot_next

            constrained_line_integral1 /= T
            constrained_line_integral_loss1 = torch.mean(constrained_line_integral1)
        
            unconstrained_line_integral2 = torch.zeros(batch_size)

            F = lambda_constrained(batch_t[0], batch_y_unconstrained_val[0, :, :])
            F = torch.nn.functional.normalize(F, dim=1)
            dr = batch_dy_unconstrained_val[0, :, :]
            dr = torch.nn.functional.normalize(dr, dim=1)
            dot_prev = torch.sum(F * dr, dim=1)

            for i in range(1, batch_t.shape[0]):
                F = lambda_constrained(batch_t[i], batch_y_unconstrained_val[i, :, :])
                F = torch.nn.functional.normalize(F, dim=1)
                dr = batch_dy_unconstrained_val[i, :, :]
                dr = torch.nn.functional.normalize(dr, dim=1)
                dot_next = torch.sum(F * dr, dim=1)

                unconstrained_line_integral2 += dt * (dot_next + dot_prev) / 2
                dot_prev = dot_next

            unconstrained_line_integral2 /= T
            unconstrained_line_integral_loss2 = torch.mean(unconstrained_line_integral2)

            constrained_line_integral2 = torch.zeros(batch_size)

            F = lambda_constrained(batch_t[0], batch_y_constrained_val[0, :, :])
            F = torch.nn.functional.normalize(F, dim=1)
            dr = batch_dy_constrained_val[0, :, :]
            dr = torch.nn.functional.normalize(dr, dim=1)
            dot_prev = torch.sum(F * dr, dim=1)

            for i in range(1, batch_t.shape[0]):
                F = lambda_constrained(batch_t[i], batch_y_constrained_val[i, :, :])
                F = torch.nn.functional.normalize(F, dim=1)
                dr = batch_dy_constrained_val[i, :, :]
                dr = torch.nn.functional.normalize(dr, dim=1)
                dot_next = torch.sum(F * dr, dim=1)

                constrained_line_integral2 += dt * (dot_next + dot_prev) / 2
                dot_prev = dot_next

            constrained_line_integral2 /= T
            constrained_line_integral_loss2 = torch.mean(constrained_line_integral2)


            X_train1 = np.concatenate((unconstrained_line_integral1[:batch_size // 2].numpy(), constrained_line_integral1[:batch_size // 2].numpy()))
            X_train2 = np.concatenate((unconstrained_line_integral2[:batch_size // 2].numpy(), constrained_line_integral2[:batch_size // 2].numpy()))
            X_test1 = np.concatenate((unconstrained_line_integral1[batch_size // 2:].numpy(), constrained_line_integral1[batch_size // 2:].numpy()))
            X_test2 = np.concatenate((unconstrained_line_integral2[batch_size // 2:].numpy(), constrained_line_integral2[batch_size // 2:].numpy()))
            y_train = np.concatenate((np.ones(batch_size // 2), np.zeros(batch_size // 2)))
            y_test = np.concatenate((np.ones(batch_size // 2), np.zeros(batch_size // 2)))

            X_train1 = X_train1.reshape(-1, 1)
            X_train2 = X_train2.reshape(-1, 1)
            X_test1 = X_test1.reshape(-1, 1)
            X_test2 = X_test2.reshape(-1, 1)

            classifier1 = LogisticRegression()
            #classifier1 = SVC()
            classifier1.fit(X_train1, y_train)
            y_pred1 = classifier1.predict(X_test1)
            accuracy1 = int(accuracy_score(y_test, y_pred1) * 100)

            classifier2 = LogisticRegression()
            #classifier2 = SVC()
            classifier2.fit(X_train2, y_train)
            y_pred2 = classifier2.predict(X_test2)
            accuracy2 = int(accuracy_score(y_test, y_pred2) * 100)

            print("Epoch {:04d} | ULI1 {:.3f} | CLI1 {:.3f} | ULI2 {:.3f} | CLI2 {:.3f} | ClfAcc1 {:2d}% | ClfAcc2 {:2d}%".format(epoch, unconstrained_line_integral_loss1.item(), constrained_line_integral_loss1.item(), unconstrained_line_integral_loss2.item(), constrained_line_integral_loss2.item(), accuracy1, accuracy2))
            accs[epoch - 1, 0] = accuracy1
            accs[epoch - 1, 1] = accuracy2


    except KeyboardInterrupt:
        break

plt.figure()
plt.plot(np.arange(1, epochs + 1, 1), accs)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Unconstrained Model", "Constrained Model"])
plt.title("Line Integral Classifier Validation Accuracy")
plt.grid()
plt.tight_layout()
plt.savefig("plots/line_integral_truemodel_acc.pdf")
plt.show()
