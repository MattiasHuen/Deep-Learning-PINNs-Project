import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
N_POINTS = 10


m = 1
k = 500
mu = 5
delta = mu/(2*m)
w_0 = np.sqrt(k)
w = np.sqrt(w_0**2 - delta**2 )
phi = np.arctan(-delta/w)
A = 1/(2*np.cos(phi))





t = np.sort(np.random.uniform(0, 0.5, N_POINTS))
y_data = np.exp(-delta*t)*(2*A*np.cos(phi+w*t))

print(t, y_data)


model_x = np.arange(0.0, 1.0, 0.01)
model_x_tensor = torch.tensor(model_x, dtype=torch.float32).unsqueeze(1)
t_tensor = torch.tensor(t, dtype=torch.float32).unsqueeze(1)
y_tensor = torch.tensor(y_data, dtype=torch.float32).unsqueeze(1)


# neural network 
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 10),  # 10 nodes mady better with more?? 
            nn.Tanh(),
            nn.Linear(10, 1),  
        )                      

    def forward(self, x):
        return self.net(x)

EPOCHS    = 5000
LR        = 5e-3

model = NeuralNet()

loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
            optimizer.zero_grad()
            loss = loss_fn(model(t_tensor), y_tensor)
            loss.backward()
            optimizer.step()

with torch.no_grad():
        output = model(model_x_tensor).numpy().squeeze()
        plt.plot(model_x, output, alpha=0.5, linewidth=1)



plt.plot(t,y_data)
plt.show()