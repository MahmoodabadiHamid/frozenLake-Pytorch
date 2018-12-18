import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt




class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden )   # hidden layer
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden * 2)
        self.hidden3 = torch.nn.Linear(n_hidden * 2, 300)
        self.predict = torch.nn.Linear(300, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = torch.relu(self.hidden2(x))
        x = torch.tanh(self.hidden3(x))
        x = self.predict(x)             # linear output
        return x   



input_size = 1
output_size = 2
num_epochs = 5000
learning_rate = 0.001
x_train = []
y_train = []



for i in range(150):
    a = random.uniform(0, 7)
    x_train.append([a])
    #y_train.append([math.sin(a)])
    y_train.append([math.sin(a),math.cos(a)])


x_train = np.array(x_train, dtype=np.float32)#, [5.5], [6.71], [6.93], [4.168], 
                    #[9.779], [6.182], [7.59], [2.167], [7.042], 
                    #[10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array(y_train, dtype=np.float32)#, [2.09], [3.19], [1.694], [1.573], 
                    #[3.366], [2.596], [2.53], [1.221], [2.827], 
                    #[3.465], [1.65], [2.904], [1.3]], dtype=np.float32)


model = Net(input_size,50,output_size)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Loss and optimizer
criterion = nn.MSELoss()


# Train the model
for epoch in range(num_epochs):
    # Convert numpy arrays to torch tensors
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % (num_epochs/2) == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
        predicted = model(torch.from_numpy(x_train)).detach().numpy()
        plt.plot(x_train, y_train, 'ro', label='Original data')
        plt.scatter(x_train, predicted[:,0], label='Fitted model to sin', marker=r'_')
        plt.scatter(x_train, predicted[:,1], label='Fitted model to cos', marker=r'_')
        plt.legend()
        plt.show()



