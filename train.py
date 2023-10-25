from data import Data
# import model
from model import Net

net = Net()
dataset = Data()
for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(dataset.trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradient
        net.optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = net.criterion(outputs, labels)
        loss.backward()
        net.optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
print('Finished Training')