import os
import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as nnfunc
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class LittleNN(nn.Module):
    def __init__(self, sizeInput=16, sizeInter=16, sizeOutput=3):
        super(LittleNN, self).__init__()
        self._sizeInput = sizeInput
        self._sizeInter = sizeInter
        self._sizeOutput = sizeOutput
        self.body = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(sizeInput, sizeInter, bias=False),
            nn.Linear(sizeInter, sizeOutput, bias=True),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.body(x)

    def trainNet(self, loader, epochs=16, onehot=False):
        optimizer = optim.Adam(self.parameters())
        scheduler = StepLR(optimizer, step_size=64, gamma=0.9)

        self.train()
        for epoch in range(epochs): 
            for step, (data, target) in enumerate(loader): 
                data, target = data.to(DEVICE), target.to(DEVICE)
                optimizer.zero_grad()
                output = self(data)
                if onehot: 
                    target = nnfunc.one_hot(target, num_classes=self._sizeOutput).float()
                loss = nnfunc.mse_loss(output, target)
                loss.backward()
                optimizer.step()
                if step % 10 == 0:
                    print('[Train] Step: {}/{}; Loss: {:.6f}'.format(epoch, step, loss.item()))


    def testNet(self, loader, onehot=False):
        self.eval()
        with torch.no_grad():
            count = 0
            correct = 0
            for iter, (data, target) in enumerate(loader): 
                count += data.shape[0]
                data, target = data.to(DEVICE), target.to(DEVICE)
                pred = self(data)
                if onehot: 
                    pred = pred.argmax(dim=1)
                correct += pred.eq(target.view_as(pred)).sum().item()
            print('[Test] Acc: {:.6f}%'.format(correct / count * 100.0))

    def predX(self, data): 
        self.eval()
        with torch.no_grad():
            return self(torch.tensor(data, device=DEVICE)).detach().cpu().numpy()


if __name__ == "__main__": 

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset = datasets.MNIST('data', train=True, download=True, transform=transform)
    testset = datasets.MNIST('data', train=False, download=True, transform=transform)

    BATCHSIZE = 64
    loaderTrain  = torch.utils.data.DataLoader(trainset, batch_size=BATCHSIZE)
    loaderTest   = torch.utils.data.DataLoader(testset, batch_size=BATCHSIZE)

    model = LittleNN(sizeInput=784, sizeInter=16, sizeOutput=10).to(DEVICE)
    print(model)
    model.trainNet(loaderTrain, epochs=4, onehot=True)
    model.testNet(loaderTest)

