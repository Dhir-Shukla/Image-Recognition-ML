from torch import nn
from torch.optim import Adam
import torch
import numpy as np
from sklearn.utils import shuffle


class Classifier(nn.Module):

    def __init__(self):
        super(classifier, self).__init__()

        self.r = nn.ReLU()
        self.s = nn.Softmax(dim=0)

        self.l1 = nn.Linear(19, 15)
        self.l11 = nn.Linear(15, 10)
        self.l2 = nn.Linear(10, 5)
        self.l3 = nn.Linear(5, 3)

    def forward(self, x):
        x = torch.tensor(x)
        x = torch.tensor(x.float())

        l1 = self.l1(x)
        # b1 = self.b1(l1)
        r1 = self.r(l1)

        l11 = self.l11(r1)
        r11 = self.r(l11)

        l2 = self.l2(r11)
        r2 = self.r(l2)

        l3 = self.l3(r2)
        l3 = self.r(l3)

        s = self.s(l3)

        return s


def train(model, data, lr=0.001, num_epochs=20):

    cross = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=0.0)

    for epoch in range(1, num_epochs + 1):
        model.train()
        for i in range(len(data[0])):
            # Making the prediction
            prediction = model.forward(data[0][i])
            # Calculating the loss
            loss = cross(prediction, torch.tensor(data[1][i], dtype=torch.float))
            # Zeroing the gradients from previous optimization step
            optimizer.zero_grad()
            # Calculating the loss for each parameter
            loss.backward()
            # Updating each parameter
            optimizer.step()

        accuracy = get_accuracy(model, data)

        print(f"Epoch {epoch}, Acc {round(accuracy, 2) * 100}%, Loss {loss}")


def get_accuracy(model, acsdata):
    model.eval()
    correct = 0
    total = 0
    for i in range(len(acsdata[0])):
        y_hat = model.forward(acsdata[0][i])
        pred = y_hat.detach().numpy()
        pred = np.argmax(pred, axis=0)
        correct += 1 if pred == np.argmax(np.array(acsdata[1][i])) else 0
        total += 1
    return correct/total


def one_hot(ar):
    data1 = []
    for i in ar:
        if i == 1:
            data1.append([1, 0, 0])
        elif i == 2:
            data1.append([0, 1, 0])
        elif i == 3:
            data1.append([0, 0, 1])
        else:
            print("error")
            exit(2)
    return data1


if __name__ == "__main__":

    # Creating an instance of our model
    model = Classifier()
    data = np.load("data.npy", allow_pickle=True)
    dt2 = one_hot(data[1])
    X, y = shuffle(data[0], dt2, random_state=26)
    # Dividing the values with 100 to make the training more stable
    for c in range(len(X)):
        X[c] = np.array(X[c])/100

    # Using around 80% of data for training and 20% of data for testing
    train_data = X[:483], y[:483]
    test_data = X[483:], y[483:]
    train(model, train_data)
    print(f"{get_accuracy(model, test_data)}%")

    for layer in model.state_dict():
        print(f"{layer} = {model.state_dict()[layer]} \n \n")

