import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        flat_data = self.flatten(input_data)
        logits = self.dense_layers(flat_data)
        predictions = self.softmax(logits)
        return predictions


def download_datasets():
    train_data = datasets.MNIST(
        root='data',
        download=True,
        train=True,
        transform=ToTensor()
    )
    validation_data = datasets.MNIST(
        root='data',
        download=True,
        train=False,
        transform=ToTensor()
    )

    return train_data, validation_data


def train_one_epoch(model, data_loader, loss_f, optimizer, device):
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        #     calculate loss
        predictions = model(inputs)
        loss = loss_f(predictions, targets)

        #     backpropagate loss and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Loss {}".format(loss.item()))


def train(model, data_loader, loss_f, optimizer, device, epochs):
    for i in range(epochs):
        print(f"epoch {i + 1}")
        train_one_epoch(model, data_loader, loss_f, optimizer, device)
        print("---------------------------------")

    print("Training is done")


if __name__ == "__main__":
    # download
    train_data, _ = download_datasets()
    # print(train)

    # dataloader
    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE)

    # build model
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print("Using {} device".format(device))
    feed_forward_net = FeedForward().to(device=device)

    # loss funciton
    loss_fn = nn.CrossEntropyLoss()

    # opitmizer
    optimizer = torch.optim.Adam(feed_forward_net.parameters(), lr=LEARNING_RATE)

    # train model
    train(feed_forward_net, train_data_loader, loss_fn, optimizer, device, EPOCHS)

    torch.save(feed_forward_net.state_dict(), "feedforwardnet.pth")
    print('Model Trained and stored at feedforwardnet.pth')
