# Tomer Shay, Roei Gida
import argparse
from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np
import logging
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import TensorDataset
import torchvision
import pickle


class BaseModel(nn.Module):
    def __init__(self, image_size, lr):
        super().__init__()

        self.name = 'Base Model'
        self.lr = lr
        self.image_size = image_size

        self.train_accuracies = []
        self.train_loss = []
        self.validate_accuracies = []
        self.validate_loss = []
        self.test_accuracies = []
        self.test_loss = []


class ModelA(BaseModel):
    def __init__(self, image_size, lr):
        super().__init__(image_size, lr)

        self.name = 'Model A'
        self.modelsketch = "100 ReLU 50 ReLU 10 Softmax"

        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

        self.optimizer = optim.SGD(self.parameters(), lr=self.lr)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class ModelA_middlelinear(BaseModel):
    def __init__(self, image_size, lr):
        super().__init__(image_size, lr)

        self.name = 'Model A middle linear'
        self.modelsketch = "100 ReLU 100 Linear 100 Linear 50 ReLU 10 Softmax"

        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 10)

        self.optimizer = optim.SGD(self.parameters(), lr=self.lr)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


class ModelB(ModelA):
    def __init__(self, image_size, lr):
        super().__init__(image_size, lr)

        self.name = 'Model B'
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)


class ModelB_middlelinear(ModelA_middlelinear):
    def __init__(self, image_size, lr):
        super().__init__(image_size, lr)

        self.name = 'Model B middle linear'
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)


class ModelC(ModelB):
    def __init__(self, image_size, lr, dropout=0.2):
        super().__init__(image_size, lr)

        self.name = 'Model C'

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class ModelC_middlelinear(ModelB_middlelinear):
    def __init__(self, image_size, lr, dropout=0.2):
        super().__init__(image_size, lr)

        self.name = 'Model C middle linear'

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

class ModelD(ModelB):
    def __init__(self, image_size, lr):
        super().__init__(image_size, lr)

        self.name = 'Model D'

        self.batch_norm_1 = nn.BatchNorm1d(100)
        self.batch_norm_2 = nn.BatchNorm1d(50)

    def forward(self, x):
        x = x.view(-1, self.image_size)

        x = self.fc0(x)
        x = self.batch_norm_1(x)
        x = F.relu(x)

        x = self.fc1(x)
        x = self.batch_norm_2(x)
        x = F.relu(x)

        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class ModelD_middlelinear(ModelB_middlelinear):
    def __init__(self, image_size, lr):
        super().__init__(image_size, lr)

        self.name = 'Model D middle linear'

        self.batch_norm_1 = nn.BatchNorm1d(100)
        self.batch_norm_2 = nn.BatchNorm1d(50)

    def forward(self, x):
        x = x.view(-1, self.image_size)

        x = self.fc0(x)
        x = self.batch_norm_1(x)
        x = F.relu(x)

        x = self.fc1(x)
        x = self.fc2(x)

        x = self.fc3(x)
        x = self.batch_norm_2(x)
        x = F.relu(x)

        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


class ModelE(BaseModel):
    def __init__(self, image_size, lr):
        super().__init__(image_size, lr)

        self.name = 'Model E'
        self.modelsketch = "128 ReLU 64 ReLU 10 ReLU 10 ReLU 10 Softmax"

        self.fc0 = nn.Linear(image_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)

        self.optimizer = optim.SGD(self.parameters(), lr=self.lr)
        self.activation_func = F.relu

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = self.activation_func(self.fc0(x))
        x = self.activation_func(self.fc1(x))
        x = self.activation_func(self.fc2(x))
        x = self.activation_func(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


class ModelE_middlelinear(BaseModel):
    def __init__(self, image_size, lr):
        super().__init__(image_size, lr)

        self.name = 'Model E middle linear'
        self.modelsketch = "128 ReLU 64 Linear 10 Linear 10 ReLU 10 Softmax"

        self.fc0 = nn.Linear(image_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)

        self.optimizer = optim.SGD(self.parameters(), lr=self.lr)
        self.activation_func = F.relu

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = self.activation_func(self.fc0(x))
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.activation_func(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

class ModelF(ModelE):
    def __init__(self, image_size, lr):
        super().__init__(image_size, lr)

        self.name = 'Model F'
        self.modelsketch = "128 Sigmoid 64 Sigmoid 10 Sigmoid 10 Sigmoid 10 Softmax"

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.activation_func = torch.sigmoid


class ModelF_middlelinear(ModelE_middlelinear):
    def __init__(self, image_size, lr):
        super().__init__(image_size, lr)

        self.name = 'Model F middle linear'
        self.modelsketch = "128 Sigmoid 64 Linear 10 Linear 10 Sigmoid 10 Softmax"

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.activation_func = torch.sigmoid


class ModelG(BaseModel):
    def __init__(self, image_size, lr):
        super().__init__(image_size, lr=lr)

        self.name = 'Model G'
        self.modelsketch = "512 LeakyReLU 256 LeakyReLU 128 LeakyReLU 64 LeakyReLU 10 Softmax"

        self.batch_norm_0 = nn.BatchNorm1d(image_size)
        self.batch_norm_1 = nn.BatchNorm1d(512)
        self.batch_norm_2 = nn.BatchNorm1d(256)
        self.batch_norm_3 = nn.BatchNorm1d(128)
        self.batch_norm_4 = nn.BatchNorm1d(64)

        self.fc0 = nn.Linear(image_size, 512)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

        self.dropout = nn.Dropout(p=0.1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = self.dropout(x)

        x = F.leaky_relu(self.batch_norm_1(self.fc0(x)))

        x = F.leaky_relu(self.batch_norm_2(self.fc1(x)))
        x = self.dropout(x)

        x = F.leaky_relu(self.batch_norm_3(self.fc2(x)))

        x = F.leaky_relu(self.batch_norm_4(self.fc3(x)))
        x = self.dropout(x)

        x = self.fc4(x)
        return F.log_softmax(x, dim=1)
    
class ModelG_middlelinear(BaseModel):
    def __init__(self, image_size, lr):
        super().__init__(image_size, lr=lr)

        self.name = 'Model G middle linear'
        self.modelsketch = "512 LeakyReLU 256 Linear 128 Linear 64 LeakyReLU 10 Softmax"

        self.batch_norm_0 = nn.BatchNorm1d(image_size)
        self.batch_norm_1 = nn.BatchNorm1d(512)
        self.batch_norm_4 = nn.BatchNorm1d(64)

        self.fc0 = nn.Linear(image_size, 512)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

        self.dropout = nn.Dropout(p=0.1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = self.dropout(x)

        x = F.leaky_relu(self.batch_norm_1(self.fc0(x)))

        x = self.fc1(x)
        x = self.dropout(x)

        x = self.fc2(x)

        x = F.leaky_relu(self.batch_norm_4(self.fc3(x)))
        x = self.dropout(x)

        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


def export_plot(model):
    plt.subplot(2, 1, 1)
    plt.title(f'{model.name}')
    plt.plot(model.train_accuracies, label="train")
    plt.plot(model.validate_accuracies, label="validate")
    plt.plot(model.test_accuracies, label="test")
    plt.ylabel("Accuracy")

    plt.subplot(2, 1, 2)
    plt.plot(model.train_loss, label="train")
    plt.plot(model.validate_loss, label="validate")
    plt.plot(model.test_loss, label="test")
    plt.ylabel('Loss')

    plt.xlabel('Epochs')

    plt.legend()
    plt.savefig(f'{model.name}.png')
    logging.info(f'plot saved to \'{model.name}.png\'')


def train(model, train_set):
    model.train()
    train_loss = 0
    correct = 0
    for _, (x, y) in enumerate(train_set):
        model.optimizer.zero_grad()
        output = model(x)
        loss = F.nll_loss(output, y)
        loss.backward()
        model.optimizer.step()
        train_loss += float(loss.data)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(y.view_as(pred)).sum().item()

    model.train_accuracies.append(100 * correct / len(train_set.dataset))
    model.train_loss.append(train_loss / (len(train_set.dataset) / train_set.batch_size))


def validate(model, validate_set, is_test=False):
    model.eval()
    validate_loss = 0
    correct = 0
    with torch.no_grad():
        for _, (x, y) in enumerate(validate_set):
            output = model(x)
            loss = F.nll_loss(output, y)
            validate_loss += float(loss.data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()

    if not is_test:
        model.validate_accuracies.append(100 * correct / len(validate_set.dataset))
        model.validate_loss.append(validate_loss / (len(validate_set.dataset) / validate_set.batch_size))
    else:
        model.test_accuracies.append(100 * correct / len(validate_set.dataset))
        model.test_loss.append(validate_loss / (len(validate_set.dataset) / validate_set.batch_size))


def running_epochs(model, epochs, is_best):
    global train_loader, validate_loader, test_loader

    best_acc_model = deepcopy(model)
    best_val_acc = 0

    for i in range(epochs):
        if is_best:
            if i == 14:
                model.optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.975)
        if i % 7 == 0:
            model.optimizer.param_groups[0]['lr'] *= 0.2

        logging.info(f'====== {model.name} EPOCH #{i + 1} ============')

        train(model, train_loader)
        logging.info(f'[train accuracy:]\t\t\t{"{:.2f}".format(model.train_accuracies[-1])}%')
        logging.info(f'[train loss:]\t\t\t\t{"{:.2f}".format(model.train_loss[-1])}')

        validate(model, validate_loader)
        logging.info(f'[validate accuracy:]\t\t{"{:.2f}".format(model.validate_accuracies[-1])}%')
        logging.info(f'[validate loss:]\t\t\t{"{:.2f}".format(model.validate_loss[-1])}')

        if best_val_acc < model.validate_accuracies[-1]:
            best_val_acc = model.validate_accuracies[-1]
            best_acc_model = deepcopy(model)
            if best_val_acc > 90:
                model.optimizer.param_groups[0]['lr'] = 0.0001
            logging.info('---- model saved! ----')

        validate(model, test_loader, is_test=True)
        logging.info(f'[test accuracy:]\t\t\t{"{:.2f}".format(model.test_accuracies[-1])}%')
        logging.info(f'[test loss:]\t\t\t\t{"{:.2f}".format(model.test_loss[-1])}')

    return best_acc_model


def load_original_mnist_fashion(batch_size, validate_percentage):
    logging.info("loading files..")
    transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])

    dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transforms, download=True)
    train_set, val_set = torch.utils.data.random_split(dataset=dataset, lengths=[1 - validate_percentage / 100,validate_percentage / 100])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    validate_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.FashionMNIST('./data', download=True, train=False, transform=transforms),
        batch_size=batch_size)

    return train_loader, validate_loader, test_loader

if __name__ == "__main__":
    # get arguments
    parser = argparse.ArgumentParser()

    # -train_x train_x -train_y train_y -test_x test_x -test_y test_y -e epochs...
    parser.add_argument("-e", dest="epochs", default="30",type=int, help="Epochs")
    parser.add_argument("-batch_size", dest="batch_size", default="64",type=int, help="Batch Size")
    parser.add_argument("-validate", dest="validate_percentage", default="10",type=int, help="Validate Percentage")
    parser.add_argument("--model", dest="model",help="The Model to run (between A to G), add ml after for middle linear")
    parser.add_argument("--filename", dest="filename",help="filename for logging and output purposes")

    args = parser.parse_args()
    logging.basicConfig(filename=f"log/{args.filename}.out", encoding='utf-8', level=logging.INFO)
    logging.info(f"{args.epochs,type(args.epochs)}")
    logging.info(f"{args.batch_size,type(args.batch_size)}")
    logging.info(f"{args.validate_percentage,type(args.validate_percentage)}")
    logging.info(f"{args.model,type(args.model)}")

    train_loader, validate_loader, test_loader = load_original_mnist_fashion(args.batch_size,args.validate_percentage)
    is_best = False
    if args.model == 'A':
        model = ModelA(image_size=28 * 28, lr=0.12)
    elif args.model == 'B':
        model = ModelB(image_size=28 * 28, lr=0.0001)
    elif args.model == 'C':
        model = ModelC(image_size=28 * 28, lr=0.0001)
    elif args.model == 'D':
        model = ModelD(image_size=28 * 28, lr=0.01)
    elif args.model == 'E':
        model = ModelE(image_size=28 * 28, lr=0.1)
    elif args.model == 'F':
        model = ModelF(image_size=28 * 28, lr=0.001)
    elif args.model == 'G':
        model = ModelG(image_size=28 * 28, lr=0.001)
        is_best = True
    elif args.model == 'Aml':
        model = ModelA_middlelinear(image_size=28 * 28, lr=0.12)
    elif args.model == 'Bml':
        model = ModelB_middlelinear(image_size=28 * 28, lr=0.0001)
    elif args.model == 'Cml':
        model = ModelC_middlelinear(image_size=28 * 28, lr=0.0001)
    elif args.model == 'Dml':
        model = ModelD_middlelinear(image_size=28 * 28, lr=0.01)
    elif args.model == 'Eml':
        model = ModelE_middlelinear(image_size=28 * 28, lr=0.1)
    elif args.model == 'Fml':
        model = ModelF_middlelinear(image_size=28 * 28, lr=0.001)
    elif args.model == 'Gml':
        model = ModelG_middlelinear(image_size=28 * 28, lr=0.001)
    else:
        raise ValueError("need to specify model")

    best_model = running_epochs(model, args.epochs, is_best=is_best)
    torch.save(best_model.state_dict(), f"{args.filename}/model.pt")
    with open(f"{args.filename}/model.pkl", "wb") as f:
        pickle.dump(best_model, f, -1)
    logging.info("========================================")
    logging.info("learn finished and best model saved")

    logging.info("exporting plot..")

    logging.info('\nfinal accuracy:')
    validate(best_model, test_loader, is_test=True)
    logging.info(f'[test accuracy:]\t\t\t{"{:.2f}".format(best_model.test_accuracies[-1])}%')
    logging.info(f'[test loss:]\t\t\t\t{"{:.2f}".format(best_model.test_loss[-1])}')
