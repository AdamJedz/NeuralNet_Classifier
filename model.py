from network import Network
import numpy as np
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt

from load_data import download_data, create_loaders


class Model:

    def __init__(self, name):
        self.name = name

        torch.manual_seed(0)
        self.model = Network().cuda()
        print(self.model)

    def num_params(self):

        model = self.model
        num_el = 0

        for item in model.parameters():
            num_el += item.numel()
        print(f'Number of elements of model {self.name} is {num_el}')

    def train_model(self, train_loader, test_loader, epochs=50, early_stopping_rounds=10, plot=True):

        def save_model(model, name):
            torch.save(model.state_dict(), f'./Model/{name}_best.pt')

        def print_stats(epoch, train_loss, train_acc, test_loss, test_acc, st_time):
            print(f'Epoch: {epoch+1} Loss: {train_loss.item():10.3f} Accuracy: {train_acc.item()*100/50000:7.2f}% \
            Val-loss: {test_loss.item():10.3f} Val-accuracy: {test_acc.item()*100/10000:7.2f}% Epoch time: {time.time() - st_time:.2f} seconds.')

        def plot_loss(train_loss, test_loss):
            plt.figure(figsize=(20,4))
            plt.plot(train_loss, label='Train Loss')
            plt.plot(test_loss, label='Test Loss')
            plt.legend()
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Crossentropy Loss')
            plt.show()

        def plot_acc(train_acc, test_acc):
            plt.figure(figsize=(20,4))
            plt.plot([t*100/50000 for t in train_acc], label='Train Accuracy')
            plt.plot([t*100/10000 for t in test_acc], label='Test Accuracy')
            plt.legend()
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.show()

        model = self.model

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)

        start_time = time.time()
        epochs = epochs
        train_losses = []
        test_losses = []
        train_correct = []
        test_correct = []
        best_accuracy = 0
        early_stopping = 0

        for e in range(epochs):
            epoch_time = time.time()
            trn_cor = 0
            tst_cor = 0
            batch_loss = 0
            batch_tst_loss = 0

            model.train()

            for b, (X_train, y_train) in enumerate(train_loader):
                X_train = X_train.cuda()
                y_train = y_train.cuda()

                y_pred = model(X_train)
                loss = criterion(y_pred, y_train)

                predicted = torch.max(y_pred.data, 1)[1]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_cor = (predicted == y_train).sum()
                trn_cor += batch_cor
                batch_loss += loss

            batch_loss = batch_loss / (50000/10)
            train_losses.append(batch_loss)
            train_correct.append(trn_cor)

            model.eval()
            with torch.no_grad():
                for b, (X_test, y_test) in enumerate(test_loader):

                    X_test = X_test.cuda()
                    y_test = y_test.cuda()

                    y_val = model(X_test)
                    predicted = torch.max(y_val.data, 1)[1]
                    batch_cor = (predicted == y_test).sum()
                    tst_cor += batch_cor

                    tst_loss = criterion(y_val, y_test)
                    batch_tst_loss += tst_loss

                batch_tst_loss = batch_tst_loss / (10000/10)
                test_losses.append(batch_tst_loss)
                test_correct.append(tst_cor)

            test_accuracy = tst_cor.item()*100/10000
            print_stats(e, batch_loss, trn_cor, batch_tst_loss, tst_cor, epoch_time)

            if test_accuracy > best_accuracy:
                save_model(model, self.name)
                best_accuracy = test_accuracy
                early_stopping = 0
            else:
                early_stopping += 1

            if early_stopping >= early_stopping_rounds:
                print(f'Early stopping after {e+1} epochs.')
                break

        print(f'Duration of training: {(time.time() - start_time)/60:.2f} minutes.')

        if plot:
            plot_loss(train_losses, test_losses)
            plot_acc(train_correct, test_correct)

    def load_model(self):
        self.model.load_state_dict(torch.load(f'./Model/{self.name}_best.pt'))





if __name__ == "__main__":
    model = Model('test')
    model.num_params()

    train_data, test_data = download_data()
    train_loader, test_loader = create_loaders(train_data, test_data)
    model.train_model(train_loader, test_loader)




