from model import Model
from load_data import download_data, create_loaders, class_names
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from torchvision.utils import make_grid

def classify_images(name):
    model = Model(name)
    model.load_model()

    train, test = download_data()
    test_loader = create_loaders(train, test, batch_size=10000, test_only=True)

    model.model.eval()
    with torch.no_grad():
        correct = 0
        for X_test, y_test in test_loader:
            X_test = X_test.cuda()
            y_test= y_test.cuda()
            y_val = model.model(X_test)
            predicted = torch.max(y_val, 1)[1]
            correct += (predicted == y_test).sum()

    print((f'Test Accuracy: {correct.item()}/{len(test)} = {correct.item()*100/(len(test)):7.2f}%'))
    return y_test.cpu(), predicted.cpu(), X_test.cpu()

def plot_conf_matrix(true, pred):
    arr = confusion_matrix(true.view(-1), pred.view(-1))
    dt_cm = pd.DataFrame(arr, class_names, class_names)
    plt.figure(figsize=(9,6))
    sns.heatmap(dt_cm, annot=True, fmt='d', cmap='BuGn')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def show_sample_misses(true, pred, data):
    misses = (pred != true).nonzero()
    random_misses = np.random.choice(misses.view(-1), 10, False)

    np.set_printoptions(formatter=dict(int=lambda x: f'{x:5}'))
    print('True:', *np.array([class_names[cl] for cl in true[random_misses]]))
    print('Pred:', *np.array([class_names[cl] for cl in pred[random_misses]]))
    images = data[random_misses]
    im = make_grid(images, nrow=5)
    plt.figure(figsize=(8, 4))
    plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
    plt.show()



if __name__ == "__main__":

    true, pred, data = classify_images('test')
    plot_conf_matrix(true,pred)
    for i in range(2):
        show_sample_misses(true, pred, data)