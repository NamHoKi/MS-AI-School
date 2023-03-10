from customdataset import customDataset
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np


from torchvision import models
import torch.nn as nn

import matplotlib.pyplot as plt


def test(model, test_loader, device) :
    model.eval()
    correct = 0
    total = 0
    label_cnt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    y_pred = []
    y_true = []

    classes = ("down_down", "down_up", "up_down", "up_up")
    with torch.no_grad() :
        for i, (image, labels) in enumerate(test_loader) :
            image, labels = image.to(device), labels.to(device)
            output = model(image)
            _, argmax = torch.max(output, 1)
            total += image.size(0)
            correct += (labels == argmax).sum().item()

            for label, prediction in zip(labels, argmax) :
                label, pred = label.item(), prediction.item()
                label_cnt[(int(label) - 1) * 2] += 1
                if label == pred :
                    label_cnt[((int(label) - 1) * 2) + 1] += 1


            # Confusion matrix
            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output)  # Save Prediction

            labels = labels.data.cpu().numpy()
            y_true.extend(labels)  # Save Truth

        # Build confusion matrix
        cf_matrix = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1), index=[i for i in classes],
                             columns=[i for i in classes])

        plt.figure(figsize=(12, 7))
        sn.heatmap(df_cm, annot=True)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig('output.png')

        # print total acc
        acc = correct / total * 100
        print("acc for {} image : {:.2f}%".format(
            total, acc
        ))

        # print label acc
        for i in range(4) :
            if label_cnt[(i*2)] > 0 :
                print(classes[i], label_cnt[(i*2) + 1] / label_cnt[(i*2)])


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_transform = A.Compose([
        A.SmallestMaxSize(max_size=256),
        A.Resize(height=224, width=224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    test_dataset = customDataset("./dataset_padding_and_resize224/padding_and_resize224/test/", transform=test_transform)

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(in_features=2048, out_features=4)
    model.load_state_dict(torch.load("./0309Resnet50/best.pt", map_location=device))
    model.to(device)

    test(model, test_loader, device)
