import copy
import sys

# import matplotlib.pyplot as plt
import torch
import albumentations as A
import os
import pandas as pd
from tqdm import tqdm  # 이렇게 해줘야 오류가 없다.
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader
from customdataset import customDataset
from torchvision import models
from timm.loss import LabelSmoothingCrossEntropy  # 이번에 이 loss를 사용한다 선생님이 오버피팅이 덜난다고 하심.


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### 0. aug setting -> train val test
    train_transform = A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    val_transform = A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    test_transform = A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    dataset_path = 'D:/downloads/Finance_Dataset/data_2'
    ### 1. Loding classification Dataset
    train_dataset = customDataset(dataset_path + "\\train\\", transform=train_transform)
    val_dataset = customDataset(dataset_path + "\\val\\", transform=val_transform)
    test_dataset = customDataset(dataset_path + "\\test\\", transform=test_transform)

    ### def visualize_augmentations()
    # def visulize_augmentations(dataset, idx=0, samples=20, cols=5):
    #     dataset = copy.deepcopy(dataset)
    #     dataset.transform = A.Compose([t for t in dataset.transform
    #                                    if not isinstance(
    #             t, (A.Normalize, ToTensorV2)
    #         )])
    #     rows = samples // cols
    #     figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    #     for i in range(samples):
    #         image, _ = dataset[idx]
    #         ax.ravel()[i].imshow(image)
    #         ax.ravel()[i].set_axis_off()
    #     plt.tight_layout()
    #     plt.show()

    # transform한 것을 시각화 해본다.
    # visulize_augmentations(train_dataset) # 이거
    # exit() # 이거 주석 처리하면 시각화 가능함.
    ### 2. Data Loader
    # 배치 사이즈는 기본으로는 128로 하고, 메모리 에러가 나면 64로 하기로 합시다.
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    ## 모델 생성
    # 선생님이 아래 주신 HUB_URL은 라이트한 버전이라고 말씀하심.
    import torch.nn as nn
    # 방법 1
    """HUB_URL = "SharanSMenon/swin-transformer-hub:main"
    MODEL_NAME = "swin_tiny_patch4_window7_224"
    net = torch.hub.load(HUB_URL, MODEL_NAME, pretrained=True)"""

    # 방법 2
    # 스윙은 대용량 데이터를 다룰때 사용을 한다고 함.

    # swin_t(batch_size=128)
    # net = models.swin_t(weights='IMAGENET1K_V1')
    # net.head = nn.Linear(in_features=768, out_features=3) # 모델을 꼭 print()를 사용해서 확인을 주어야 한다.
    # # print(net)
    # net.to(device) # device를 해주어야 한다.

    # resnet50(batch_size=128)
    net = models.resnet50(pretrained=True)
    net.fc = nn.Linear(in_features=2048, out_features=5)
    net.to(device)

    """# resnet50(batch_size=128)
    net = models.resnet50(pretrained=True)
    net.fc = nn.Linear(in_features=2048, out_features=450)
    net.to(device)"""

    # # test model
    # net = models.efficientnet_b4()
    # net.classifier[1] = nn.Linear(in_features=1792, out_features=450)
    # net.load_state_dict(torch.load("./best_efficientnet_64.pt"))
    # net.to(device)

    """
    # efficientnet_b4(batch_size=64)
    net = models.efficientnet_b4(pretrained=True)
    net.classifier[1] = nn.Linear(in_features=1792, out_features=450)
    net.to(device)
    """

    """
    # .vgg(batch_size128)
    net = models.vgg19(pretrained=True)
    net.classifier[6] = nn.Linear(in_features=4096, out_features=450)
    net.to(device)
    """

    """
    # vit_b_16
    net = models.vit_b_16(pretrained=True)
    net.heads[0] = nn.Linear(in_features=768, out_features=450)
    net.to(device)
    """

    ### 4. epoch. optim, loss
    loss_function = LabelSmoothingCrossEntropy()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)  # net의 파라메타를 넣어줘야 한다.
    epochs = 50

    # 필요하면 스케줄러를 걸어주면 된다고 하심. 첫번째나 두번째 시간때 하셨다고 하심.(loss가 떨어진다면)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    best_val_acc = 0.0

    train_steps = len(train_loader)
    val_steps = len(val_loader)
    save_path = "best.pt"

    # 저장을 해준다.
    dfForAccuracy = pd.DataFrame(index=list(range(epochs)),
                                 columns=["Epoch", "Train_Accuracy", "Train_Loss", "Val_Accuracy", "Val_Loss"])

    if os.path.exists(save_path):
        best_val_acc = max(pd.read_csv("./modelAccuracy.csv")["Accuracy"].tolist())

    for epoch in range(epochs):
        runing_loss = 0
        val_acc = 0
        train_acc = 0

        net.train()
        # tqdm은 프로세스 진행 상태를 나타내준다.
        train_bar = tqdm(train_loader, file=sys.stdout, colour="green")
        for step, data in enumerate(train_bar):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)

            loss = loss_function(outputs, labels)
            optimizer.zero_grad()
            train_acc += (torch.argmax(outputs,
                                       dim=1) == labels).sum().item()  # torch.max로 사용해도 된다., 텐서값 뽑을 때, item()을 해준다.
            loss.backward()
            optimizer.step()
            runing_loss += loss.item()

            train_bar.desc = f"train epoch [{epoch + 1}/{epochs}], loss >> {loss.data:.3f}"

        # val을 하기 위해서 eval모드로 전환
        # 아래 두줄은 짝꿍
        net.eval()
        with torch.no_grad():
            val_loss = 0
            valid_bar = tqdm(val_loader, file=sys.stdout, colour="red")  # val은 빨간색으로 해준다.
            for data in valid_bar:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                loss = loss_function(outputs, labels)
                val_loss += loss.item()

                val_acc += (torch.argmax(outputs, dim=1) == labels).sum().item()

        # val에서는 평가만 할꺼기 때문에 loss를 하지않고 accuracy만 해주었다.
        val_accuracy = val_acc / len(val_dataset)
        train_accuracy = train_acc / len(train_dataset)

        dfForAccuracy.loc[epoch, "Epoch"] = epoch + 1
        dfForAccuracy.loc[epoch, "Train_Accuracy"] = round(train_accuracy, 3)  # round는 반올림해준다.
        dfForAccuracy.loc[epoch, "Train_Loss"] = round(runing_loss / train_steps, 3)
        dfForAccuracy.loc[epoch, "Val_Accuracy"] = round(val_accuracy, 3)
        dfForAccuracy.loc[epoch, "Val_Loss"] = round(val_loss / val_steps, 3)
        print(
            f"epoch [{epoch + 1}/{epochs}] trian_loss{(runing_loss / train_steps):.3f} train_acc : {train_accuracy:.3f} val_acc : {val_accuracy:.3f}")

        if val_accuracy > best_val_acc:  # best를 loss로하는 경우도 있고 accuracy로 하는 경우도 있다.
            best_val_acc = val_accuracy
            torch.save(net.state_dict(), save_path)

        if epoch % 5 == 0 :
            torch.save(net.state_dict(), save_path + str(epoch))

        if epoch == epochs - 1:
            dfForAccuracy.to_csv("./modelAccuracy.csv", index=False)

    torch.save(net.state_dict(), "./last.pt")

    ## test
    # def acc_function(correct, total):
    #     acc = correct / total * 100
    #     return acc
    #
    # def test(model, data_loader, device):  # <- 기본 standard이다.
    #     model.eval()
    #     correct = 0
    #     total = 0
    #     with torch.no_grad():
    #         for i, (image, label) in enumerate(data_loader):
    #             images, labels = image.to(device), label.to(device)
    #             output = model(images)  # output에서는 예측된 값이 나온다.
    #             _, argmax = torch.max(output, 1)  # output에서 max인걸 뽑는다.
    #             total += images.size(0)
    #             correct += (labels == argmax).sum().item()
    #
    #         acc = acc_function(correct, total)
    #         print("accuracy for {} image : {:.2f}%".format(total, acc))
    #
    # test(net,test_loader, device)


if __name__ == '__main__':
    main()
