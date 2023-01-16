# import copy
# import os.path
# import time
#
# import torch
# import torchvision
# from customdataset import my_dataset
# import albumentations as A
# from albumentations.pytorch.transforms import ToTensorV2
# from torch.utils.data import DataLoader
# import torch.nn as nn
# from timm.loss import LabelSmoothingCrossEntropy
#
# import pandas as pd
#
# import cv2
#
# from PIL import Image
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # 수정
# # train_transforms = A.Compose([
# #     A.SmallestMaxSize(max_size=224),
# #     A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.08, rotate_limit=20,
# #                        p=0.8),
# #     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224,0.255)),
# #     ToTensorV2(),
# # ])
#
# train_transform = A.Compose([
#     A.SmallestMaxSize(max_size=160),
#     A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.09,
#                         rotate_limit=25, p=0.6),
#     A.Resize(width=224, height=224),
#     # A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.6),
#     A.RandomBrightnessContrast(p=0.6),
#     # A.HorizontalFlip(p=0.6),
#     A.GaussNoise(p=0.5),
#     # A.Equalize(p=0.5),
#     A.VerticalFlip(p=0.6),
#     # A.ISONoise(always_apply=False, p=0.5, intensity=(0.1, 0.5), color_shift=(0.01, 0.22)),
#     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#     ToTensorV2()
# ])
#
# val_transform = A.Compose([
#     A.SmallestMaxSize(max_size=160),
#     A.Resize(width=224, height=224),
#     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#     ToTensorV2()
# ])
# test_transform = A.Compose([
#     A.SmallestMaxSize(max_size=160),
#     A.Resize(width=224, height=224),
#     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#     ToTensorV2()
# ])
#
# # dataset
# train_dataset = my_dataset("./dataset/train/", transform=train_transform)
# val_dataset = my_dataset("./dataset/val/", transform=val_transform)
# # test_dataset = my_dataset("./dataset/test/", transform=test_transform)
# # dataloader
# train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True,
#                           num_workers=2, pin_memory=True)
# val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False,
#                         num_workers=2, pin_memory=True)
# # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
# model = torch.hub.load('facebookresearch/deit:main',
#                        'deit_tiny_patch16_224', pretrained=False)
# model.head = nn.Linear(in_features=192, out_features=3)
# model.to(device)
# # model =  torchvision.models.swin_t(weights="IMAGENET1K_V1")
# # model.head = torch.nn.Linear(in_features=1024, out_features=100)
# # model.to(device)
# # model = torchvision.models.resnet18(pretrained=False)
# # model.fc = torch.nn.Linear(in_features=512, out_features=6)
# # model.to(device)
#
# criterion = LabelSmoothingCrossEntropy()
# # optimizer = torch.optim.Adam(model.head.parameters(), lr=0.01)
# # optimizer = torch.optim.Adam(model.head.parameters(), lr=0.001)
#
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)  # net의 파라메타를 넣어줘야 한다.
#
# # lr scheduler
# exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
#
# def train(model, criterion, train_loader, val_loader, optimizer, scheduler, num_epochs=20,
#           device=device) :
#     total = 0
#     best_loss = 9999
#     since = time.time()
#     best_model_wts = copy.deepcopy(model.state_dict())
#
#     for epoch in range(num_epochs) :
#         print(f"Epoch {epoch} / {num_epochs - 1}")
#         print("-"*10)
#
#         for index, (image, label) in enumerate(train_loader) :
#             image, label = image.to(device), label.to(device)
#             output = model(image)
#             loss = criterion(output, label)
#             scheduler.step()
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             _, argmax = torch.max(output, 1)
#             acc = (label == argmax).float().mean()
#             total += label.size(0)
#
#             if (index + 1) % 10 == 0 :
#                 print("Epoch [{}/{}], Step [{}/{}], Loss {:.4f}, Acc {:.2f}".format(
#                     epoch + 1, num_epochs, index+1, len(train_loader), loss.item(),
#                     acc.item() * 100
#                 ))
#         aveg_loss, val_acc = validation(epoch, model, val_loader, criterion, device)
#         if aveg_loss < best_loss :
#             best_loss = aveg_loss
#             best_model_wts = copy.deepcopy(model.state_dict())
#             save_model(model, save_dir="./")
#
#     time_elapsed = time.time()  - since
#     print("Training complete in {:.0f}m {:.0f}s".format(
#         time_elapsed // 60, time_elapsed % 6
#     ))
#     model.load_state_dict(best_model_wts)
#
# def validation(epoch, model, val_loader, criterion, device) :
#     print("Start validation # {}" .format(epoch+1))
#
#     dfForAccuracy = pd.DataFrame(index=list(range(epoch)),
#                                  columns=["Epoch", "Accurascy"])
#     model.eval()
#     with torch.no_grad() :
#         total = 0
#         correct = 0
#         total_loss = 0
#         cnt = 0
#         batch_loss = 0
#
#         for i, (imgs, labels) in enumerate(val_loader) :
#             imags, labels = imgs.to(device), labels.to(device)
#             output = model(imags)
#             loss = criterion(output, labels)
#             batch_loss += loss.item()
#
#             total += imags.size(0)
#             _, argmax = torch.max(output, 1)
#             correct += (labels == argmax).sum().item()
#             total_loss += loss
#             cnt +=1
#
#     avrg_loss = total_loss / cnt
#     val_acc = (correct / total * 100)
#     print("Validation #{} Acc : {:.2f}% Average Loss : {:.4f}".format(
#         epoch + 1,
#         correct / total * 100,
#         avrg_loss
#     ))
#
#     # dfForAccuracy.loc[epoch, "Epoch"] = epoch + 1
#     # dfForAccuracy.loc[epoch, "Train_Accuracy"] = round(train_accuracy, 3)
#     # dfForAccuracy.loc[epoch, "Train_Loss"] = round(runing_loss / train_steps, 3)
#     # dfForAccuracy.loc[epoch, "Val_Accuracy"] = round(val_accuracy, 3)
#     # dfForAccuracy.loc[epoch, "Val_Loss"] = round(val_loss / val_steps, 3)
#     # print(
#     #     f"epoch [{epoch + 1}/{epochs}] trian_loss{(runing_loss / train_steps):.3f} train_acc : {train_accuracy:.3f} val_acc : {val_accuracy:.3f}")
#     #
#     # if epoch == epochs - 1:
#     #     dfForAccuracy.to_csv("./modelAccuracy.csv", index=False)
#
#     return avrg_loss, val_acc
#
# def save_model(model, save_dir, file_name = "best_resnet18.pt") :
#     output_path = os.path.join(save_dir,file_name)
#     torch.save(model.state_dict(), output_path)
#
# def test(model, test_loader, device) :
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad() :
#         for i, (image, labels) in enumerate(test_loader) :
#             image, labels = image.to(device), labels.to(device)
#             output = model(image)
#             _, argmax = torch.max(output,1)
#             total += image.size(0)
#             correct += (labels == argmax).sum().item()
#
#         acc = correct / total * 100
#         print("acc for {} image : {:.2f}%".format(
#             total, acc
#         ))
#
#
# # def test_image(model, img, device, test_transform) :
# #     model.eval()
# #     img = Image.open(img)  # 이미지 불러오기
# #     img_size = img.size  # 이미지의 크기 측정
# #
# #     x = img_size[0]  # 넓이값
# #     y = img_size[1]  # 높이값
# #
# #     if x != y:
# #         size = max(x, y)
# #         resized_img = Image.new(mode='RGB', size=(size, size), color=(0, 0, 0))
# #         offset = (round((abs(x - size)) / 2), round((abs(y - size)) / 2))
# #         resized_img.paste(img, offset)
# #         result_img = resized_img.resize((224, 224))
# #     else:
# #         result_img = img.resize((224, 224))
# #
# #     input_img = copy.deepcopy(result_img)
# #     input_img = test_transform(image=img)["image"]
# #     print(model(input_img))
# #     result_img.show()
#
#
# if __name__ == "__main__" :
#     # resnet 50 -> 90.20%
#     # f -> 89.40%
#     # model.load_state_dict(torch.load("./temp/best_resnet.pt", map_location=device))
#
#     # model.load_state_dict(torch.load("./best_resnet34_8.pt", map_location=device))
#     # test(model, test_loader, device)
#
#     train(model, criterion,train_loader, val_loader, optimizer, scheduler=exp_lr_scheduler, device=device)
#
#     # test_image(model, './dataset/test/Cass_7.png', device, test_transform)

import copy
import os.path
import os
import sys
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from customdataset import my_dataset
from torchvision import models
from timm.loss import LabelSmoothingCrossEntropy
from tqdm import tqdm

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = A.Compose([
        A.SmallestMaxSize(max_size=160),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.09,
                            rotate_limit=25, p=0.6),
        A.Resize(width=224, height=224),
        A.RandomBrightnessContrast(p=0.6),
        A.VerticalFlip(p=0.6),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    val_transform = A.Compose([
        A.SmallestMaxSize(max_size=160),
        A.Resize(width=224, height=224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    test_transform = A.Compose([
        A.SmallestMaxSize(max_size=160),
        A.Resize(width=224, height=224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    train_dataset = my_dataset("./dataset/train/", transform=train_transform)
    val_dataset = my_dataset("./dataset/val/", transform=val_transform)
    # test_dataset = my_dataset("./dataset/test/", transform=test_transform)

    def visulize_augmentations(dataset, samples=4, cols=2):
        dataset = copy.deepcopy(dataset)
        dataset.transform = A.Compose([t for t in dataset.transform
                                        if not isinstance(
                t, (A.Normalize, ToTensorV2)
            )])
        rows = samples // cols
        _, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
        for i in range(samples):
            image, _ = dataset[10]
            ax.ravel()[i].imshow(image)
            ax.ravel()[i].set_axis_off()
        plt.tight_layout()
        plt.show()

    # test = visulize_augmentations(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)


    # model = models.vgg19()
    # model.head = nn.Linear(in_features=4096, out_features=6)
    # model.to(device)
    # net = models.efficientnet_b0(pretrained=True)
    # net.classifier[1] = nn.Linear(in_features=1280, out_features=3)
    # net.to(device)
    # model = torchvision.models.resnet34(pretrained=True)
    # model.fc = torch.nn.Linear(in_features=512, out_features=6)
    # model.to(device)
    model = torch.hub.load('facebookresearch/deit:main',
                           'deit_tiny_patch16_224', pretrained=False)
    model.head = nn.Linear(in_features=192, out_features=6)
    model.to(device)

    loss_function = LabelSmoothingCrossEntropy()
    # optimizer = torch.optim.RMSprop(net.parameters(), lr=0.001)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)  # net의 파라메타를 넣어줘야 한다.
    # lr scheduler
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30,
                                                       gamma=0.1)
    epochs = 20

    best_val_acc = 0.0

    train_steps = len(train_loader)
    val_steps = len(val_loader)
    save_path = "./best_resnet34_4.pt"
    dfForAccuracy = pd.DataFrame(index=list(range(epochs)),
                                    columns=["Epoch", "Accurascy"])
    if os.path.exists(save_path):
        best_val_acc = max(pd.read_csv("./modelAccuracy4.csv")["Accuracy"].tolist())

    for epoch in range(epochs):
        runing_loss = 0
        val_acc = 0
        train_acc = 0

        # net.train()
        model.train()
        train_bar = tqdm(train_loader, file=sys.stdout, colour='green')
        for step, data in enumerate(train_bar):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = loss_function(outputs, labels)

            scheduler.step()
            optimizer.zero_grad()
            train_acc += (torch.argmax(outputs, dim=1) == labels).sum().item()
            loss.backward()
            optimizer.step()
            runing_loss += loss.item()

            train_bar.desc = f"train epoch [{epoch + 1}/{epochs}], loss >> {loss.data:.3f}"

        model.eval()
        with torch.no_grad():
            val_loss = 0
            valid_bar = tqdm(val_loader, file=sys.stdout, colour='red')
            for data in valid_bar:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_function(outputs, labels)
                val_loss += loss.item()
                val_acc += (torch.argmax(outputs, dim=1) == labels).sum().item()

        val_accuracy = val_acc / len(val_loader)
        train_accuracy = train_acc / len(train_loader)

        dfForAccuracy.loc[epoch, "Epoch"] = epoch + 1
        dfForAccuracy.loc[epoch, "Train_Accuracy"] = round(train_accuracy, 3)
        dfForAccuracy.loc[epoch, "Train_Loss"] = round(runing_loss / train_steps, 3)
        dfForAccuracy.loc[epoch, "Val_Accuracy"] = round(val_accuracy, 3)
        dfForAccuracy.loc[epoch, "Val_Loss"] = round(val_loss / val_steps, 3)
        print(f"epoch [{epoch+1}/{epochs}] trian_loss{(runing_loss / train_steps):.3f} train_acc : {train_accuracy:.3f} val_acc : {val_accuracy:.3f}")

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), save_path)

        if epoch == epochs - 1:
            dfForAccuracy.to_csv("./modelAccuracy4.csv", index=False)

    torch.save(model.state_dict(), "./best_deit.pt")


if __name__ == '__main__':
    main()

