import os
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import copy

from customdataset import customDataset


# train loop
def train(num_epoch, model, train_loader, val_loader, criterion, optimizer, save_dir, device):
    print("Start training.....")
    total = 0
    best_loss = 9999

    for epoch in range(num_epoch):
        for i, (imgs, labels) in enumerate(train_loader):
            img, label = imgs.to(device), labels.to(device)
            output = model(img)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, argmax = torch.max(output, 1)
            acc = (label == argmax).float().mean()

            total += label.size(0)

            if (i + 1) % 10 == 0:
                print("Epoch >> [{}/{}], step >> [{}/{}], Loss >> {:.4f}, acc >> {:.2f}%"
                      .format(epoch + 1, num_epoch, i + 1, len(train_loader), loss.item(), acc.item() * 100))
        avrg_loss, val_acc = validation(model, val_loader, criterion, device)

        if avrg_loss < best_loss: # 9999로 했기때문에 작으면 if문에 들어감
            print("Best pt save")
            best_loss = avrg_loss
            save_model(model, save_dir)

    save_model(model, save_dir, file_name="last.pt")

def validation(model, val_loader, criterion, device):
    print("val Start !!! ")
    model.eval() # 평가 모델로 전환해준다.
    with torch.no_grad():
        total = 0
        total_loss = 0
        cnt = 0
        batch_loss = 0
        correct = 0

        for i, (imgs, labels) in enumerate(val_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            batch_loss += loss.item()

            total += imgs.size(0)
            _, argmax = torch.max(outputs, 1)
            correct += (labels == argmax).sum().item()
            total_loss += loss.item()
            cnt += 1
    avrg_loss = total_loss / cnt # 평균 로스가 나오게 된다.
    val_acc = (correct / total * 100)
    print("Acc >> {:.2f} Average loss >> {:.4f}".format(
        val_acc, avrg_loss
    ))

    model.train()

    return avrg_loss, val_acc

def save_model(model, save_dir, file_name = "best.pt"): # default는 best.pt로 해둔다.
    output_path = os.path.join(save_dir, file_name)
    torch.save(model.state_dict(), output_path) # 이렇게 하면 저장을 해준다.


def visualize_augmentations(dataset, idx=800, cols=5) :
    dataset = copy.deepcopy(dataset)
    samples = 30
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(
        t, (A.Normalize, ToTensorV2)
    )])
    rows = samples // cols
    figure , ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12,6))

    for i in range(samples) :
        image , _ = dataset[idx]
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()

    plt.tight_layout()
    plt.show()


train_transform = A.Compose([
    A.SmallestMaxSize(max_size=256),
    A.Resize(height=224, width=224),
    A.RandomShadow(p=0.8),
    A.RandomFog(p=0.3),
    # A.RandomSnow(p=0.7),
    A.RandomBrightnessContrast(p=0.4),
    A.Rotate(25, p=0.8),
    A.ShiftScaleRotate(shift_limit=5, scale_limit=0.05,
                       rotate_limit=25, p=0.7),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

my_dataset = customDataset('./dataset/train/', transform=train_transform)
my_visualize_aug = visualize_augmentations(my_dataset)
