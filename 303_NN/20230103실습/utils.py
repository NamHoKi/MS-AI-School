import os

import torch

def train(num_epoch, model, train_loader, val_loader, criterion, optimizer,
          save_dir, device):
    print("Start training.....")
    total = 0
    best_loss = 9999

    for epoch in range(num_epoch) :
        for i , (imgs, labels) in enumerate(train_loader) :
            img , label = imgs.to(device) , labels.to(device)
            output = model(img)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _,argmax = torch.max(output, 1)
            acc = (label == argmax).float().mean()

            total += label.size(0)

            if (i+1) % 10 == 0 :
                print("Epoch>> [{}/{}], step>> [{}/{}], Loss>> {:.4f}, acc>> "
                      "{:.2f}%".format(
                        epoch + 1,
                        num_epoch,
                        i + 1,
                        len(train_loader),
                        loss.item(),
                        acc.item() * 100
                ))
        avrg_loss, val_acc = validation(model, val_loader, criterion, device)
        if avrg_loss < best_loss :
            print("Best pt save")
            best_loss = avrg_loss
            save_model(model, save_dir)

    save_model(model, save_dir, file_name="last.pt")

def validation(model, val_loader, criterion, device) :
    print("val Start !!! ")
    model.eval()
    with torch.no_grad() :
        total = 0
        correct = 0
        total_loss = 0
        cnt = 0
        batch_loss = 0

        for i, (imgs, labels) in enumerate(val_loader) :
            imgs, labels = imgs.to(device) , labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            batch_loss += loss.item()

            total += imgs.size(0)
            _, argmax = torch.max(outputs, 1)
            correct += (labels == argmax).sum().item()
            total_loss += loss.item()
            cnt += 1

    avrg_loss = total_loss / cnt
    val_acc = (correct / total * 100)
    print("Acc >> {:.2f} Average loss >> {:.4f}".format(
        val_acc,
        avrg_loss
    ))

    model.train()

    return avrg_loss, val_acc


def save_model(model, save_dir, file_name ="best.pt"):
    output_path = os.path.join(save_dir, file_name)
    torch.save(model.state_dict(), output_path)


def acc_function(correct, total):
    acc = correct / total * 100
    return acc
def test_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, argmax = torch.max(outputs, 1)
            total += images.size(0)
            correct += (labels == argmax).sum().item()

        acc = acc_function(correct, total)
        print("accuracy for {} image : {:.2f}%".format(total, acc))