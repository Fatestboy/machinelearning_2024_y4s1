import torch
import torchvision
import torchvision.transforms as transforms
import timm
import time
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100

def main():

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # dense161 모델에 맞는 입력 크기 조정
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = CIFAR100(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = timm.create_model('densenet161', pretrained=True)  # 사전 학습된 모델 로드
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 100)  # CIFAR100의 100개 클래스에 맞게 조정

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 15

    best_accuracy = 0.0
    model_save_path = 'Densenet161_best_model.pth'
    total_start_time = time.time()
    for epoch in range(epochs):  # 에포크 수는 필요에 따라 조정
        model.train()
        running_loss = 0.0
        epoch_start_time = time.time()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time

        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')
        print(f"Duration {epoch_duration:.2f}s")

        # 평가 단계
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_accuracy = 100 * correct / total
        print(f'Accuracy of the model on the 10000 test images: {epoch_accuracy}%')

        # 현재 에포크의 모델이 이전보다 더 나은 정확도를 가질 경우 저장
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            torch.save(model.state_dict(), model_save_path)
            print(f'Model saved to {model_save_path} with accuracy: {best_accuracy}%')

    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    print(f"Total Duration {total_time:.2f}s")


if __name__ == '__main__':
    main()