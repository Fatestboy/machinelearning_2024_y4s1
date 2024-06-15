import torch
import torchvision
import torchvision.transforms as transforms
import timm
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np


def main():
    # 디바이스 설정
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 데이터셋을 위한 변환 설정
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # EfficientNet에 맞는 이미지 크기 조정
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # CIFAR-100 훈련 및 테스트 데이터셋 로드
    train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                  download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                 download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    # EfficientNet-B7 모델 로드 및 디바이스로 이동
    model = timm.create_model('efficientnet_b7', pretrained=False, num_classes=100).to(device)

    # 손실 함수 및 최적화기 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    def train(model, criterion, optimizer, train_loader, epochs=10):
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

    def evaluate(model, test_loader):
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.view(-1).cpu().numpy())
                all_labels.extend(labels.view(-1).cpu().numpy())

        accuracy = 100 * correct / total
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')
        print(f'Accuracy: {accuracy}%')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')

        # 클래스별 이름 출력
        print("Classes in CIFAR-100: ")
        print(train_dataset.classes)

    # 모델 학습 및 평가
    train(model, criterion, optimizer, train_loader)
    evaluate(model, test_loader)


if __name__ == '__main__':
    main()