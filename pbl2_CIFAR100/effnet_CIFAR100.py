import torch
import torchvision
import torchvision.transforms as transforms
import timm
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import time



def main():
    # 디바이스 설정
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 데이터셋을 위한 변환 설정

    # Train Dataset Augmentation
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Test dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # EfficientNet에 맞는 이미지 크기 조정
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # CIFAR-100 훈련 및 테스트 데이터셋 로드
    train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                  download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                 download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    # EfficientNet 모델 로드 (예: 'efficientnet_b0') 및 디바이스로 이동
    model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=100).to(device)

    # 손실 함수 및 최적화 함수 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.0001)
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, weight_decay=0.00001, momentum=0.9)
    epochs = 30


    # 모델을 평가하는 함수
    def evaluate(model, test_loader, best_acc):

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

        val_acc = 100. * correct/total
        print(f'Current Accuracy: {val_acc}%')

        if val_acc > best_acc:
            print(f'New best model with accuracy {val_acc}% (previous best was {best_acc}%)')
            best_acc = val_acc
            torch.save(model.state_dict(), './CIFAR_effnetb4_best_result.pth')
        return best_acc

    # 모델을 학습하는 함수
    def train(model, criterion, optimizer, train_loader, epochs):
        best_acc = 0.0
        model.train()
        total_start_time = time.time()
        for epoch in range(epochs):
            epoch_start_time = time.time()
            running_loss = 0.0
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
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}, Duration: {epoch_duration:.2f}s")

            best_acc = evaluate(model, test_loader, best_acc)

            print(f"Epoch {epoch + 1} Best Accuracy: {best_acc}%")


        total_end_time = time.time()
        print(f"Total training time: {total_end_time - total_start_time:.2f}s")



    # 모델 학습 및 평가
    train(model, criterion, optimizer, train_loader, epochs)

    # 학습이 끝난 후 최고 성능 모델로 추론
    model.load_state_dict(torch.load('./CIFAR_effnetb4_best_result.pth'))
    evaluate(model, test_loader, best_acc=0)


if __name__ == '__main__':
    main()