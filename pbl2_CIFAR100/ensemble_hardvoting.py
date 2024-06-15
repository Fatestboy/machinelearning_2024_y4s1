import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import torch.nn as nn
import torch.nn.functional as F

# 데이터 로딩을 위한 변환 정의
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# CIFAR100 데이터셋 로딩
test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 모델 구조 생성
model1 = timm.create_model('efficientnet_b2', pretrained=False, num_classes=100)
model2 = timm.create_model('densenet161', pretrained=False, num_classes=100)
model3 = timm.create_model('resnext50_32x4d', pretrained=False, num_classes=100)
model4 = timm.create_model('resnet50', pretrained=False, num_classes=100)
model5 = timm.create_model('efficientnet_b3', pretrained=False, num_classes=100)
model6 = timm.create_model('efficientnet_b4', pretrained=False, num_classes=100)

# 사전 학습된 가중치 로드
model1.load_state_dict(torch.load('CIFAR_effnetb2_40_result.pth'))
model2.load_state_dict(torch.load('CIFAR_Densenet161_13_result.pth'))
model3.load_state_dict(torch.load('CIFAR_ResNext50_32x4d_20_result.pth'))
model4.load_state_dict(torch.load('CIFAR_resnet50_15_result.pth'))
model5.load_state_dict(torch.load('CIFAR_effnetb3_20_result.pth'))
model6.load_state_dict(torch.load('CIFAR_effnetb4_22_result.pth'))

# GPU 사용 설정 (가능한 경우)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model1 = model1.to(device)
model2 = model2.to(device)
model3 = model3.to(device)
model4 = model4.to(device)
model5 = model5.to(device)
model6 = model6.to(device)

# 하드 보팅 앙상블 예측
def ensemble_predict_hard_voting(model1, model2, model3, model4, model5, model6, loader):
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()
    model6.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs1 = model1(images)
            outputs2 = model2(images)
            outputs3 = model3(images)
            outputs4 = model4(images)
            outputs5 = model5(images)
            outputs6 = model6(images)
            _, predictions1 = torch.max(outputs1, 1)
            _, predictions2 = torch.max(outputs2, 1)
            _, predictions3 = torch.max(outputs3, 1)
            _, predictions4 = torch.max(outputs4, 1)
            _, predictions5 = torch.max(outputs5, 1)
            _, predictions6 = torch.max(outputs6, 1)
            # 가장 많이 나온 레이블 선택
            predictions = torch.mode(torch.stack([predictions1, predictions2, predictions3, predictions4, predictions5, predictions6]), dim=0)[0]
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
    print(f'Ensemble 5 models Hard Voting Accuracy: {100 * correct / total}%')

# 테스트 데이터셋에 대해 하드 보팅 앙상블 예측 수행
ensemble_predict_hard_voting(model1, model2, model3, model4, model5, model6, test_loader)
