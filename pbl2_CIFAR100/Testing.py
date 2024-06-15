import torch
import torchvision
import torchvision.transforms as transforms
import timm
from torch.utils.data import DataLoader
from tqdm import tqdm


def main():

    # 모델 가중치 파일 경로
    MODEL_PATH = 'CIFAR_effnetb4_22_result.pth'

    # 데이터셋을 위한 전처리
    transform = transforms.Compose([
        transforms.Resize(224),  # EfficientNet-B2의 입력 크기에 맞게 조정
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # CIFAR100 데이터셋 로드
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

    # resnet50 모델 로드 및 가중치 적용
    model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=100)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # GPU 사용 설정
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 테스트 데이터셋에 대한 정확도 계산
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the CIFAR100 test images: {accuracy:.2f}%')

if __name__ == '__main__':
    main()
