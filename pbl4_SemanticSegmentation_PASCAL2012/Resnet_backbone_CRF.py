import torch
import torchvision.transforms as transforms
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import torchvision.models.segmentation as segmentation
import torch.optim as optim
import torch.nn as nn
from PIL import Image
import functools
from metrics import SegMetrics  # Import your custom metrics
import matplotlib.pyplot as plt
from torchcrf import CRF

# 데이터 변환 설정
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# 명시적인 함수로 변환
def target_transform_func(img):
    target = np.array(img, dtype=np.int64)
    return target


target_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=Image.NEAREST),
    transforms.Lambda(target_transform_func)
])


# VOCSegmentation 클래스 수정 (target을 텐서로 변환)
class CustomVOCSegmentation(VOCSegmentation):
    def __init__(self, root, year, image_set, download, transform=None, target_transform=None):
        super().__init__(root, year, image_set, download, transform=transform, target_transform=target_transform)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img = self.images[index]
        target = self.masks[index]

        img = Image.open(img).convert('RGB')
        target = Image.open(target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, torch.from_numpy(target).long()


# 명시적인 collate_fn 정의
def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    targets = torch.stack(targets, dim=0)
    return images, targets


# 데이터셋 로드, 첫 시작 시 transform = True, 그 후 False로 변경
train_dataset = CustomVOCSegmentation(root='./data', year='2012', image_set='train', download=False,
                                      transform=transform, target_transform=target_transform)
val_dataset = CustomVOCSegmentation(root='./data', year='2012', image_set='val', download=False, transform=transform,
                                    target_transform=target_transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4,
                          collate_fn=functools.partial(collate_fn))
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4,
                        collate_fn=functools.partial(collate_fn))


# CRF Segmentation Model 정의
class CRFSegmentationModel(nn.Module):
    def __init__(self, base_model, num_classes, filter_size=11, n_iter=5, smoothness_weight=1, smoothness_theta=1,
                 requires_grad=True, returns='logits'):
        super(CRFSegmentationModel, self).__init__()
        self.base_model = base_model
        self.crf = CRF(num_classes, batch_first=True)
        self.num_classes = num_classes
        self.filter_size = filter_size
        self.n_iter = n_iter
        self.smoothness_weight = smoothness_weight
        self.smoothness_theta = smoothness_theta
        self.requires_grad = requires_grad
        self.returns = returns

    def forward(self, x):
        features = self.base_model(x)['out']
        b, c, h, w = features.size()

        # Reshape features to (batch_size, height*width, num_classes)
        features = features.view(b, c, h * w).permute(0, 2, 1)

        # Reshape mask to (batch_size, height*width)
        mask = torch.ones((b, h * w), dtype=torch.bool).to(x.device)  # Assuming no padding

        # Apply CRF
        crf_output = self.crf.decode(features, mask=mask)

        # Reshape CRF output to original image dimensions
        crf_output = torch.tensor(crf_output).view(b, h, w)

        # Return logits or other format as specified
        if self.returns == 'logits':
            return features.permute(0, 2, 1).reshape(b, self.num_classes, h, w)
        return crf_output


# 학습 루프
def train(model, device, train_loader, val_loader, num_epochs, model_path):
    # 손실 함수와 옵티마이저 설정
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    # Poly Scheduler 설정
    def poly_lr_scheduler(epoch):
        return (1 - epoch / num_epochs) ** 0.9

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_lr_scheduler)

    best_miou = 0.0
    miou_history = []
    pa_history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, targets in tqdm(train_loader):
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()  # 학습률 스케줄러 업데이트
        train_loss = running_loss / len(train_loader)

        # Epoch별로 평가
        pa, miou = evaluate(model, val_loader, device)

        miou_history.append(miou)
        pa_history.append(pa)

        print(
            f'Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss}, Pixel Accuracy: {100 * pa:.3f}%, mIoU: {100 * miou:.3f}%')

        # 최상의 mIoU를 기록하고 모델 저장
        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), model_path)
            print(f'Model saved with mIoU: {100 * best_miou:.3f}%')

    return miou_history, pa_history


# 평가 함수
def evaluate(model, loader, device):
    model.eval()
    seg_metrics = SegMetrics(num_classes=21)

    with torch.no_grad():
        for images, targets in tqdm(loader):
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)

            seg_metrics.update(outputs, targets)
            pa, miou = seg_metrics.get_result()

    return pa, miou


def main():
    # 평가 실행
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # DeeplabV3+ 모델 로드
    base_model = segmentation.deeplabv3_resnet101(
        weights=segmentation.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1)

    # CRF 파라미터 설정
    filter_size = 11
    n_iter = 2
    smoothness_weight = 1
    smoothness_theta = 1
    requires_grad = True
    returns = 'logits'

    # CRF 모델 생성
    model = CRFSegmentationModel(base_model, num_classes=21, filter_size=filter_size, n_iter=n_iter,
                                 smoothness_weight=smoothness_weight, smoothness_theta=smoothness_theta,
                                 requires_grad=requires_grad, returns=returns)
    model = model.to(device)

    model_path = './saved_model/Resnet101_Pretrained.pth'
    num_epochs = 50

    miou_history, pa_history = train(model, device, train_loader, val_loader, num_epochs, model_path)
    # 최상의 모델 로드
    model.load_state_dict(torch.load(model_path))
    print('Loaded best model for final evaluation.')

    # 최종 평가
    pa, miou = evaluate(model, val_loader, device)
    print(f'Final Pixel Accuracy: {100 * pa:.3f}%, mIoU: {100 * miou:.3f}%')

    # mIoU 변화량 그래프 그리기
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), pa_history, linestyle='-', color='r', label='Pixel Accuracy')
    plt.plot(range(1, num_epochs + 1), miou_history, linestyle='-', color='b', label='mIOU')
    plt.title('mIoU and PA over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
