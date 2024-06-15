import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader, ConcatDataset, Subset
import numpy as np
from tqdm import tqdm
import torchvision.models.segmentation as segmentation
import torch.optim as optim
import torch.nn as nn
from PIL import Image
import functools
import matplotlib.pyplot as plt
from metrics import SegMetrics
import segmentation_models_pytorch as smp
import random


# 데이터 변환 설정
class RandomAffinePair:
    def __init__(self, degrees, translate=None, scale=None, shear=None):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear

    def __call__(self, img, mask):
        # 같은 변환 파라미터를 사용하기 위해 random parameters를 얻음
        params = transforms.RandomAffine.get_params(
            self.degrees, self.translate, self.scale, self.shear, img.size)
        img = F.affine(img, *params, interpolation=Image.BILINEAR)
        mask = F.affine(mask, *params, interpolation=Image.NEAREST)
        return img, mask

class RandomFlipPair:
    def __init__(self, horizontal_flip_prob=0.5, vertical_flip_prob=0.5):
        self.horizontal_flip_prob = horizontal_flip_prob
        self.vertical_flip_prob = vertical_flip_prob

    def __call__(self, img, mask):
        if random.random() < self.horizontal_flip_prob:
            img = F.hflip(img)
            mask = F.hflip(mask)
        if random.random() < self.vertical_flip_prob:
            img = F.vflip(img)
            mask = F.vflip(mask)
        return img, mask

class RandomCropPair:
    def __init__(self, size, scale=(0.8, 1.0)):
        self.size = size
        self.scale = scale

    def __call__(self, img, mask):
        i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=self.scale, ratio=(1.0, 1.0))
        img = F.resized_crop(img, i, j, h, w, self.size, interpolation=Image.BILINEAR)
        mask = F.resized_crop(mask, i, j, h, w, self.size, interpolation=Image.NEAREST)
        return img, mask

class CustomVOCSegmentation(VOCSegmentation):
    def __init__(self, root, year, image_set, download, transform=None, target_transform=None, apply_affine=False, apply_flip=False, apply_crop=False):
        super().__init__(root, year, image_set, download, transform=transform, target_transform=target_transform)
        self.transform = transform
        self.target_transform = target_transform
        self.apply_affine = apply_affine
        self.apply_flip = apply_flip
        self.apply_crop = apply_crop
        if apply_affine:
            self.random_affine = RandomAffinePair(degrees=(-15, 15), translate=(0.1, 0.1), scale=(0.8, 1.2), shear=(-10, 10))
        if apply_flip:
            self.random_flip = RandomFlipPair(horizontal_flip_prob=0.15, vertical_flip_prob=0.15)
        if apply_crop:
            self.random_crop = RandomCropPair(size=(224, 224), scale=(0.8, 1.0))

    def __getitem__(self, index):
        img = self.images[index]
        target = self.masks[index]

        img = Image.open(img).convert('RGB')
        target = Image.open(target)

        if self.apply_affine:
            img, target = self.random_affine(img, target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, torch.from_numpy(target).long()

# 데이터 변환 설정
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Custom target transform function to convert PIL Image to numpy array
def target_transform_func(img):
    target = np.array(img, dtype=np.int64)
    return target

target_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=Image.NEAREST),
    transforms.Lambda(target_transform_func)
])

# 명시적인 collate_fn 정의
def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    targets = torch.stack(targets, dim=0)
    return images, targets

# 원본 데이터셋과 affine 변환된 데이터셋 생성
train_dataset = CustomVOCSegmentation(root='./data', year='2012', image_set='train', download=False,
                                      transform=transform, target_transform=target_transform)
affine_dataset = CustomVOCSegmentation(root='./data', year='2012', image_set='train', download=False,
                                             transform=transform, target_transform=target_transform, apply_affine=True, apply_flip=True)
# affine 데이터셋의 샘플링 크기 설정 (40%)
affine_sample_size = int(0.4 * len(train_dataset))
affine_indices = random.sample(range(len(affine_dataset)), affine_sample_size)
affine_subset = Subset(affine_dataset, affine_indices)

crop_sample_size = int(0.3 * affine_sample_size)
crop_indices = random.sample(range(len(affine_subset)), crop_sample_size)
crop_subset = Subset(CustomVOCSegmentation(root='./data', year='2012', image_set='train', download=False,
                                           transform=transform, target_transform=target_transform, apply_crop=True), crop_indices)

train_combined_dataset = ConcatDataset([train_dataset, affine_subset, crop_subset])

val_dataset = CustomVOCSegmentation(root='./data', year='2012', image_set='val', download=False, transform=transform,
                                    target_transform=target_transform)

train_loader = DataLoader(train_combined_dataset, batch_size=8, shuffle=True, num_workers=4,
                          collate_fn=functools.partial(collate_fn))
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4,
                        collate_fn=functools.partial(collate_fn))

# 학습 루프
def train(model, device, train_loader, val_loader, num_epochs, model_path, loss_weights=(0.75, 0.25)):
    ce_weight, dice_weight = loss_weights

    criterion_ce = nn.CrossEntropyLoss(ignore_index=255)
    criterion_dice = smp.losses.DiceLoss(mode='multiclass', ignore_index=255)

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
            outputs = model(images)['out']

            loss_ce = criterion_ce(outputs, targets)
            loss_dice = criterion_dice(outputs, targets)

            loss = ce_weight * loss_ce + dice_weight * loss_dice
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

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
            outputs = model(images)['out']

            seg_metrics.update(outputs, targets)
            pa, miou = seg_metrics.get_result()

    return pa, miou

def main():
    # 평가 실행
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # DeeplabV3+ 모델 로드
    # model = segmentation.deeplabv3_resnet101(weights=None, num_classes=21)
    model = segmentation.deeplabv3_resnet101(weights=segmentation.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1)
    model = model.to(device)

    model_path = './saved_model/Resnet101_Crop4030.pth'
    num_epochs = 50

    # 손실 함수 비율: CrossEntropy 75%, Dice 25%
    loss_weights = (0.75, 0.25)

    miou_history, pa_history = train(model, device, train_loader, val_loader, num_epochs, model_path, loss_weights)
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
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
