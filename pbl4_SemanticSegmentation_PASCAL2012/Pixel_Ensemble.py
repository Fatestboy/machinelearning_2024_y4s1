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
from metrics import *
import matplotlib.pyplot as plt
from collections import Counter
import random

# 데이터 변환 설정
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def target_transform_func(img):
    target = np.array(img, dtype=np.int64)
    return target

target_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=Image.NEAREST),
    transforms.Lambda(target_transform_func)
])

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

def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    targets = torch.stack(targets, dim=0)
    return images, targets

train_dataset = CustomVOCSegmentation(root='./data', year='2012', image_set='train', download=False,
                                      transform=transform, target_transform=target_transform)
val_dataset = CustomVOCSegmentation(root='./data', year='2012', image_set='val', download=False, transform=transform,
                                    target_transform=target_transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4,
                          collate_fn=functools.partial(collate_fn))
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4,
                        collate_fn=functools.partial(collate_fn))

def load_models(model_paths, device):
    models = []
    for path in model_paths:
        model = segmentation.deeplabv3_resnet101(weights=segmentation.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1)
        model.load_state_dict(torch.load(path))
        model = model.to(device)
        model.eval()
        models.append(model)
    return models

def get_voting_prediction(models, image, device):
    predictions = []
    for model in models:
        output = model(image)['out']
        pred = torch.argmax(output, dim=1).cpu().numpy()
        predictions.append(pred)
    predictions = np.stack(predictions, axis=0)
    final_prediction = np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], axis=0, arr=predictions)
    return final_prediction, predictions


def evaluate_voting(models, loader, device):
    seg_metrics = SegMetrics(num_classes=21)
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(loader)):
            images = images.to(device)
            targets = targets.to(device)
            batch_predictions = []
            all_model_predictions = []
            for i, image in enumerate(images):
                image = image.unsqueeze(0)  # Add batch dimension
                final_prediction, model_predictions = get_voting_prediction(models, image, device)
                batch_predictions.append(final_prediction)
                all_model_predictions.append(model_predictions)

            if batch_idx == 0:  # Only visualize the first batch to avoid excessive plotting
                visualize_segmentation(images, targets, all_model_predictions, batch_predictions)

            batch_predictions = np.stack(batch_predictions, axis=0)
            batch_predictions = torch.from_numpy(batch_predictions).long().to(device)
            batch_predictions = batch_predictions.float()
            seg_metrics.update(batch_predictions, targets)

    return seg_metrics.get_result()


def visualize_segmentation(images, targets, predictions, final_predictions):
    indices = random.sample(range(len(images)), 2)  # 랜덤하게 2개의 인덱스 선택
    fig, axes = plt.subplots(2, len(predictions[0]) + 3, figsize=(20, 10))

    for j, idx in enumerate(indices):
        image = images[idx]
        target = targets[idx]
        final_prediction = final_predictions[idx]

        axes[j, 0].imshow(image.permute(1, 2, 0).cpu().numpy())
        axes[j, 0].set_title("Original Image")
        axes[j, 1].imshow(target.cpu().numpy())
        axes[j, 1].set_title("Ground Truth")

        for i in range(len(predictions[0])):  # 각 모델의 예측을 반복
            axes[j, i + 2].imshow(predictions[idx][i][0])  # assuming batch size of 1 for visualization
            axes[j, i + 2].set_title(f"Model {i + 1} Prediction")

        axes[j, len(predictions[0]) + 2].imshow(final_prediction[0])  # Remove the batch dimension
        axes[j, len(predictions[0]) + 2].set_title("Voting Prediction")

    for ax_row in axes:
        for ax in ax_row:
            ax.axis('off')

    plt.show()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 모델 경로 설정
    model_paths = ['./saved_model/Resnet101_Pretrained.pth', './saved_model/Resnet101_MixedLoss_Pretrain.pth']

    # 모델 로드
    models = load_models(model_paths, device)
    print('Loaded models for voting.')

    # 최종 평가
    pa, miou = evaluate_voting(models, val_loader, device)
    print(f'Final Pixel Accuracy (Voting): {100 * pa:.3f}%, mIoU (Voting): {100 * miou:.3f}%')

if __name__ == '__main__':
    main()
