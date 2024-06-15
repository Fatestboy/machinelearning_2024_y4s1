import torch
import torchvision.transforms as transforms
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.models.segmentation as segmentation
import random
import functools
import os

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
    transforms.Lambda(lambda img: torch.from_numpy(target_transform_func(img)))
])

# VOCSegmentation 클래스 수정 (target을 텐서로 변환)
class CustomVOCSegmentation(VOCSegmentation):
    def __init__(self, root, year, image_set, download, transform=None, target_transform=None):
        super().__init__(root, year, image_set, download, transform=transform, target_transform=target_transform)
        self.transform = transform
        self.target_transform = target_transform
        self.segmentation_class_dir = os.path.join(root, 'VOCdevkit', 'VOC2012', 'SegmentationClass')

    def __getitem__(self, index):
        img_path = self.images[index]
        img_filename = os.path.basename(img_path).replace('.jpg', '.png')
        segmentation_class_path = os.path.join(self.segmentation_class_dir, img_filename)

        img = Image.open(img_path).convert('RGB')
        segmentation_class_img = Image.open(segmentation_class_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
            segmentation_class_img = self.transform(segmentation_class_img)

        return img, segmentation_class_img, img_path

def collate_fn(batch):
    images, targets, paths = zip(*batch)
    images = torch.stack(images, dim=0)
    targets = torch.stack(targets, dim=0)
    return images, targets, paths

val_dataset = CustomVOCSegmentation(root='./data', year='2012', image_set='val', download=False, transform=transform, target_transform=target_transform)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=functools.partial(collate_fn))

# 모델 로드 및 평가 모드 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = segmentation.deeplabv3_resnet101(weights=segmentation.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1)
#model = segmentation.deeplabv3_resnet101(weights=None, num_classes=21)
model.load_state_dict(torch.load('./saved_model/Resnet101_Crop4030.pth'))
model = model.to(device)
model.eval()

# 랜덤 이미지 선택 및 시각화
def visualize_random_samples(model, data_loader, device, num_samples=3):
    model.eval()
    samples = random.sample(range(len(data_loader.dataset)), num_samples)

    fig, axs = plt.subplots(num_samples, 3, figsize=(9, num_samples * 3))

    for i, sample in enumerate(samples):
        img, target, img_path = data_loader.dataset[sample]
        img = img.unsqueeze(0).to(device)
        target = target.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img)['out']
        pred = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

        axs[i, 0].imshow(img.squeeze().permute(1, 2, 0).cpu())
        axs[i, 0].set_title(f'Original Image\n{img_path}')
        axs[i, 0].axis('off')

        axs[i, 1].imshow(target.squeeze().permute(1, 2, 0).cpu())
        axs[i, 1].set_title('Segmentation Class Image')
        axs[i, 1].axis('off')

        axs[i, 2].imshow(pred)
        axs[i, 2].set_title('Predicted Segmentation')
        axs[i, 2].axis('off')

    plt.tight_layout()
    plt.show()

# 랜덤 샘플 시각화
visualize_random_samples(model, val_loader, device, num_samples=3)
