import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
import json
import numpy as np
import timm
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 및 데이터셋 설정에 대한 가정
class MultiTaskModel_1(nn.Module):
    def __init__(self, num_domains=4, num_categories=65):
        super().__init__()
        self.base_model = timm.create_model('efficientnet_b1', pretrained=False, num_classes=0, features_only=True)
        feature_dim = self.base_model.feature_info.channels()[-1]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.domain_classifier = nn.Linear(feature_dim, num_domains)
        self.category_classifier = nn.Linear(feature_dim, num_categories)

    def forward(self, x):
        features = self.base_model(x)[-1]
        features = self.pool(features)
        features = torch.flatten(features, start_dim=1)
        domain_preds = self.domain_classifier(features)
        category_preds = self.category_classifier(features)
        return domain_preds, category_preds

class MultiTaskModel_2(nn.Module):
    def __init__(self, num_domains=4, num_categories=65):
        super().__init__()
        self.base_model = timm.create_model('efficientnet_b2', pretrained=False, num_classes=0, features_only=True)
        feature_dim = self.base_model.feature_info.channels()[-1]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.domain_classifier = nn.Linear(feature_dim, num_domains)
        self.category_classifier = nn.Linear(feature_dim, num_categories)

    def forward(self, x):
        features = self.base_model(x)[-1]
        features = self.pool(features)
        features = torch.flatten(features, start_dim=1)
        domain_preds = self.domain_classifier(features)
        category_preds = self.category_classifier(features)
        return domain_preds, category_preds


class OfficeHomeDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self._prepare_dataset()

    def _prepare_dataset(self):
        domain_index = 0
        self.domain_to_category_index = {}  # 도메인별 카테고리 인덱스 매핑

        for domain in os.listdir(self.root_dir):
            domain_path = os.path.join(self.root_dir, domain)
            # 도메인 이름에서 숫자 추출 또는 새 인덱스 할당
            if '_' in domain and domain.split('_')[1].isdigit():
                domain_idx = int(domain.split('_')[1])
            else:
                domain_idx = domain_index
                domain_index += 1

            category_index = 0  # 각 도메인마다 카테고리 인덱스를 0부터 시작

            for category in os.listdir(domain_path):
                category_path = os.path.join(domain_path, category)
                # 카테고리 이름에서 숫자 추출 또는 새 인덱스 할당
                parts = category.split('_')
                if len(parts) > 1 and parts[1].isdigit():
                    category_idx = int(parts[1])
                else:
                    # 도메인별로 별도의 카테고리 인덱스를 유지
                    if domain_idx not in self.domain_to_category_index:
                        self.domain_to_category_index[domain_idx] = category_index
                    else:
                        category_index = self.domain_to_category_index[domain_idx]
                    category_idx = category_index
                    category_index += 1
                    self.domain_to_category_index[domain_idx] = category_index

                for img_filename in os.listdir(category_path):
                    img_path = os.path.join(category_path, img_filename)
                    self.samples.append((img_path, domain_idx, category_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, domain_idx, category_idx = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, domain_idx, category_idx


def predict_and_save(models_1, models_2, dataset_1, dataset_2, output_path):
    data_loader_1 = DataLoader(dataset_1, batch_size=32, shuffle=False)
    data_loader_2 = DataLoader(dataset_2, batch_size=32, shuffle=False)

    with torch.no_grad(), open(output_path, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        # CSV 파일 헤더
        headers = []
        for i in range(len(models_1 + models_2)):
            headers += [f'Model {i + 1} Domain Prob {j + 1}' for j in range(4)]
            headers += [f'Model {i + 1} Category Prob {k + 1}' for k in range(65)]
        headers += ['Domain Label', 'Category Label']
        csv_writer.writerow(headers)

        for images, domain_labels, category_labels in data_loader_1:
            rows = [list() for _ in range(len(domain_labels))]  # 이미지 배치 크기만큼 행 리스트 초기화

            for model in models_1:
                domain_preds, category_preds = model(images.to(device))
                domain_probs = torch.softmax(domain_preds, dim=1).cpu().numpy()
                category_probs = torch.softmax(category_preds, dim=1).cpu().numpy()

                for i in range(len(domain_labels)):
                    rows[i].extend(domain_probs[i].tolist())
                    rows[i].extend(category_probs[i].tolist())

            for model in models_2:
                domain_preds, category_preds = model(images.to(device))
                domain_probs = torch.softmax(domain_preds, dim=1).cpu().numpy()
                category_probs = torch.softmax(category_preds, dim=1).cpu().numpy()

                for i in range(len(domain_labels)):
                    rows[i].extend(domain_probs[i].tolist())
                    rows[i].extend(category_probs[i].tolist())

            for i in range(len(domain_labels)):
                rows[i].extend([domain_labels[i].item(), category_labels[i].item()])
                csv_writer.writerow(rows[i])


# 모델 로딩
def load_models_1(paths):
    models = []
    for path in paths:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MultiTaskModel_1().to(device)
        model.load_state_dict(torch.load(path))
        model.eval()
        models.append(model)
    return models


def load_models_2(paths):
    models = []
    for path in paths:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MultiTaskModel_2().to(device)
        model.load_state_dict(torch.load(path))
        model.eval()
        models.append(model)
    return models


# 메인 실행부
if __name__ == "__main__":

    dataset_1 = OfficeHomeDataset(
        root_dir='C:/Users/USER/PycharmProjects/pythonProject/Office_Home/Splitted_OfficeHomeDataset_Origin/train',
        transform=transforms.Compose([
            transforms.Resize((240, 240)),
            transforms.ToTensor(),
        ])
    )

    dataset_2 = OfficeHomeDataset(
        root_dir='C:/Users/USER/PycharmProjects/pythonProject/Office_Home/Splitted_OfficeHomeDataset_Origin/train',
        transform=transforms.Compose([
            transforms.Resize((260, 260)),
            transforms.ToTensor(),
        ])
    )

    model_paths_1 = [
        'C:/Users/USER/PycharmProjects/pythonProject/Office_Home/best_model/effnetb1_30_unpadded_lowLR_0.0001.pth',
        'C:/Users/USER/PycharmProjects/pythonProject/Office_Home/best_model/effnetb1_30_midsample_unpadded_1.pth',
    ]
    model_paths_2 = [
        'C:/Users/USER/PycharmProjects/pythonProject/Office_Home/best_model/effnetb2_30_resized_Evaltest.pth',
        'C:/Users/USER/PycharmProjects/pythonProject/Office_Home/best_model/effnetb2_30_resized.pth',
    ]
    models_1 = load_models_1(model_paths_1)
    models_2 = load_models_2(model_paths_2)
    predict_and_save(models_1, models_2, dataset_1, dataset_2, 'effnet_train_prob.csv')