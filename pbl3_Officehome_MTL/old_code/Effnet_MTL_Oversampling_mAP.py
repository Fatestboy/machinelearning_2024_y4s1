import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
from torchvision import transforms
import timm
import os
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def compute_mAP(labels, scores, num_classes):
    """ Compute the mean Average Precision (mAP) """
    APs = []
    for i in range(num_classes):
        # 각 클래스에 대한 라벨과 점수를 추출
        class_labels = (labels == i)
        class_scores = scores[:, i]

        # 각 클래스별 AP 계산
        if class_labels.any():
            AP = average_precision_score(class_labels, class_scores)
            APs.append(AP)

    # 모든 클래스에 대한 AP의 평균을 계산
    mAP = np.mean(APs) if APs else 0
    return mAP

class DomainCategoryDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.domain_dict = {}
        self.category_dict = {}
        self._prepare_dataset()

    def _prepare_dataset(self):
        domain_index = 0
        category_index = 0
        for domain in os.listdir(self.root_dir):
            domain_path = os.path.join(self.root_dir, domain)
            if os.path.isdir(domain_path):
                if domain not in self.domain_dict:
                    self.domain_dict[domain] = domain_index
                    domain_index += 1
                for category in os.listdir(domain_path):
                    category_path = os.path.join(domain_path, category)
                    if os.path.isdir(category_path):
                        if category not in self.category_dict:
                            self.category_dict[category] = category_index
                            category_index += 1
                        for img_filename in os.listdir(category_path):
                            img_path = os.path.join(category_path, img_filename)
                            self.samples.append((img_path, self.domain_dict[domain], self.category_dict[category]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, domain_idx, category_idx = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, domain_idx, category_idx

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Test Dataset Load
test_dataset = DomainCategoryDataset(root_dir='C:/Users/USER/PycharmProjects/pythonProject/Office_Home/Splitted_OfficeHomeDataset/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Train Dataset Load
dataset = DomainCategoryDataset(root_dir='C:/Users/USER/PycharmProjects/pythonProject/Office_Home/Splitted_OfficeHomeDataset/train', transform=transform)
train_size = int(0.7 * len(dataset))
eval_size = len(dataset)-train_size
train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

# 오버샘플링 적용
domain_counts = np.zeros(len(dataset.domain_dict))
category_counts = np.zeros(len(dataset.category_dict))

for _, domain_idx, category_idx in dataset.samples:
    domain_counts[domain_idx] += 1
    category_counts[category_idx] += 1

domain_weights = 1.0 / domain_counts
category_weights = 1.0 / category_counts
samples_weights = [domain_weights[domain_idx] + category_weights[category_idx] for _, domain_idx, category_idx in dataset.samples]
samples_weights = torch.DoubleTensor(samples_weights)
sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

class MultiTaskModel(nn.Module):
    def __init__(self, num_domains=4, num_categories=65):
        super().__init__()
        self.base_model = timm.create_model('efficientnet_b1', pretrained=True, num_classes=0, features_only=True)
        # 마지막 특징 맵의 채널 수를 가져오기
        feature_dim = self.base_model.feature_info.channels()[-1]

        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # 전역 평균 풀링을 추가
        self.domain_classifier = nn.Linear(feature_dim, num_domains)
        self.category_classifier = nn.Linear(feature_dim, num_categories)

    def forward(self, x):
        features = self.base_model(x)[-1]  # 마지막 특징 맵을 사용
        features = self.pool(features)  # 평균 풀링 적용
        features = torch.flatten(features, start_dim=1)  # 평탄화

        domain_preds = self.domain_classifier(features)
        category_preds = self.category_classifier(features)
        return domain_preds, category_preds


model = MultiTaskModel().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

train_losses = []
eval_losses = []
domain_mAPs = []
category_mAPs = []

num_epochs = 1

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, domains, categories in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} Training'):
        images, domains, categories = images.to(device), domains.to(device), categories.to(device)
        optimizer.zero_grad()

        domain_preds, category_preds = model(images)

        loss_domain = criterion(domain_preds, domains)
        loss_category = criterion(category_preds, categories)
        loss = loss_domain + loss_category
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {total_loss / len(train_loader)}")

    # Evaluation
    model.eval()
    all_domain_labels = []
    all_domain_scores = []
    all_category_labels = []
    all_category_scores = []
    total_eval_loss = 0

    for images, domains, categories in tqdm(eval_loader, desc=f'Epoch {epoch + 1} Evaluation'):
        images, domains, categories = images.to(device), domains.to(device), categories.to(device)
        with torch.no_grad():
            domain_preds, category_preds = model(images)
            loss_domain = criterion(domain_preds, domains)
            loss_category = criterion(category_preds, categories)
            loss = loss_domain + loss_category
            total_eval_loss += loss.item()

        all_domain_labels.append(domains.cpu())
        all_domain_scores.append(torch.softmax(domain_preds, dim=1).cpu())
        all_category_labels.append(categories.cpu())
        all_category_scores.append(torch.softmax(category_preds, dim=1).cpu())

    # 리스트를 텐서로 변환
    all_domain_labels = torch.cat(all_domain_labels)
    all_domain_scores = torch.cat(all_domain_scores)
    all_category_labels = torch.cat(all_category_labels)
    all_category_scores = torch.cat(all_category_scores)

    # mAP 계산
    domain_mAP = compute_mAP(all_domain_labels, all_domain_scores, len(dataset.domain_dict))
    category_mAP = compute_mAP(all_category_labels, all_category_scores, len(dataset.category_dict))

    domain_mAPs.append(domain_mAP)
    category_mAPs.append(category_mAP)

    # 평균 손실 계산
    avg_eval_loss = total_eval_loss / len(eval_loader)
    eval_losses.append(avg_eval_loss)

    print(f"Eval Loss: {avg_eval_loss:.4f}, Domain mAP: {100 * domain_mAP:.2f}%, Category mAP: {100 * category_mAP:.2f}%")


def test(model, data_loader, criterion):
    model.eval()
    all_domain_labels = []
    all_domain_scores = []
    all_category_labels = []
    all_category_scores = []
    total_loss = 0

    with torch.no_grad():
        for images, domains, categories in tqdm(data_loader, desc="Testing"):
            images, domains, categories = images.to(device), domains.to(device), categories.to(device)
            domain_preds, category_preds = model(images)

            loss_domain = criterion(domain_preds, domains)
            loss_category = criterion(category_preds, categories)
            loss = loss_domain + loss_category
            total_loss += loss.item()

            all_domain_labels.append(domains.cpu())
            all_domain_scores.append(torch.softmax(domain_preds, dim=1).cpu())
            all_category_labels.append(categories.cpu())
            all_category_scores.append(torch.softmax(category_preds, dim=1).cpu())

    # 리스트를 텐서로 변환
    all_domain_labels = torch.cat(all_domain_labels)
    all_domain_scores = torch.cat(all_domain_scores)
    all_category_labels = torch.cat(all_category_labels)
    all_category_scores = torch.cat(all_category_scores)

    # mAP 계산
    domain_mAP = compute_mAP(all_domain_labels, all_domain_scores, len(test_dataset.domain_dict))
    category_mAP = compute_mAP(all_category_labels, all_category_scores, len(test_dataset.category_dict))

    avg_loss = total_loss / len(data_loader)

    return avg_loss, domain_mAP, category_mAP, all_domain_labels, all_domain_scores, all_category_labels, all_category_scores


# 테스트 실행
test_loss, test_domain_mAP, test_category_mAP, test_domain_labels, test_domain_scores, test_category_labels, test_category_scores = test(
    model, test_loader, criterion)
print(
    f"Test Loss: {test_loss:.4f}, Test Domain mAP: {100 * test_domain_mAP:.2f}%, Test Category mAP: {100 * test_category_mAP:.2f}%")

# 4. Precision-Recall 곡선 그리기
plt.figure(figsize=(10, 8))
for i in range(len(test_dataset.domain_dict)):
    precision, recall, _ = precision_recall_curve((test_domain_labels == i).numpy(), test_domain_scores[:, i].numpy())
    plt.plot(recall, precision, lw=2, label=f'Class {i}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Domain_Precision-Recall curve')
plt.legend(loc="best")
plt.show()

plt.figure(figsize=(10, 8))
for i in range(len(test_dataset.category_dict)):
    precision, recall, _ = precision_recall_curve((test_category_labels == i).numpy(), test_category_scores[:, i].numpy())
    plt.plot(recall, precision, lw=2, label=f'Class {i}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Category_Precision-Recall curve')
plt.legend(loc="best")
plt.show()

# epoch-loss, epoch-accuracy graph
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(eval_losses, label='Eval Loss')
plt.title('Loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot([100 * m for m in domain_mAPs], label='Domain mAP')
plt.plot([100 * m for m in category_mAPs], label='Category mAP')
plt.title('mAP over epochs')
plt.xlabel('Epochs')
plt.ylabel('mAP (%)')
plt.legend()

plt.tight_layout()
plt.show()