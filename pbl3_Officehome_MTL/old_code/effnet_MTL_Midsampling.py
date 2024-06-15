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
import random
from sklearn.metrics import precision_recall_curve, average_precision_score
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
best_domain_acc = 0.0
best_category_acc = 0.0
save_filename = 'effnetb1_30_midsample'

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


# Precision-Recall 곡선 그리기
def precision_recall_graph(test_dataset, test_domain_labels, test_domain_scores, test_category_labels, test_category_scores):
    plt.figure(figsize=(10, 8))
    for i in range(len(test_dataset.domain_dict)):
        precision, recall, _ = precision_recall_curve((test_domain_labels == i).numpy(),
                                                      test_domain_scores[:, i].numpy())
        plt.plot(recall, precision, lw=2, label=f'Class {i}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Domain_Precision-Recall curve')
    plt.legend(loc="best")
    plt.show()

    plt.figure(figsize=(10, 8))
    for i in range(len(test_dataset.category_dict)):
        precision, recall, _ = precision_recall_curve((test_category_labels == i).numpy(),
                                                      test_category_scores[:, i].numpy())
        plt.plot(recall, precision, lw=2, label=f'Class {i}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Category_Precision-Recall curve')
    plt.legend(loc="best")
    plt.show()

# epoch-loss, epoch-accuracy graph

def print_mAP_plot(train_losses, eval_losses, domain_mAPs, category_mAPs):

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

class TestCategoryDataset(Dataset):
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

class DomainCategoryDataset(Dataset):
    def __init__(self, root_dir, transform=None, intermediation_sampling=True):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.domain_dict = {}
        self.category_dict = {}
        self.intermediation_sampling = intermediation_sampling
        self._prepare_dataset()

    def _prepare_dataset(self):
        domain_index = 0
        category_index = 0
        domain_image_count = {}
        image_count = []
        for domain in os.listdir(self.root_dir):
            domain_path = os.path.join(self.root_dir, domain)
            if os.path.isdir(domain_path):
                if domain not in self.domain_dict:
                    self.domain_dict[domain] = domain_index
                    domain_index += 1
                domain_images = []
                for category in os.listdir(domain_path):
                    category_path = os.path.join(domain_path, category)
                    if os.path.isdir(category_path):
                        # category_key = (self.domain_dict[domain], category)
                        if category not in self.category_dict:
                            self.category_dict[category] = category_index
                            category_index += 1
                        for img_filename in os.listdir(category_path):
                            img_path = os.path.join(category_path, img_filename)
                            image_details = (img_path, self.domain_dict[domain], self.category_dict[category])
                            domain_images.append(image_details)
                            image_count.append(image_details)
                domain_image_count[self.domain_dict[domain]] = domain_images

        if self.intermediation_sampling:
            self.samples = self._apply_intermediation_sampling(image_count, domain_image_count)
        else:
            self.samples = image_count

    def _apply_intermediation_sampling(self, image_count, domain_image_count):
        """
            중간 샘플링(intermediation sampling)을 수행하여 데이터 불균형을 처리

            Args:
            - image_count (list): 모든 이미지의 경로 및 해당 도메인 및 카테고리 인덱스를 포함하는 리스트
            - domain_image_count (dict): 각 도메인별 이미지 목록을 포함하는 사전

            Returns:
            - new_samples (list): 중간 샘플링을 적용한 새로운 데이터 샘플 리스트
        """

        # 각 도메인의 이미지 수를 측정하여 평균 계산
        domain_lengths = [len(v) for v in domain_image_count.values()]
        if not domain_lengths:  # 도메인이 없는 경우 예외 처리
            return []

        domain_avg = np.mean(domain_lengths) if np.any(domain_lengths) else 0
        # 모든 도메인의 길이가 0이 아닐 때만 평균 계산

        if domain_avg == 0:
            return []  # 모든 도메인의 길이가 0인 경우 처리

        balanced_images = []
        # 각 도메인에 대해 중간 샘플링 수행
        for domain_imgs in domain_image_count.values():
            # 도메인별 이미지 수가 평균 이하인 경우, 이미지를 복제하여 추가
            if len(domain_imgs) < domain_avg:
                weight = int(np.ceil(domain_avg / len(domain_imgs)))
                balanced_images.extend(domain_imgs * weight)
            else: # 도메인별 이미지 수가 평균 이상인 경우, 일부 이미지를 무작위로 선택하여 추가
                balanced_images.extend(random.sample(domain_imgs, int(domain_avg)))

        # 각 카테고리별 데이터 수를 측정하고, 평균 계산
        category_count = {key: 0 for key in self.category_dict.values()}
        for _, _, category_idx in balanced_images:
            category_count[category_idx] += 1

        avg_samples = np.mean(list(category_count.values()))

        # 각 카테고리에 대해 중간 샘플링 수행
        new_samples = []
        for category_idx in category_count:
            filtered_samples = [sample for sample in balanced_images if sample[2] == category_idx]

            if category_count[category_idx] == 0:
                continue  # 분모가 0이면 이 카테고리에 대해 샘플링하지 않음

            # 카테고리별 데이터 수가 평균 이하인 경우, 이미지를 복제하여 추가
            if category_count[category_idx] < avg_samples:
                weight = int(np.ceil(avg_samples / category_count[category_idx]))
                new_samples.extend(filtered_samples * weight)
            else: # 카테고리별 데이터 수가 평균 이상인 경우, 일부 이미지를 무작위로 선택하여 추가
                new_samples.extend(random.sample(filtered_samples, int(avg_samples)))

        return new_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, domain_idx, category_idx = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, domain_idx, category_idx

def save_best_model(model, domain_acc, category_acc):
    global best_domain_acc
    global best_category_acc
    global save_filename
    if not os.path.exists('./save_model'):
        os.makedirs('./save_model')

    if best_domain_acc + best_category_acc < domain_acc + category_acc:
        best_domain_acc = domain_acc
        best_category_acc = category_acc
        torch.save(model.state_dict(), os.path.join('./save_model', f'./{save_filename}.pth'))
        print(f"Saved best model Domain mAP : {100 * domain_acc:.2f}%, Category mAP : {100 * category_acc:.2f}%")

##########################################################################################################

def test(model, test_dataset, data_loader, criterion):
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

def test_saved_model(test_loader, test_dataset):
    # 모델 초기화 및 불러오기
    global save_filename
    model = MultiTaskModel().to(device)
    model.load_state_dict(torch.load(os.path.join(f'./best_model', f'./{save_filename}.pth')))
    model.eval()

    criterion = nn.CrossEntropyLoss()

    test_loss, test_domain_mAP, test_category_mAP, test_domain_labels, test_domain_scores, test_category_labels, test_category_scores = test(model, test_dataset, test_loader, criterion)

    precision_recall_graph(test_dataset, test_domain_labels, test_domain_scores, test_category_labels, test_category_scores)
    print(f"Test Loss: {test_loss:.4f}, Test Domain mAP: {100 * test_domain_mAP:.2f}%, Test Category mAP: {100 * test_category_mAP:.2f}%")

def get_dataset():
    # Train Dataset Augmentation

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Test Dataset

    Test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Test Dataset Load
    test_dataset = TestCategoryDataset(root_dir='C:/Users/USER/PycharmProjects/pythonProject/Office_Home/Splitted_OfficeHomeDataset/test', transform=Test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Train Dataset Load
    dataset = DomainCategoryDataset(root_dir='C:/Users/USER/PycharmProjects/pythonProject/Office_Home/Splitted_OfficeHomeDataset/train', transform=transform)
    train_size = int(0.7 * len(dataset))
    eval_size = len(dataset)-train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

    return train_loader, eval_loader, test_loader, test_dataset, dataset

##########################################################################################################

def train(dataset, train_loader, eval_loader, model, criterion, num_epochs = 30):
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    eval_losses = []
    domain_mAPs = []
    category_mAPs = []

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

    ##########################################################################################################

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
        save_best_model(model, domain_mAP, category_mAP)

    return train_losses, eval_losses, domain_mAPs, category_mAPs


# 테스트 실행
def main():
    train_loader, eval_loader, test_loader, test_dataset, dataset = get_dataset()

    model = MultiTaskModel().to(device)
    criterion = nn.CrossEntropyLoss()
    train_losses, eval_losses, domain_mAPs, category_mAPs = train(dataset, train_loader, eval_loader, model, criterion, num_epochs = 30)
    test_saved_model(test_loader, test_dataset)
    print_mAP_plot(train_losses, eval_losses, domain_mAPs, category_mAPs)

if __name__ =='__main__':
    main()
