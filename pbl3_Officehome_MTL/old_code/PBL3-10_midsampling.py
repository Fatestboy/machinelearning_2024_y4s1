import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
from torchvision import transforms
import timm
import os
from PIL import Image, ImageOps
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics import precision_recall_curve, average_precision_score
from collections import Counter
import cv2
import json
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

best_domain_acc = 0.0
best_category_acc = 0.0
save_filename = 'effnetb1_30_midsample_v2'

# 각 레이블 별 불균형 조정에 필요한 행렬 로드
with open('regulateMatrices_1.json', 'r') as json_file:
    matrices_1 = json.load(json_file)
with open('regulateMatrices_2.json', 'r') as json_file:
    matrices_2 = json.load(json_file)

regulateMat_1 = matrices_1['regulateMat_1']
regulateMat_2 = matrices_2['regulateMat_2']

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

def detect_edges(image_path):
    image = cv2.imread(image_path, 0)
    edges = cv2.Canny(image, 100, 200)
    return edges


def balancing_dataset():
    rootDir = 'C:/Users/USER/PycharmProjects/pythonProject/Office_Home/balance_per_domain/train'
    domainList = os.listdir(rootDir)
    for domainIdx, domainIter in enumerate(domainList):
        domainDir = os.path.join(rootDir, domainIter)
        categoryList = os.listdir(domainDir)
        for categoryIdx, categoryIter in enumerate(categoryList):
            dummyNum = int(regulateMat_2[domainIdx][categoryIdx])
            if dummyNum > 0:
                categoryDir = os.path.join(domainDir, categoryIter)
                if not os.path.exists(categoryDir):
                    break

                for fileIdx in range(dummyNum):
                    curFiles = os.listdir(categoryDir)
                    nextFileNumb = str(len(curFiles) + 1)
                    curImgDir = os.path.join(categoryDir, curFiles[fileIdx])
                    saveImgDir = os.path.join(categoryDir, '0' * (5 - len(nextFileNumb)) + nextFileNumb + '.jpg')
                    cv2.imwrite(saveImgDir, detect_edges(curImgDir))

class ResizeAndPad:
    def __init__(self, size, fill=255, padding_mode='edge'):  # 변환할 이미지의 최종크기(size), 패딩색상(fill), 패딩방법(padding_mode) 설정
        self.size = size
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        original_width, original_height = img.size  # 이미지의 원본 너비, 높이를 가져와서,
        ratio = min(self.size[0] / original_width, self.size[1] / original_height)  # 이미지가 목표 크기에 맞도록 축소되어야 할 비율을 계산한 후,
        new_width = int(original_width * ratio)  # 앞서 구한 ratio를 이용해 새로운 width를 계산하고
        new_height = int(original_height * ratio)  # 앞서 구한 ratio를 이용해 새로운 height를 계산하고
        img = F.resize(img, (new_height, new_width), Image.BICUBIC)  # 이미지를 새로운 크기로 재조정
        padding_left = (self.size[0] - new_width) // 2
        padding_top = (self.size[1] - new_height) // 2
        padding_right = self.size[0] - new_width - padding_left
        padding_bottom = self.size[1] - new_height - padding_top
        img = F.pad(img, (padding_left, padding_top, padding_right, padding_bottom), self.fill, self.padding_mode)
        return img  # 위에서 계산된 패딩을 이미지에 적용하고, self.fill을 이용해 패딩 영역의 색상 정하기. 이때, init에서 padding_mode=edge로 설정해서, 가장자리 색상을 이용해 패딩하게 됨

def apply_canny_edge_detection(img):  # canny 엣지 감지 이용해서 경계선 이미지 이용하는 class
    image_gray = np.array(img.convert('L'))  # 입력 받은 색을 grayscale 이미지로 변환해서 numpy 배열로 저장
    median_intensity = np.median(image_gray)  # 이미지 픽셀의 중앙값 계산
    lower_threshold = int(max(0, (1.0 - 0.33) * median_intensity))  # 중앙값의 0.67배를 하위 임계값으로 설정. 최소값은 0(하위임계값 이하의 edge는 무시하기위해)
    upper_threshold = int(min(255, (1.0 + 0.33) * median_intensity))  # 중앙값의 1.33배를 상위 임계값으로 설정. 최대값은 255(상위임계값 이상의 edge를 표시하기 위해 )
    image_canny = cv2.Canny(image_gray, lower_threshold, upper_threshold)
    return Image.fromarray(image_canny)

def create_transform():
    # 변형을 적용할 파이프라인 정의
    return transforms.Compose([
        ResizeAndPad((224, 224)),
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),  # 랜덤 크롭 및 크기 조정
        transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), shear=(-5, 5)),  # 회전, 시프트, 시어링
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 색상 조정
        transforms.RandomHorizontalFlip(),  # 50% 확률로 수평 뒤집기
        transforms.RandomVerticalFlip(),    # 50% 확률로 수직 뒤집기
        transforms.ToTensor(),  # 텐서로 변환
        # 필요에 따라 추가 변환
    ])

def augment_image(image_path):
    transform = create_transform()
    image = Image.open(image_path).convert('RGB')
    return transform(image)

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
        category_images = {}

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
                        images = [os.path.join(category_path, img) for img in os.listdir(category_path)]
                        category_images[(self.domain_dict[domain], self.category_dict[category])] = images

        # regulateMat을 사용하여 이미지 샘플링
        self.samples = self._apply_regulation(category_images)

    def _apply_regulation(self, category_images):
        new_samples = []
        for (domain_idx, category_idx), images in category_images.items():
            adjust_count = round(regulateMat_2[domain_idx][category_idx])
            current_count = len(images)

            if adjust_count > 0 and current_count > 0:
                # 데이터 증강을 통한 이미지 추가
                for _ in range(adjust_count):
                    img_path = random.choice(images)
                    aug_image = augment_image(img_path)  # 이 함수는 이미지를 텐서로 변환
                    aug_image_pil = to_pil_image(aug_image)  # 텐서를 PIL 이미지로 변환
                    save_path = f"{img_path[:-4]}_aug_{_}.jpg"
                    aug_image_pil.save(save_path, format='JPEG')
                    new_samples.append((save_path, domain_idx, category_idx))
            elif adjust_count < 0 and current_count + adjust_count > 0:
                # 이미지를 제거
                reduced_images = random.sample(images, current_count + adjust_count)
                new_samples.extend([(img, domain_idx, category_idx) for img in reduced_images])
            else:
                # 조정이 필요 없거나 조정 후 이미지 수가 0 미만이 되는 경우는 그대로 둠
                new_samples.extend([(img, domain_idx, category_idx) for img in images])

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
    if not os.path.exists('./best_model'):
        os.makedirs('./best_model')

    if best_domain_acc + best_category_acc < domain_acc + category_acc:
        best_domain_acc = domain_acc
        best_category_acc = category_acc
        torch.save(model.state_dict(), os.path.join('./best_model', f'./{save_filename}.pth'))
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
