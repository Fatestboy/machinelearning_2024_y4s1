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
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_domain_acc = 0.0
best_category_acc = 0.0
print(f"Using device: {device}")

class MultiTaskModel(nn.Module):
    def __init__(self, num_domains=4, num_categories=65):
        super().__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0, features_only=True)
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
        domain_lengths = [len(v) for v in domain_image_count.values()]
        if not domain_lengths:  # 도메인이 없는 경우 예외 처리
            return []

        domain_avg = np.mean(domain_lengths) if np.any(domain_lengths) else 0  # 모든 도메인의 길이가 0이 아닐 때만 평균 계산

        if domain_avg == 0:
            return []  # 모든 도메인의 길이가 0인 경우 처리

        balanced_images = []
        for domain_imgs in domain_image_count.values():
            if len(domain_imgs) < domain_avg:
                weight = int(np.ceil(domain_avg / len(domain_imgs)))
                balanced_images.extend(domain_imgs * weight)
            else:
                balanced_images.extend(random.sample(domain_imgs, int(domain_avg)))

        category_count = {key: 0 for key in self.category_dict.values()}
        for _, _, category_idx in balanced_images:
            category_count[category_idx] += 1

        avg_samples = np.mean(list(category_count.values()))
        new_samples = []
        for category_idx in category_count:
            filtered_samples = [sample for sample in balanced_images if sample[2] == category_idx]

            if category_count[category_idx] == 0:
                continue  # 분모가 0이면 이 카테고리에 대해 샘플링하지 않음

            if category_count[category_idx] < avg_samples:
                weight = int(np.ceil(avg_samples / category_count[category_idx]))
                new_samples.extend(filtered_samples * weight)
            else:
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


def print_plot(num_epochs, train_losses, eval_losses, domain_accuracies, category_accuracies):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), eval_losses, label='Evaluation Loss')
    plt.title('EfficientNet_b0 Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()    

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), domain_accuracies, label='Domain Accuracy')
    plt.plot(range(1, num_epochs + 1), category_accuracies, label='Category Accuracy')
    plt.title('EfficientNet_b0 Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()

def save_best_model(model, domain_acc, category_acc):
    global best_domain_acc
    global best_category_acc
    if not os.path.exists('./best_model'):
        os.makedirs('./best_model')

    if best_domain_acc + best_category_acc < domain_acc + category_acc:
        best_domain_acc = domain_acc
        best_category_acc = category_acc
        torch.save(model.state_dict(), os.path.join('./best_model', './best_model.pth'))

def get_dataset():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])  

    dataset = DomainCategoryDataset(root_dir='./OfficeHomeDataset_10072016', transform=transform)   

    train_size = int(0.7 * len(dataset))
    eval_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - eval_size
    train_dataset, eval_dataset, test_dataset = random_split(dataset, [train_size, eval_size, test_size])   

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, eval_loader, test_loader

def train(train_loader, eval_loader, model, num_epochs = 10):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    eval_losses = []
    domain_accuracies = []
    category_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_eval_loss = 0
        correct_domain = 0
        correct_category = 0
        total = 0
        cnt = 0
        for images, domains, categories in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} Training'):
            images, domains, categories = images.to(device), domains.to(device), categories.to(device)
            optimizer.zero_grad()

            domain_preds, category_preds = model(images)

            loss_domain = criterion(domain_preds, domains)
            loss_category = criterion(category_preds, categories)
            loss = loss_domain + loss_category
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted_domains = torch.max(domain_preds, 1)
            _, predicted_categories = torch.max(category_preds, 1)

            correct_domain += (predicted_domains == domains).sum().item()
            correct_category += (predicted_categories == categories).sum().item()
            total += domains.size(0)
            domain_accuracy   = 100 * correct_domain / total
            category_accuracy = 100 * correct_category / total
            if cnt % 30 == 0.:
                print(f"\nTrain Loss: {total_loss / len(eval_loader):.2f}, Domain Acc: {domain_accuracy:.2f}%, Category Acc: {category_accuracy:.2f}%")
            cnt+=1
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {total_loss / len(train_loader)}")
########################################################################################################################################################
        # Evaluation
        model.eval()
        total_eval_loss = 0
        correct_domain = 0
        correct_category = 0
        total = 0
        for images, domains, categories in tqdm(eval_loader, desc=f'Epoch {epoch + 1} Evaluation'):
            images, domains, categories = images.to(device), domains.to(device), categories.to(device)
            domain_preds, category_preds = model(images)

            loss_domain = criterion(domain_preds, domains)
            loss_category = criterion(category_preds, categories)
            total_eval_loss += (loss_domain.item() + loss_category.item())

            # Calculate accuracy
            _, predicted_domains = torch.max(domain_preds, 1)
            _, predicted_categories = torch.max(category_preds, 1)

            correct_domain += (predicted_domains == domains).sum().item()
            correct_category += (predicted_categories == categories).sum().item()
            total += domains.size(0)
            

        avg_eval_loss = total_eval_loss / len(eval_loader)
        eval_losses.append(avg_eval_loss)

        domain_accuracy = 100 * correct_domain / total
        category_accuracy = 100 * correct_category / total

        domain_accuracies.append(domain_accuracy)
        category_accuracies.append(category_accuracy)

        domain_accuracy = 100 * correct_domain / total
        category_accuracy = 100 * correct_category / total
        print(f"\nEval Loss: {total_eval_loss / len(eval_loader)}, Domain Acc: {domain_accuracy:.2f}%, Category Acc: {category_accuracy:.2f}%")
        save_best_model(model, domain_accuracy, category_accuracy)
    print_plot(num_epochs, train_losses, eval_losses, domain_accuracies, category_accuracies)

def test(test_loader, model):
    criterion = nn.CrossEntropyLoss()

    model.eval()
    total_test_loss = 0
    correct_domain = 0
    correct_category = 0
    total = 0

    for images, domains, categories in tqdm(test_loader, desc='Testing'):
        images, domains, categories = images.to(device), domains.to(device), categories.to(device)
        domain_preds, category_preds = model(images)

        loss_domain = criterion(domain_preds, domains)
        loss_category = criterion(category_preds, categories)
        total_test_loss += (loss_domain.item() + loss_category.item())

        _, predicted_domains = torch.max(domain_preds, 1)
        _, predicted_categories = torch.max(category_preds, 1)

        correct_domain += (predicted_domains == domains).sum().item()
        correct_category += (predicted_categories == categories).sum().item()
        total += domains.size(0)

    avg_test_loss = total_test_loss / len(test_loader)

    domain_accuracy = 100 * correct_domain / total
    category_accuracy = 100 * correct_category / total

    print(f"Test Loss: {avg_test_loss}, Domain Acc: {domain_accuracy:.2f}%, Category Acc: {category_accuracy:.2f}%")


def main():
  train_loader, eval_loader, test_loader = get_dataset()
 
  model = MultiTaskModel().to(device)
  train(train_loader, eval_loader, model, num_epochs = 1)
  test(test_loader, model)
    
    
if __name__ =='__main__':
    main()