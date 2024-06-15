import os
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

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
        domain_avg = np.mean(domain_lengths) if domain_lengths else 0
        balanced_images = []

        for domain_imgs in domain_image_count.values():
            if len(domain_imgs) < domain_avg:
                weight = int(np.ceil(domain_avg / len(domain_imgs)))
                balanced_images.extend(domain_imgs * weight)
            else:
                balanced_images.extend(random.sample(domain_imgs, int(domain_avg)))

        self._print_label_counts(balanced_images)
        return balanced_images

    def _print_label_counts(self, images):
        domain_count = {}
        category_count = {}

        for img_path, domain_idx, category_idx in images:
            domain_count[domain_idx] = domain_count.get(domain_idx, 0) + 1
            category_count[category_idx] = category_count.get(category_idx, 0) + 1

        print("Domain counts after sampling:")
        for idx, count in domain_count.items():
            print(f"{next(key for key, value in self.domain_dict.items() if value == idx)}: {count}")

        print("Category counts after sampling:")
        for idx, count in category_count.items():
            print(f"{next(key for key, value in self.category_dict.items() if value == idx)}: {count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, domain_idx, category_idx = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, domain_idx, category_idx

def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset_path = 'C:/Users/USER/PycharmProjects/pythonProject/Office_Home/Splitted_OfficeHomeDataset/train'
    dataset = DomainCategoryDataset(root_dir=dataset_path, transform=transform, intermediation_sampling=True)

if __name__ == '__main__':
    main()
