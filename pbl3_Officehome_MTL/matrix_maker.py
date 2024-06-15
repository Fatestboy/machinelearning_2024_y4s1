import os
import numpy as np
from collections import defaultdict
import json
from torchvision import datasets, transforms

# 데이터셋의 훈련용 디렉토리 경로
data_root = 'C:/Users/USER/PycharmProjects/pythonProject/Office_Home/Splitted_OfficeHomeDataset_Origin/train/'
domains = ['Art', 'Clipart', 'Product', 'Real_World']
num_classes = 65

# 각 도메인에 대한 카테고리별 이미지 수를 저장할 딕셔너리
category_counts = {domain: defaultdict(int) for domain in domains}

# 이미지에 적용할 변환
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 데이터 로드 및 각 도메인에서 카테고리별 빈도 계산
for domain in domains:
    domain_path = os.path.join(data_root, domain)
    dataset = datasets.ImageFolder(domain_path, transform=transform)

    for _, label in dataset:
        category_counts[domain][label] += 1

# 모든 도메인의 모든 카테고리 수를 모아 전체 평균 계산
all_counts = []
for domain in domains:
    all_counts.extend([category_counts[domain][j] for j in range(num_classes)])

global_mean = np.mean(all_counts)

# 차이 행렬 초기화
diff_matrix = np.zeros((4, num_classes))

# 전체 평균 대비 각 도메인의 카테고리별 차이 계산
for i, domain in enumerate(domains):
    counts = np.array([category_counts[domain][j] for j in range(num_classes)])
    diff_matrix[i, :] = global_mean - counts

# JSON 형식으로 변환
json_data = {
    "regulateMat_5": diff_matrix.tolist()
}

# JSON 파일로 저장
output_file = "regulateMatrices_5.json"
with open(output_file, "w") as f:
    json.dump(json_data, f, indent=4)

print(f"JSON 데이터가 {output_file}에 저장되었습니다.")
