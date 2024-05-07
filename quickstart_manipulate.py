import torch
from torchvision import transforms
from PIL import Image
import os
import math
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from templates import *
from templates_cls import *
from experiment_classifier import ClsModel

# 기존 설정 및 모델 초기화 부분은 동일하게 유지
device = 'cuda:0'
conf = ffhq256_autoenc()
model = LitModel(conf)
state = torch.load(f'checkpoints/{conf.name}/last.ckpt', map_location='cpu')
model.load_state_dict(state['state_dict'], strict=False)
model.ema_model.eval()
model.ema_model.to(device)

cls_conf = ffhq256_autoenc_cls()
cls_model = ClsModel(cls_conf)
state = torch.load(f'checkpoints/{cls_conf.name}/last.ckpt', map_location='cpu')
cls_model.load_state_dict(state['state_dict'], strict=False)
cls_model.to(device)

while True:
    # 이미지 파일 경로 입력받기
    image_name = input('Enter the number of the image:\t')
    image_path = f'imgs_align/20240503_{image_name}_capture.png'
    if not os.path.exists(image_path):
        print(f"Error: The file {image_path} does not exist.")
        exit()

    # 이미지를 로드하고 전처리
    transform = transforms.Compose([
        transforms.Resize(conf.img_size),  # 이미지 크기 조정
        transforms.CenterCrop(conf.img_size),  # 중앙에서 크롭
        transforms.ToTensor(),  # 텐서로 변환
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 정규화
    ])

    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)  # 배치 차원 추가 및 장치 할당

    # 인코딩 및 조건부 생성
    cond = model.encode(img_tensor)
    xT = model.encode_stochastic(img_tensor, cond, T=250)

    # 이미지 시각화
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ori = (img_tensor + 1) / 2
    ax[0].imshow(ori[0].permute(1, 2, 0).cpu())
    ax[1].imshow(xT[0].permute(1, 2, 0).cpu())

    print(CelebAttrDataset.id_to_cls)
    cls = input('Enter the attribute:\t')
    cls_id = CelebAttrDataset.cls_to_id[cls]

    cond2 = cls_model.normalize(cond)
    cond2 = cond2 + 0.3 * math.sqrt(512) * F.normalize(cls_model.classifier.weight[cls_id][None, :], dim=1)
    cond2 = cls_model.denormalize(cond2)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    img = model.render(xT, cond2, T=100)
    ax[0].imshow(ori[0].permute(1, 2, 0).cpu())
    ax[1].imshow(img[0].permute(1, 2, 0).cpu())

    plt.savefig(f'outputs/manipulated/{image_name}_{cls_id}_compare.png')
    save_image(img[0], f'outputs/manipulated/{image_name}_{cls_id}_output.png')
    
