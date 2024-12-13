import streamlit as st
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os
import torch
import numpy as np
from mymodel import Custom_ResNet
import building_info  

def save_uploaded_img(directory, file):
     if not os.path.exists(directory):
          os.makedirs(directory)
     with open(os.path.join(directory, file.name), 'wb') as f:
          f.write(file.getbuffer())
     return st.success('Image Uploaded')

st.title('UNIST! Where is it?')

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 기존 매핑 (class_to_idx)
class_to_idx = {
    '0': 0, '1': 1, '10': 2, '11': 3, '12': 4, '13': 5, '14': 6, '15': 7, '16': 8, '17': 9, '18': 10, '19': 11,
    '2': 12, '20': 13, '21': 14, '22': 15, '23': 16, '24': 17, '25': 18, '26': 19, '27': 20, '28': 21, '29': 22,
    '3': 23, '30': 24, '31': 25, '32': 26, '33': 27, '34': 28, '35': 29, '36': 30, '37': 31, '38': 32, '39': 33,
    '4': 34, '40': 35, '41': 36, '42': 37, '43': 38, '44': 39, '45': 40, '46': 41, '47': 42, '48': 43, '49': 44,
    '5': 45, '50': 46, '51': 47, '52': 48, '53': 49, '54': 50, '55': 51, '56': 52, '57': 53, '58': 54, '59': 55,
    '6': 56, '60': 57, '61': 58, '62': 59, '63': 60, '64': 61, '65': 62, '66': 63, '67': 64, '68': 65, '69': 66,
    '7': 67, '70': 68, '71': 69, '72': 70, '73': 71, '74': 72, '75': 73, '76': 74, '77': 75, '78': 76, '79': 77,
    '8': 78, '80': 79, '81': 80, '82': 81, '83': 82, '84': 83, '85': 84, '86': 85, '87': 86, '9': 87
}

# 매핑을 뒤집기
idx_to_class = {v: int(k) for k, v in class_to_idx.items()}

# 예측된 값 -> 원래 클래스 이름으로 변환
def get_original_class(predicted_label):
    """
    예측된 숫자를 원래 폴더 이름으로 변환
    :param predicted_label: 모델이 예측한 레이블 (int)
    :return: 원래 폴더 이름 (str)
    """
    return idx_to_class.get(predicted_label, "Unknown")  # 존재하지 않는 레이블이면 "Unknown" 반환




st.subheader('Upload your photo 📷 ')
img_file = st.file_uploader('► Upload Only UNIST Place ↓', type=['png', 'jpg', 'jpeg'])


if img_file is not None:
     # save_uploaded_img('input_img', img_file)

     # 모델 불러오기
     model = Custom_ResNet()

     # CUDA 환경 확인 및 map_location 설정
     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     model.load_state_dict(torch.load('./100_weights.pth', map_location=device))

     model = model.to(device)
     model.eval()  # 평가 모드로 설정


     # 이미지를 tensor로 변환
     img = Image.open(img_file).convert("RGB")
     img_tensor = test_transforms(img).unsqueeze(0).to(device)  # img to tensor
    
     # 모델을 통해 예측 수행
     with torch.no_grad():
          output_class = model(img_tensor)  # 모델 출력 (logits)
          # output_class = F.softmax(output_class, dim=1)
     predicted_class = torch.argmax(output_class, dim=1).item()  # 예측된 클래스 (label)
     print(predicted_class)
     # 제대로 된 class로 바꾸기
     predicted_class = get_original_class(predicted_class)
     print(predicted_class)

     # 예측된 클래스에 해당하는 건물 정보 가져오기
     building_label = building_info.get_building_label(predicted_class)
     building_name = building_info.get_building_name(building_label)

     # 대표 이미지 가져오기
     building_img_path = f'./images/{building_label}_1.jpg'
     ppmap_img_path = f'./images/{building_label}_2.jpg'

     # 건물 설명 가져오기
     building_description_path = f'./discription/{building_label}.txt'

     # 건물 설명 파일 읽기
     try:
          with open(building_description_path, 'r', encoding='utf-8') as file:
               building_description = file.read()
     except FileNotFoundError:
          building_description = "건물 설명을 찾을 수 없습니다."

     # 레이아웃 설정
     col1, col2 = st.columns([2, 3])

     with col1:
          st.header("Your Location") 
          st.write(building_name)
          tab1, tab2, tab3 = st.tabs(['Location', 'Place Image', 'Input Image'])
          # 지도 이미지 및 건물 이미지 표시
          try:
               map_img = Image.open(ppmap_img_path)
          except:
               map_img = Image.open('./images/test_map_img.png')  # 기본 지도 이미지
          try:
               place_img = Image.open(building_img_path)
          except:
               place_img = Image.open('./images/test_place_img.png')  # 기본 건물 이미지
          with tab1:
               st.image(map_img)
          with tab2:
               st.image(place_img)
          with tab3:
               st.image(img_file, caption="Input Image")

               

     with col2:
          st.header("Place Information")
          # 건물 이름과 설명 출력
          st.subheader(building_name)
          try:
               st.write(building_description)
          except:
               st.write("There is No Matched Building")