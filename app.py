import streamlit as st
from PIL import Image
import os
import torch
import numpy as np
from mymodel import Custom_ResNet
import building_info  # 이 부분은 이전에 작성한 building_info 모듈을 사용

def save_uploaded_img(directory, file):
     if not os.path.exists(directory):
          os.makedirs(directory)
     with open(os.path.join(directory, file.name), 'wb') as f:
          f.write(file.getbuffer())
     return st.success('Image Uploaded')

st.title('UNIST! Where is it?')

st.subheader('Upload your photo 📷 ')
img_file = st.file_uploader('► Upload Only UNIST Place ↓', type=['png', 'jpg', 'jpeg'])

if img_file is not None:
     save_uploaded_img('input_img', img_file)

     # 모델 불러오기
     model = Custom_ResNet()

     # CUDA 환경 확인 및 map_location 설정
     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     model.load_state_dict(torch.load('./train_weights.pth', map_location=device))

     model = model.to(device)
     model.eval()  # 평가 모드로 설정


     # 이미지를 numpy 배열로 변환하고 224x224 크기로 리사이즈
     img = Image.open(img_file).resize((224, 224))
     img = np.array(img).astype(np.float32) 
     img = torch.tensor(img).permute(2, 0, 1)  # C, H, W 순서로 바꾸기 (채널, 높이, 너비)
     img = img.unsqueeze(0)  # 배치 차원 추가 (1, C, H, W)

     # 모델을 통해 예측 수행
     with torch.no_grad():
          output_class = model(img)  # 모델 출력 (logits)
     predicted_class = torch.argmax(output_class, dim=1).item()  # 예측된 클래스 (label)
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
               map_img = Image.open('./test_map_img.png')  # 기본 지도 이미지
          try:
               place_img = Image.open(building_img_path)
          except:
               place_img = Image.open('./test_place_img.png')  # 기본 건물 이미지
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