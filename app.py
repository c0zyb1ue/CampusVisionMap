import streamlit as st
from PIL import Image
import os

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

col1, col2 = st.columns([2, 3])

with col1:
     st.header("Your Location")
     st.checkbox('Image of pin-pointed map should be matched with AI Backend')
     tab1, tab2 = st.tabs(['Location', 'Place Image'])
     map_img = Image.open('./test_map_img.png')
     place_img = Image.open('./test_place_img.png')
     with tab1:
          st.image(map_img)
     with tab2:
          st.image(place_img)
with col2:
     st.header("Place Information")
     st.checkbox('Place information should be inplemented')

# Based on Classification Result from backend

col2.write("This Place is 106, Engineering building of Computer Science blah ~~")