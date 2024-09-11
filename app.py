from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from src.utils.all_utils import read_yaml, create_directory
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image
import os
import cv2
from mtcnn import MTCNN
import numpy as np

config = read_yaml("config/config.yaml")
params = read_yaml("params.yaml")

artifacts = config['artifacts']
artifacts_dir = artifacts['artifacts_dir']

upload_image_dir = artifacts['upload_image_dir']
upload_path = os.path.join(artifacts_dir, upload_image_dir)

pickle_format_data_dir = artifacts['pickle_format_data_dir']
img_pickle_file_name = artifacts['img_pickle_file_name']
raw_local_dir_path = os.path.join(artifacts_dir, pickle_format_data_dir)
pickle_file = os.path.join(raw_local_dir_path, img_pickle_file_name)

feature_extractor_dir = artifacts['feature_extraction_dir']
extracted_feature_name = artifacts['extracted_features_name']

feature_extraction_path = os.path.join(artifacts_dir, feature_extractor_dir)
feature_name = os.path.join(feature_extraction_path, extracted_feature_name)

model_name = params['base']['BASE_MODEL']
include_tops = params['base']['include_top']
pooling = params['base']['pooling']


detector = MTCNN()
model = VGGFace(model = model_name, include_top = include_tops, pooling = pooling, input_shape=(224,224,3))
filenames = pickle.load(open(pickle_file, 'rb'))
feature_list = pickle.load(open(feature_name, 'rb'))


def save_upload_image(uploadimage):
    try:
        create_directory(dirs = [upload_path])
        with open(os.path.join(upload_path, uploadimage.name), 'wb') as f:
            f.write(uploadimage.getbuffer())

        return True
    except:
        return False

def extract_feature(img_path, model, detector):
    img = cv2.imread(img_path)
    result = detector.detect_faces(img)

    x,y,w,h = result[0]['box']
    face = img[y:y+h, x:x+w]

    image = Image.fromarray(face)
    image = image.resize((224,224))

    face_array = np.asarray(image)
    face_array = face_array.astype('float32')

    expanded_img = np.expand_dims(face_array, axis = 0) 
    preprocess_img = preprocess_input(expanded_img)
    result = model.predict(preprocess_img).flatten()

    return result

def recommend(feature_list, features):
    similarity = []

    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(features.reshape(1,-1), feature_list[i].reshape(1,-1))[0][0])

    index_pos = sorted(list(enumerate(similarity)), reverse=True, key = lambda x: x[1])[0][0]
    return index_pos

st.title('To whom does your face Match')
uploadimage = st.file_uploader('chose an image')

if uploadimage is not None:
    if save_upload_image(uploadimage):
        display_image = Image.open(uploadimage)

        features = extract_feature(os.path.join(upload_path, uploadimage.name), model, detector)

        indexpos = recommend(feature_list, features)
        predictor_actor = " ".join(filenames[indexpos].split('\\')[0].split('_'))

        col1, col2 = st.columns(2)

        with col1:
            st.header('Your uploaded image')
            st.image(display_image)
        
        with col2:
            st.header('Seems like' + predictor_actor)
            st.image(filenames[indexpos], width = 360)


