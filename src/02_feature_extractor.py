from src.utils.all_utils import read_yaml, create_directory
import argparse
import os
import logging
import pickle
from tensorflow.keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
from tqdm import tqdm


logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = 'logs'
os.makedirs(log_dir, exist_ok= True)
logging.basicConfig(filename=os.path.join(log_dir, "running_log.log"), level=logging.INFO, 
format=logging_str, filemode= 'a')


def extractor(img_path, model):
    img = image.load_img(img_path, target_size = (224,224))
    img_array = image.img_to_array(img)
    expanded_img = np.expand_dims(img_array, axis = 0) 
    preprocess_img = preprocess_input(expanded_img)

    result = model.predict(preprocess_img).flatten()
    return result


def feature_extractor(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    
    artifacts = config['artifacts']
    artifacts_dir = artifacts['artifacts_dir']
    pickle_format_data_dir = artifacts['pickle_format_data_dir']
    img_pickle_file_name = artifacts['img_pickle_file_name']

    img_pickle_file_name = os.path.join(artifacts_dir, pickle_format_data_dir, img_pickle_file_name)
    
    filenames = pickle.load(open(img_pickle_file_name, 'rb'))

    model_name = params['base']['BASE_MODEL']
    include_tops = params['base']['include_top']
    pooling = params['base']['pooling']

    model = VGGFace(model = model_name, include_top = include_tops, pooling = pooling, input_shape=(224,224,3))

    feature_extractor_dir = artifacts['feature_extraction_dir']
    extracted_feature_name = artifacts['extracted_features_name']

    feature_extraction_path = os.path.join(artifacts_dir, feature_extractor_dir)

    create_directory(dirs = [feature_extraction_path])

    feature_name = os.path.join(feature_extraction_path, extracted_feature_name)

    features = []

    for file in tqdm(filenames):
        features.append(extractor(file, model))
    
    pickle.dump(features, open(feature_name, 'wb'))

    
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', "-c", default= 'config/config.yaml')
    args.add_argument('--params', "-p", default= 'params.yaml')
    parsed_args = args.parse_args()

    try:
        logging.info(">>>>>>> stage2 started")
        feature_extractor(config_path= parsed_args.config, params_path= parsed_args.params)
        logging.info("stage2 completed >>>>>>>>>")

    except Exception as e:
        logging.exception(e)
        raise e

