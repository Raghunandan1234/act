import os
import pickle
'''actors=os.listdir('data')
filenames=[]
for actor in actors:
    for file in os.listdir(os.path.join('data',actor)):
        filenames.append(os.path.join('data',actor,file))
print(filenames)
print(actors)
pickle.dump(filenames,open('filenames.pkl','wb'))'''
from tensorflow import keras
from keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import pickle
import tqdm
filenames=pickle.load(open('filenames.pkl','rb'))
model=VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')
def feature_extracto(img_path,model):
    img=image.load_img(img_path,target_size=(224,224))
    img_array=image.img_to_array(img)
    expanded_img=np.expand_dims(img_array,axis=0)
    preprocesses_img=preprocess_input(expanded_img)
    result=model.predict(preprocesses_img).flatten()
    return result
features=[]
for file in (filenames):
    features.append(feature_extracto(file,model))
    
pickle.dump(features,open('embedding.pkl','wb'))
