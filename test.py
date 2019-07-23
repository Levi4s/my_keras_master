#coding=utf-8

from PIL import Image
import numpy as np
import os
from keras.layers import Dense
from keras.preprocessing.image import load_img
from keras.preprocessing import image
from tqdm import tqdm
from keras.models import load_model

class Test():
    def load_model(self,model_type):
        if model_type == 'vgg16':
            model = load_model('model/vgg16_weights.h5')
        elif model_type == 'resnet50':
            model = load_model('model/restnet50_weights.h5')
        return model

    def main(self,model_type):
        model = self.load_model(model_type=model_type)
        with open(os.path.join('./result', '{}_result.txt').format(model_type), 'w') as f:
            for root, dirs, filenames in os.walk('./test'):
                for filename in tqdm(sorted(filenames, key=lambda x: int(x.split('.')[0][2])), desc='开始预测测试集'):
                    filepath = os.path.join('./test', filename)
                    img = load_img(filepath,target_size=(224, 224))
                    img = image.img_to_array(img) / 255.0
                    img = np.expand_dims(img, axis=0)
                    predictions = model.predict(img)
                    if predictions[0][0]>predictions[0][1]:
                        result = 'CREAK'
                    else:
                        result = 'HEL'
                    f.write('{} {}\n'.format(filename, result))

if __name__ == '__main__':
    print("网络结构：vgg16,resnet50")
    model_type = str(input("选择输入网络结构:"))
    test = Test()
    test.main(model_type=model_type)


