from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dropout,Flatten,Dense
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras import backend
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')



class Train():
    def __init__(self,model_type):
        if model_type == 'resnet50':
            self.target_size = (224, 224)  # 图像将被resize成该尺寸
            self.weights_dir = 'model/restnet50_weights.h5'
        elif model_type == 'vgg16':
            self.target_size = (224,224)
            self.weights_dir = 'model/vgg16_weights.h5'
        self.train_dir = './data/train'  # 训练数据保存路径
        self.val_dir = './data/val'  # 验证数据保存路径
        self.test_dir = './data/test'  # 测试数据保存路径
        self.num = 2  # 多少分类

    # 图片读取与图片扩增和归一化处理
    def unit_img(self):
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True)
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        val_datagen = ImageDataGenerator(rescale=1. / 255)
        # 图片归一化处理
        train_flow = train_datagen.flow_from_directory(self.train_dir,  # 训练目录
                                                       self.target_size,  # 所有图像调整为
                                                       batch_size=8,
                                                       class_mode='categorical')  # 使用的损失函数 多分类设为categorical
        val_flow = val_datagen.flow_from_directory(self.val_dir, self.target_size, batch_size=8, class_mode='categorical')
        test_flow = test_datagen.flow_from_directory(self.test_dir, self.target_size, batch_size=8, class_mode='categorical')
        return train_flow,val_flow,test_flow

    def creat_model(self,model_type):
        if model_type == 'resnet50':
            base_model = ResNet50(include_top=True, weights=None)
            print(len(base_model.layers))
            base_model = Model(input=base_model.input, output=base_model.get_layer(index=175).output)
            x = base_model.output
            x = Dense(250,activation='relu')(x)
            predictions = Dense(self.num, activation='softmax')(x)
            model = Model(inputs=base_model.input, outputs=predictions)
            model.summary()
        elif model_type == 'vgg16':
            base_model = VGG16(include_top=True, weights=None)
            print(len(base_model.layers))
            base_model = Model(input=base_model.input, output=base_model.get_layer(index=21).output)
            x = base_model.output
            predictions = Dense(self.num, activation='softmax')(x)
            model = Model(inputs=base_model.input, outputs=predictions)
            model.summary()
        return model

    # 绘制训练 & 验证的损失值
    def show(self,history):
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

    #编译和训练模型
    def train(self,model_type):
        model = self.creat_model(model_type=model_type)
        model.compile(optimizer=SGD(lr=0.001, momentum=0.9, decay=0.0005),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        train_flow,val_flow,test_flow = self.unit_img()
        # 训练
        history = model.fit_generator(train_flow,
                                      steps_per_epoch=30,
                                      epochs=100,
                                      validation_data=val_flow,
                                      validation_steps=12,
                                      callbacks=[TensorBoard(log_dir='./logs')])
        test_loss, test_acc = model.evaluate_generator(test_flow, 96)
        print('test acc:', test_acc)
        # 保存模型和权重
        model.save(self.weights_dir)
        self.show(history=history)

if __name__ == '__main__':
    print("网络结构：vgg16,resnet50")
    model_type = str(input("选择输入网络结构:"))
    tr = Train(model_type=model_type)
    tr.train(model_type=model_type)







