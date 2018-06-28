import os
import numpy as np
import struct
from sklearn.preprocessing import label_binarize
def load_data(path,kind="train"):
    labels_path = os.path.join(path,"%s-labels.idx1-ubyte"%kind)
    images_path = os.path.join(path,"%s-images.idx3-ubyte"%kind)
    with open(labels_path,'rb') as labpath:
        magic_number,item_number = struct.unpack(">II",labpath.read(8))
        labels = np.fromfile(labpath,dtype=np.uint8)
    with open(images_path,'rb') as imgpath:
        magic_number,image_number,rows,cols =struct.unpack(">IIII",imgpath.read(16))
        images = np.fromfile(imgpath,dtype=np.uint8)
    return labels,images
orig_labels,orig_images = np.array(load_data("data/",kind="train"))
print("original labels shape is "+str(orig_labels.shape))
print("original images shape is "+str(orig_images.shape))
'''
开始处理数据，将数据输入处理成（784，60000）
                输出处理成（10，60000），手写数字有0-9共10个数字
'''
##labels = orig_labels.reshape(1,orig_labels.shape[0])
images = orig_images.reshape(60000,784)/255 ##注意此处是先将orig_image先reshape成（60000，784）若是改成（784，60000）就会变成乱码
images = images.T
'''
因为输出层是softmax，因此需要将labels先进行one-hot编码
'''
def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]

mid_labels = get_one_hot(orig_labels,10)  ##the mid_labels shape is (60000,10)
labels =mid_labels.T         ###the labels shape is (10,60000)

print("------------after tunning the images's shape and the labels's shape have changed------------------ ")
print("the images' shape is "+str(images.shape))
print("the labels' shape is "+str(labels.shape))


'''
load the test set
'''


print("-------------------load the test set---------------------------------------------------------------")
orig_test_labels,orig_test_images = np.array(load_data("data/",kind="t10k"))
print("original test labels shape"+str(orig_test_labels.shape))
print("original test images shape "+str(orig_test_images.shape))
test_images = orig_test_images.reshape(10000,784)/255
test_images = test_images.T

mid_test_labels = get_one_hot(orig_test_labels,10)
test_labels = mid_test_labels.T

print("the test-images-set shape is "+str(test_images.shape))
print("the test-labels-set shape is "+str(test_labels.shape))
'''
本例子线性神经网络结构为
                        输入层（784个输入维度）---隐藏层（num_hidden_units）----输出层（softmax，10）
                    x----(784,60000)
                    w1----(num_hidden_units,784)
                    b1----(num_hidden_units,1)
                    a1----(num_hidden_units,60000)
                    w2----(10,num_hidden_units)
                    b2----(10,1)
                    a2----(10,60000)
                    
'''
np.random.seed(1)
def initializer_with_hidden_layers(num_hidden_units):
    w1 = np.random.randn(num_hidden_units,784)
    b1 = np.zeros((num_hidden_units,1))
    w2 = np.random.randn(10,num_hidden_units)
    b2 = np.zeros((10,1))
    parameters={"w1":w1,
                "b1":b1,
                "w2":w2,
                "b2":b2}
    return parameters
'''
定以了两个激活函数，其中有
                sigmoid函数-----隐藏层
                softmax函数-----输出层
'''
def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s
def softmax(z):
    total = np.sum(np.exp(z),axis=0,keepdims=True)
    s = np.exp(z)/total
    return s
def forward_propagation(input_x,output_y,parameters):
    m = input_x.shape[1]
    w1 = parameters["w1"]
    b1 = parameters["b1"]
    w2 = parameters["w2"]
    b2 = parameters["b2"]
    a1 = sigmoid(np.dot(w1,input_x)+b1)
    a2 = softmax(np.dot(w2,a1)+b2)
    value_cost = -1/m*np.sum(output_y*np.log(a2))
    return a1,a2,value_cost
def backward_propagation(input_x,output_y,parameters,learning_rate,iterations):
    m = input_x.shape[1]
    w1 = parameters["w1"]
    b1 = parameters["b1"]
    w2 = parameters["w2"]
    b2 = parameters["b2"]
    for i in range(iterations):
        a1,a2,cost = forward_propagation(input_x,output_y,parameters)
        dz2 = a2-output_y
        dw2 = 1/m*np.dot(dz2,a1.T)
        db2 = 1/m*np.sum(dz2,axis=1,keepdims=True)
        dz1 = 1/m*np.dot(w2.T,dz2)*a1*(1-a1)
        dw1 = 1/m*np.dot(dz1,input_x.T)
        db1 = 1/m*np.sum(dz1,axis=1,keepdims=True)
        w1 = w1-learning_rate*dw1
        b1 = b1-learning_rate*db1
        w2 = w2-learning_rate*dw2
        b2 = b2-learning_rate*db2
        assert (w1.shape==dw1.shape)
        assert (b1.shape==db1.shape)
        assert (w2.shape==dw2.shape)
        assert (b2.shape==db2.shape)
        y_predict = np.eye(10)[np.array(a2.argmax(0))].T
        acc = 1-np.sum(np.abs(y_predict-output_y))/m
        if i%100==0:
            print("cost after iteration %i: %f"%(i,cost))
            print("accuracy is "+str(acc))
        parameters={"w1":w1,
                "b1":b1,
                "w2":w2,
                "b2":b2}
    return parameters
def predict(input_x,output_y,parameters):
    m = output_y.shape[1]
    _,y_hat,_ =forward_propagation(input_x,output_y,parameters)
    y_predict = np.eye(10)[np.array(y_hat.argmax(0))]
    return y_predict.T
def accuracy(y_predict,output_y):
    assert (y_predict.shape==output_y.shape)
    m = output_y.shape[1]
    acc  = 1-np.sum(np.abs(y_predict-output_y))/m
    return acc
def model(input_x,output_y,hidden_units,learning_rate,iterations):
    parameters = initializer_with_hidden_layers(hidden_units)
    parameters = backward_propagation(input_x,output_y,parameters,learning_rate,iterations)
    y_prediction = predict(input_x,output_y,parameters)
    acc = accuracy(y_prediction,output_y)
    print("the training-set accuracy is "+str(acc))
    return parameters
parameters = model(images,labels,hidden_units=784,learning_rate=0.45,iterations=2000)
test_y_prediction = predict(test_images,test_labels,parameters)
test_accuracy = accuracy(test_y_prediction,test_labels)
print("the testing-set accuracy is "+str(test_accuracy))




