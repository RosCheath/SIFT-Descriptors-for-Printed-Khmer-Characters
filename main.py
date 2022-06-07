import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import sklearn
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split

def read_data(label2id):
    X = []
    Y = []
    for label in os.listdir('Char-dataset/trainingset'):
        for img_file in os.listdir(os.path.join('Char-dataset/trainingset', label)):
            img1 = cv2.imread(os.path.join('Char-dataset/trainingset', label, img_file))
            # convert images to grayscale
            img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            X.append(img)
            Y.append(label2id[label])
    print('Total data train :',len(Y))
    return X, Y
# Label to id, used to convert string label to integer
label2id = {'1':0, '2':1, '3':2, '4':3, '5':4,'6':5,'7':6,'8':7,'9':8,'10':9,'11':10,'12':11,'13':12,'14':13,'15':14,'16':15,'17':16,'18':17,'19':18,'20':19,'21':20,'22':21,'23':22,'24':23,'25':24,'26':25,'27':26,'28':27,'29':28,'30':29,'31':30,'32':31,'33':32}
X, Y = read_data(label2id)

#Trich xuat dac trung
def extract_sift_features(X):
    image_descriptors = []
    sift = cv2.SIFT_create()

    for i in range(len(X)):
        kp, des = sift.detectAndCompute(X[i], None)
        image_descriptors.append(des)

    return image_descriptors

image_descriptors = extract_sift_features(X)

#Xay dung tu dien
all_descriptors = []
for descriptors in image_descriptors:
    if descriptors is not None:
        for des in descriptors:
            all_descriptors.append(des)

def kmeans_bow(all_descriptors, num_clusters):
    bow_dict = []
    kmeans = KMeans(n_clusters=num_clusters).fit(all_descriptors)
    bow_dict = kmeans.cluster_centers_
    return bow_dict

num_clusters = 100

# train model to sklean
if not os.path.isfile('Char-dataset/data_character.pkl'):
    BoW = kmeans_bow(all_descriptors, num_clusters)
    pickle.dump(BoW, open('Char-dataset/data_character.pkl', 'wb'))
    print('not enough data')
else:
    BoW = pickle.load(open('Char-dataset/data_character.pkl', 'rb'))
    print('have data in dictionary')

#Xay dung vecto dac trung tu dict
def create_features_bow(image_descriptors, BoW, num_clusters):
    X_features = []
    for i in range(len(image_descriptors)):
        features = np.array([0] * num_clusters)

        if image_descriptors[i] is not None:
            distance = cdist(image_descriptors[i], BoW)
            argmin = np.argmin(distance, axis=1)
            for j in argmin:
                features[j] += 1
        X_features.append(features)
    return X_features

X_features = create_features_bow(image_descriptors, BoW, num_clusters)


#Xay dung model
X_train = []
X_test = []
Y_train = []
Y_test = []
X_train, X_test, Y_train, Y_test = train_test_split(X_features, Y, test_size=0.1, random_state=0)

svm = sklearn.svm.SVC(kernel= 'linear' ,C = 1)
svm.fit(X_train, Y_train)

#Thu predict
img_test = cv2.imread('Char-dataset/image_test/24-1.jpg')
img = [img_test]
img_sift_feature = extract_sift_features(img)
img_bow_feature = create_features_bow(img_sift_feature, BoW, num_clusters)
img_predict = svm.predict(img_bow_feature)
# print(img_predict)

for key, value in label2id.items():
    if value == img_predict[0]:
        print('index prediction: ', key)

label = ['ក ', 'ខ ', 'គ ', 'ឃ ', 'ង ','ច ','ឆ ','ជ ','ឈ ','ញ ','ដ ','ឋ ','ឌ ','ឍ ','ណ ','ត ','ថ ','ទ ','ធ ','ន ','ប ','ផ ','ព ','ភ ','ម ','យ ','រ ','ល ','វ ','ស ','ហ ','ឡ ','អ ']
prediction = svm.predict(img_bow_feature)
print('your prediction :', label[prediction[0]])
#Accuracy
accuracy = svm.score(X_test, Y_test)
print("Accuracy :", accuracy)
#Show image
plt.imshow(img_test, 'gray'),plt.show()