import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import sklearn
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn import metrics
import time


#read image path
def read_data(label2id):
    X = []
    Y = []
    for label in os.listdir('Char-dataset/trainingset'):
        for img_file in os.listdir(os.path.join('Char-dataset/trainingset', label)):
            img = cv2.imread(os.path.join('Char-dataset/trainingset', label, img_file))
            # # convert images to grayscale
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # # convert images to 200px
            # img = cv2.resize(img, (200, 200))
            X.append(img)
            Y.append(label2id[label])
    print('Total data :',len(Y))
    return X, Y

# Label to id, used to convert string label to integer
label2id = {'1':0, '2':1, '3':2, '4':3, '5':4,'6':5,'7':6,'8':7,'9':8,'10':9,'11':10,'12':11,'13':12,'14':13,'15':14,'16':15,'17':16,'18':17,'19':18,'20':19,'21':20,'22':21,'23':22,'24':23,'25':24,'26':25,'27':26,'28':27,'29':28,'30':29,'31':30,'32':31,'33':32}
label = ['ក ', 'ខ ', 'គ ', 'ឃ ', 'ង ','ច ','ឆ ','ជ ','ឈ ','ញ ','ដ ','ឋ ','ឌ ','ឍ ','ណ ','ត ','ថ ','ទ ','ធ ','ន ','ប ','ផ ','ព ','ភ ','ម ','យ ','រ ','ល ','វ ','ស ','ហ ','ឡ ','អ ']
X, Y = read_data(label2id)

# Get SIFT descriptors and keypoints of image.
def extract_sift_features(X):
    image_descriptors = []
    sift = cv2.SIFT_create()
    for i in range(len(X)):
        kp, des = sift.detectAndCompute(X[i], None)
        image_descriptors.append(des)
    return image_descriptors
image_descriptors = extract_sift_features(X)
all_descriptors = []
for descriptors in image_descriptors:
    if descriptors is not None:
        for des in descriptors:
            all_descriptors.append(des)

#Bag of Words using KMeans
def kmeans_bag_of_words(all_descriptors, num_clusters):
    # bag_of_words_dict = []
    kmeans = KMeans(n_clusters=num_clusters).fit(all_descriptors)
    bag_of_words_dict = kmeans.cluster_centers_
    return bag_of_words_dict

# train model to pickle
num_clusters = 1000
if not os.path.isfile('Char-dataset/data_character.pkl'):
    print('start Train Data : ple Wait ')
    start = time.time()
    Bag_of_Words = kmeans_bag_of_words(all_descriptors, num_clusters)
    pickle.dump(Bag_of_Words, open('Char-dataset/data_character.pkl', 'wb'))
    print('process time of train: ', time.time() - start)
else:
    Bag_of_Words = pickle.load(open('Char-dataset/data_character.pkl', 'rb'))
    print('have data in dictionary')

# Use bag of words
def create_features_bag_of_words(image_descriptors, Bag_of_Words, num_clusters):
    X_features = []
    for i in range(len(image_descriptors)):
        features = np.array([0] * num_clusters)
        if image_descriptors[i] is not None:
            distance = cdist(image_descriptors[i], Bag_of_Words)
            argmin = np.argmin(distance, axis=1)
            for j in argmin:
                features[j] += 1
        X_features.append(features)
    return X_features

#Sklean X and Y model
X_features = create_features_bag_of_words(image_descriptors, Bag_of_Words, num_clusters)
X_train, X_test, Y_train, Y_test = train_test_split(X_features, Y, test_size=0.25, random_state=42)
svm = sklearn.svm.SVC(kernel= 'linear', C=1, probability=True)
svm.fit(X_train, Y_train)

# # prediction data 30% of 100%
# predict = svm.predict(X_test)
# Accuracy1 = svm.score(X_train, Y_train)
# Accuracy2 = svm.score(X_test, Y_test)
# # Accuracy = metrics.accuracy_score(Y_test,predict)
# print('Total data Train :', len(X_train))
# print('Total data Test :', len(X_test))
# print('Accuracy data Train :', Accuracy1)
# print('Accuracy data Test :', Accuracy2)
# print(f"Accuracy data pre 30% : {metrics.accuracy_score(Y_test,predict)*100}")

#prediction of the input 1 image
img_test = cv2.imread('Char-dataset/testingset/1/1.jpg')
# convert images to grayscale
img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
# convert images to 200px
img_test = cv2.resize(img_test, (200, 200))
img = [img_test]
img_sift_descriptors = extract_sift_features(img)
img_bow_feature = create_features_bag_of_words(img_sift_descriptors, Bag_of_Words, num_clusters)
# accu probability

probability = svm.predict_proba(img_bow_feature)
for ind,val in enumerate(label):
    print(f'{val} = {probability[0][ind]*100}')

# #prediction of the input 1 image
# img_test = cv2.imread('Char-dataset/testingset/1/1.jpg')
# # convert images to grayscale
# img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
# # convert images to 200px
# img_test = cv2.resize(img_test, (200, 200))
# img = [img_test]
# img_sift_descriptors = extract_sift_features(img)
# img_bow_feature = create_features_bag_of_words(img_sift_descriptors, Bag_of_Words, num_clusters)
# img_predict = svm.predict(img_bow_feature)
# for key, value in label2id.items():
#     if value == img_predict[0]:
#         print('index prediction: ', key)
# print('your prediction : ', label[img_predict[0]])
# # print(score)
# # accuracy = svm.score(img_predict)
# # print("Accuracy : ", accuracy)
# #Show image
# # plt.title(label[prediction[0]])
# # plt.imshow(img_test, 'gray'), plt.show()
# # plt.axis('off')



# #prediction File Path
# for i in os.listdir('Char-dataset/image_test/'):
#     img_test = cv2.imread('Char-dataset/image_test/'+i)
#     # convert images to grayscale
#     img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
#     # convert images to 200px
#     img_test = cv2.resize(img_test, (200, 200))
#     img = [img_test]
#     img_sift_feature = extract_sift_features(img)
#     img_bow_feature = create_features_bag_of_words(img_sift_feature, Bag_of_Words, num_clusters)
#     img_predict = svm.predict(img_bow_feature)
#
#     for key, value in label2id.items():
#         if value == img_predict[0]:
#             print('index prediction:', key)
#     prediction = svm.predict(img_bow_feature)
#     print('your prediction:', label[prediction[0]])
# # #Accuracy
# #     accuracy = svm.score(X_test, Y_test)
# #     print("Accuracy : ", accuracy)
