#!/usr/bin/env python

'''
Affine invariant feature-based image matching sample.

This sample is similar to find_obj.py, but uses the affine transformation
space sampling technique, called ASIFT [1]. While the original implementation
is based on SIFT, you can try to use SURF or ORB detectors instead. Homography RANSAC
is used to reject outliers. Threading is used for faster affine sampling.

[1] http://www.ipol.im/pub/algo/my_affine_sift/

USAGE
  asift.py [--feature=<sift|surf|orb>[-flann]] [ <image1> <image2> ]

  --feature  - Feature to use. Can be sift, surf of orb. Append '-flann' to feature name
                to use Flann-based matcher instead bruteforce.

  Press left mouse button on a feature point to see its matching point.
'''

#http://www.robots.ox.ac.uk/~az/icvss08_az_bow.pdf
#http://www.cmap.polytechnique.fr/~yu/research/ASIFT/demo.html

from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
# import matplotlib.pyplot as pl
import cPickle as pickle

import numpy as np
import cv2

import feature_extractor as fe

class BoWClassifier:
    def __init__(self, mapping=None):
        self.mapping = mapping
        if mapping:
            self.reverse_mapping = {v: k for k, v in self.mapping.items()}

    def predict(self, data):
        return self.svm.predict(self._normalize_histogram(data))

    def predict_label(self, data):
        predicted = self.predict(data)[0]
        return self.reverse_mapping[predicted]

    def accuracy(self, data, labels):
        if len(data) != len(labels):
            raise Exception("Data size: " + str(len(data)) + " labels size:" + str(len(labels)))
        predicted = self.predict(data)
        labels = np.asarray(map(lambda x: self.mapping[x], labels))
        return np.sum([x == y for (x, y) in zip(predicted, labels)]) / float(len(data))


    def _normalize_histogram(self, data):
        data_clustered = []
        for cluster in data:
            histogram = Counter(self.km.predict(cluster)).values()
            sum_in_example = float(sum(histogram))
            histogram = map(lambda x: x / sum_in_example, histogram)
            data_clustered.append(histogram)

        print data_clustered
        return data_clustered

    def train(self, data, labels, nr_clusters=8):
        print "###### Training started ####### "
        labels = np.asarray(map(lambda x: mapping[x], labels))
        # print len(labels)
        stacked_data = np.vstack(data)
        self.km = KMeans(init='k-means++', n_clusters=nr_clusters, n_init=2, max_iter=20).fit(stacked_data)
        print "###### Kmeans ####### "
        self.svm = OneVsRestClassifier(LinearSVC()).fit(self._normalize_histogram(data), labels)


    def save(self, filename="model.p"):
        # pickle.dump({"km": self.km, "svm": self.svm, "mapping": self.mapping}, open(filename, 'wb'))
        pass

    def load(self, filename="model.p"):
        tmp_dict = pickle.load(open(filename, 'rb'))
        self.km = tmp_dict["km"]
        self.svm = tmp_dict["svm"]
        self.mapping = tmp_dict["mapping"]
        self.reverse_mapping = {v: k for k, v in self.mapping.items()}

    def predict_labels(self, data):
        predicted = self.predict(data)
        return [self.mapping[prediction] for prediction in predicted]


import sys


def load_features(all_features):
    feature_files = None
    with open(all_features) as f:
        feature_files = "".join(f.readlines()).split()
    data = []
    for feature_file in feature_files:
        data.append(np.load(feature_file))
    return data


def load_labels(all_labels):
    labels = None
    with open(all_labels) as f:
        labels = "".join(f.readlines()).split();
    return labels


train_model = False
score_model = False

if __name__ == '__main__':

    # pool=ThreadPool(processes = cv2.getNumberOfCPUs())
    # kp1, desc1 = affine_detect(detector, img1, pool=pool)
    # #kp2, desc2 = affine_detect(detector, img2, pool=pool)
    # print 'img1: - %d features, desc1 - %d descriptors' % (len(kp1), len(desc1))
    #
    all_features_train = "local/all.sift"
    all_labels_train = "local/all.labels"
    all_features_test = "local/test/all.sift"
    all_labels_test = "local/test/all.labels"

    bow = BoWClassifier()
    if train_model:

        if len(sys.argv) == 0:
            all_features_train = sys.argv[1]
            all_labels_train = sys.argv[2]
            all_features_test = sys.argv[3]
            all_features_labels = sys.argv[4]

        data_train = load_features(all_features_train)
        labels_train = load_labels(all_labels_train)

        mapping = {}
        for (i, label) in zip(range(len(labels_train)), labels_train):
            if not label in mapping:
                mapping[label] = i

        bow = BoWClassifier(mapping)


        bow.train(data_train, labels_train)
        bow.save()

    else:
        bow.load()

    if score_model:
        data_test = load_features(all_features_test)
        labels_test = load_labels(all_labels_test)
        print bow.accuracy(data_test, labels_test)


    cap = cv2.VideoCapture(0)


    hello = "Hola Amigo!"
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = frame
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Display the resulting frame


        k = cv2.waitKey(1)
        if k & 0xFF == ord('q'):
            break
        if k & 0xFF == ord('r'):
            features = [fe.detect(frame)]
            # print features.shape
            hello = str(bow.predict_label(features))
        cv2.putText(gray,str(hello),(10,200), font, 4,(255,255,255))
        cv2.imshow('frame',gray)
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    # X = np.asarray([[0, 0], [1, 1], [2, 2], [3, 3]])
    # y = np.asarray([0, 1, 2, 3])
    # iris = datasets.load_iris()
    # print X[0]white rabbit jefferson airplane
    # OneVsOneClassifier(svm.SVC()).fit(X, Y).predict(X)