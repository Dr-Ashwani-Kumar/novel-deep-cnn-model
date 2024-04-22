from matplotlib import pyplot as plt
from scipy.stats import skew, kurtosis
from scipy.stats import hmean,gmean, entropy
from glob import glob
import cv2
import numpy as np
from keras.applications import ResNet101  # ResNet 101
from keras.layers import Dropout, Flatten, BatchNormalization, Activation, Dense
from keras import Model
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
import keras.utils
from Classifiers.CATBOOST import CATBOOST_classifier
from Classifiers.DCNN import DeepCNN
from Classifiers.DT import DT_classifier
from Classifiers.KNN import KNN_classifier
from Classifiers.LightGBM import lightgbm
from Classifiers.RForest import RF
from Classifiers.SVC import svc
from Classifiers.Xgboost import Xgboost

# resnet feature extraction
def main_resnet_feat_ext(base_model):
    ## Take Base model output
    x = base_model.output
    ## Apply Dropout Layer
    x = Dropout(0.05)(x)
    ## Apply Flatten Layer
    x = Flatten()(x)
    ## Apply BatchNormalizaton Layer
    x = BatchNormalization()(x)
    ## Apply Dense Layer
    x = Dense(1024, kernel_initializer='he_uniform')(x)
    ## Apply BatchNormalizaton Layer
    x = BatchNormalization()(x)
    ## Apply Activation Function
    x = Activation('relu')(x)
    ## Apply Dropout Layer
    x = Dropout(0.05)(x)
    ## Apply Dense Layer with units 1024
    x = Dense(1024, kernel_initializer='he_uniform')(x)
    ## Apply BatchNormalizaton Layer
    x = BatchNormalization()(x)
    ## Apply Activation Function
    x = Activation('relu')(x)
    ## Apply Dropout Layer
    x = Dropout(0.05)(x)
    ## Apply Dense Layer with units 1024
    x = Dense(1024, kernel_initializer='he_uniform')(x)
    ## Apply BatchNormalizaton Layer
    x = BatchNormalization()(x)
    ## Apply Activation Function
    x = Activation('relu')(x)
    ## Apply Dropout Layer
    x = Dropout(0.05)(x)

    predictions = Dense(100, activation='softmax')(x)

    model_feat = Model(inputs=base_model.input, outputs=predictions)
    return model_feat


def resnet_feature(image_array, input_size):
    ## ResNet101 Base Model
    base_model = ResNet101(include_top=False, weights='imagenet', input_shape=input_size)
    ## Feature Extraction Method
    Model_feat = main_resnet_feat_ext(base_model)
    ## Feature extraction for image array
    feat = Model_feat.predict(image_array)
    return feat

# statistical_feature extraction
def statistical_feature(img):
    mean1 = np.mean(img) ## Mean
    stdev = np.std(img)  ### Standard Deviation
    var1 = np.var(img)   ## Variation
    median1 = np.median(img)  ## Median
    skew1 = skew(img, axis=1, bias=False)  ## Skewness
    skew1 = np.mean(skew1) ## mean skewness
    kurtosis1 = kurtosis(img, axis=1, bias=False) ## Kurtosis
    kurtosis1 = np.mean(kurtosis1)  ## Mean Kurtosis
    hmean1 = hmean(img)  ## Harmonic Mean
    hmean1 = np.mean(hmean1)  ## Harmonic Mean
    gmean1 = gmean(img) ## Geometric Mean
    gmean1 = np.mean(gmean1)## Geometric Mean
    st_entropy = entropy(img)## Entropy
    st_entropy = np.mean(st_entropy)## Entropy Mean
    feat = np.hstack((mean1, stdev, var1, median1, skew1, kurtosis1, hmean1, gmean1, st_entropy))
    stat = feat.reshape(1, feat.shape[0])
    return stat


# Data preprocessing and feature extraction
def preprocessing_features_extraction(path):
    features = []
    label = []
    all_folders = glob(path + "/**")
    for i in range(len(all_folders)):
        all_images = glob(all_folders[i] + "/*.*")
        for filenames in all_images[:500]:
            print(i)
            image = cv2.imread(filenames)  # read an image
            dst = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)  # Preprocessing
            img = cv2.resize(dst, (32, 32))  # resize the preprocessed image
            img = img.reshape(1, img.shape[0],img.shape[1], img.shape[2])  # reshape the resized image
            resnet = resnet_feature(img, input_size=(img.shape[1], img.shape[2], img.shape[3]))  # resnet feature extraction
            statistical = statistical_feature(dst)  # statistical feature extraction
            feat1 = np.hstack((resnet, statistical))
            features.append(feat1)
            label.append(i)

    feat = np.vstack(features)
    lab = np.hstack(label)
    np.save("New Dataset\\Features.npy", feat)
    np.save("New Dataset\\Label.npy", lab)

# Analysis metrics
def main_est_perf_metrics(preds, y_test):
    ## Confusion Matrix
    cm = confusion_matrix(preds, y_test)
    ## True Positive
    TP = cm[0, 0]
    ## False Positive
    FP = cm[0, 1]
    ## False Negative
    FN = cm[1, 0]
    ## True Negative
    TN = cm[1, 1]
    ## Accuracy
    ACC = (TP + TN) / (TP + TN + FP + FN)
    ## Sensitivity
    SEN = TP / (TP + FN)
    ## Specificity
    SPE = TN / (TN + FP)
    return ACC, SEN, SPE

def Analysis():
    ## Load Features and Labels
    feat = np.load('New Dataset/Features1.npy')
    feat1 = (np.nan_to_num(feat, posinf=1e100, neginf=-1e100))
    lab = np.load('New Dataset/Label.npy')
    ## Training percentage
    tr = [0.4, 0.5, 0.6, 0.7, 0.8] ## 0.4:06,0.5:0.5,0.6:0.4,0.7:0.3,0.8:0.2
    epochs = [20, 40, 60, 80, 100] ## No.of Iterations
    opt = [0, 1, 2, 3] ## 0-Deepcnn Model, 1-HHO+DCNN,2-SSO+DCNN, 3-Proposed DCNN
    ACC, SEN, SPE = np.zeros((12, len(tr))), np.zeros((12, len(tr))), np.zeros((12, len(tr)))
    for i in range(len(tr)):
        # Split x and y into train and test
        X_train, X_test, y_train, y_test = train_test_split(feat1, lab, train_size=tr[i], random_state=42)
        print(" X_train, X_test, y_train, y_test :",  X_train.shape, X_test.shape, y_train.shape, y_test.shape)

        # Normalizing X_train & X_test
        X_train = X_train.astype(np.float32) / X_train.max()
        X_test = X_test.astype(np.float32) / X_test.max()

        # Convert into Hot Vector
        y1_train = keras.utils.to_categorical(y_train) # it will add no.of classes
        y1_test = keras.utils.to_categorical(y_test)

        preds_1, y_test_1 = KNN_classifier(X_train, y_train, X_test, y_test)  # KNN
        preds_2, y_test_2 = DT_classifier(X_train, y_train, X_test, y_test)  # DT
        preds_3, y_test_3 = RF(X_train, y_train, X_test, y_test)  # RF
        preds_4, y_test_4 = svc(X_train, y_train, X_test, y_test)  # SVC
        preds_5, y_test_5 = DeepCNN(X_train, y1_train, X_test, y1_test, epochs[0], opt[0])  # DCNN
        preds_6, y_test_6 = DeepCNN(X_train, y1_train, X_test, y1_test, epochs[0], opt[1])  # HHO Optimizer
        preds_7, y_test_7 = DeepCNN(X_train, y1_train, X_test, y1_test, epochs[0], opt[2])  # SSO Optimizer
        preds_8, y_test_8 = DeepCNN(X_train, y1_train, X_test, y1_test, epochs[0], opt[3])  # prop Optimizer
        preds_9, y_test_9 = DeepCNN(X_train, y1_train, X_test, y1_test, epochs[1], opt[3])  # performance 2
        preds_10, y_test_10 = DeepCNN(X_train, y1_train, X_test, y1_test, epochs[2], opt[3])  # performance 3
        preds_11, y_test_11 = DeepCNN(X_train, y1_train, X_test, y1_test, epochs[3], opt[3])  # performance 4
        preds_12, y_test_12 = DeepCNN(X_train, y1_train, X_test, y1_test, epochs[4], opt[3])  # performance 5

        # Following function is used to evaluate metrics accuracy, sensitivity, specificity
        ACC[0, i], SEN[0, i], SPE[0, i] = main_est_perf_metrics(preds_1, y_test_1)
        ACC[1, i], SEN[1, i], SPE[1, i] = main_est_perf_metrics(preds_2, y_test_2)
        ACC[2, i], SEN[2, i], SPE[2, i] = main_est_perf_metrics(preds_3, y_test_3)
        ACC[3, i], SEN[3, i], SPE[3, i] = main_est_perf_metrics(preds_4, y_test_4)
        ACC[4, i], SEN[4, i], SPE[4, i] = main_est_perf_metrics(preds_5, y_test_5)
        ACC[5, i], SEN[5, i], SPE[5, i] = main_est_perf_metrics(preds_6, y_test_6)
        ACC[6, i], SEN[6, i], SPE[6, i] = main_est_perf_metrics(preds_7, y_test_7)
        ACC[7, i], SEN[7, i], SPE[7, i] = main_est_perf_metrics(preds_8, y_test_8)
        ACC[8, i], SEN[8, i], SPE[8, i] = main_est_perf_metrics(preds_9, y_test_9)
        ACC[9, i], SEN[9, i], SPE[9, i] = main_est_perf_metrics(preds_10, y_test_10)
        ACC[10, i], SEN[10, i], SPE[10, i] = main_est_perf_metrics(preds_11, y_test_11)
        ACC[11, i], SEN[11, i], SPE[11, i] = main_est_perf_metrics(preds_12, y_test_12)

    np.save("Accuracy.npy", ACC)
    np.save("Sensitivity.npy", SEN)
    np.save("Specificity.npy", SPE)

def mean_parameter(acc, sen, spe):
    acc = np.mean(acc)
    sen = np.mean(sen)
    spe = np.mean(spe)
    return acc, sen, spe

def kfanalysis():
    ## Load Features and Labels
    X = np.load("New Dataset/Features1.npy")
    X = (np.nan_to_num(X, posinf=1e100, neginf=-1e100))
    y = np.load("New Dataset/Label.npy")
    ## KFold Values
    kr = [4, 5, 6, 7, 8]
    epochs = [20, 40, 60, 80, 100]  ## No.of Iterations
    opt = [0, 1, 2, 3]  ## 0-Deepcnn Model, 1-HHO+DCNN,2-SSO+DCNN, 3-Proposed DCNN
    tr =kr
    ACC, SEN, SPE = np.zeros((12, len(tr))), np.zeros((12, len(tr))), np.zeros((12, len(tr)))
    for w in range(len(kr)):
        i = w
        print(kr[w])
        from sklearn import preprocessing
        from ReliefF import ReliefF
        fs = ReliefF(n_neighbors=20, n_features_to_keep=500)
        # split Features and labels into Train and Test sets
        strtfdKFold = StratifiedKFold(n_splits=kr[w])
        kfold = strtfdKFold.split(X, y)
        ACC1, SEN1, SPE1 = np.zeros((1, kr[w])), np.zeros((1, kr[w])), np.zeros((1, kr[w]))
        ACC2, SEN2, SPE2 = np.zeros((1, kr[w])), np.zeros((1, kr[w])), np.zeros((1, kr[w]))
        ACC3, SEN3, SPE3 = np.zeros((1, kr[w])), np.zeros((1, kr[w])), np.zeros((1, kr[w]))
        ACC4, SEN4, SPE4 = np.zeros((1, kr[w])), np.zeros((1, kr[w])), np.zeros((1, kr[w]))
        ACC5, SEN5, SPE5 = np.zeros((1, kr[w])), np.zeros((1, kr[w])), np.zeros((1, kr[w]))
        ACC6, SEN6, SPE6 = np.zeros((1, kr[w])), np.zeros((1, kr[w])), np.zeros((1, kr[w]))
        ACC7, SEN7, SPE7 = np.zeros((1, kr[w])), np.zeros((1, kr[w])), np.zeros((1, kr[w]))
        ACC8, SEN8, SPE8 = np.zeros((1, kr[w])), np.zeros((1, kr[w])), np.zeros((1, kr[w]))
        ACC9, SEN9, SPE9 = np.zeros((1, kr[w])), np.zeros((1, kr[w])), np.zeros((1, kr[w]))
        ACC10, SEN10, SPE10 = np.zeros((1, kr[w])), np.zeros((1, kr[w])), np.zeros((1, kr[w]))
        ACC11, SEN11, SPE11 = np.zeros((1, kr[w])), np.zeros((1, kr[w])), np.zeros((1, kr[w]))
        ACC12, SEN12, SPE12 = np.zeros((1, kr[w])), np.zeros((1, kr[w])), np.zeros((1, kr[w]))

        for k, (train, test) in enumerate(kfold):
            tr_data = X[train, :]
            tr_data = tr_data[:, :]
            y_train = y[train]
            tst_data = X[test, :]
            tst_data = tst_data[:, :]
            y_test = y[test]
            X_train = tr_data
            X_test= tst_data

            X_train[X_train > 1e308] = 0
            X_test[X_test > 1e308] = 0

            # Normalizing X_train & X_test
            X_train = X_train.astype(np.float32) / X_train.max()
            X_test = X_test.astype(np.float32) / X_test.max()

            # Convert into Hot Vector
            y1_train = keras.utils.to_categorical(y_train)
            y1_test = keras.utils.to_categorical(y_test)

            preds_1, y_test_1 = KNN_classifier(X_train, y_train, X_test, y_test)  # KNN
            preds_2, y_test_2 = DT_classifier(X_train, y_train, X_test, y_test)  # DT
            preds_3, y_test_3 = RF(X_train, y_train, X_test, y_test)  # RF
            preds_4, y_test_4 = svc(X_train, y_train, X_test, y_test)  # SVC
            preds_5, y_test_5 = DeepCNN(X_train, y1_train, X_test, y1_test, epochs[0], opt[0])  # DCNN
            preds_6, y_test_6 = DeepCNN(X_train, y1_train, X_test, y1_test, epochs[0], opt[1])  # HHO Optimizer
            preds_7, y_test_7 = DeepCNN(X_train, y1_train, X_test, y1_test, epochs[0], opt[2])  # SSO Optimizer
            preds_8, y_test_8 = DeepCNN(X_train, y1_train, X_test, y1_test, epochs[0], opt[3])  # prop Optimizer
            preds_9, y_test_9 = DeepCNN(X_train, y1_train, X_test, y1_test, epochs[1], opt[3])  # performance 2
            preds_10, y_test_10 = DeepCNN(X_train, y1_train, X_test, y1_test, epochs[2], opt[3])  # performance 3
            preds_11, y_test_11 = DeepCNN(X_train, y1_train, X_test, y1_test, epochs[3], opt[3])  # performance 4
            preds_12, y_test_12 = DeepCNN(X_train, y1_train, X_test, y1_test, epochs[4], opt[3])  # performance 5

            # Following function is used to evaluate metrics accuracy, sensitivity, specificity
            ACC1[0, k], SEN1[0, k], SPE1[0, k] = main_est_perf_metrics(preds_1, y_test_1)
            ACC2[0, k], SEN2[0, k], SPE2[0, k] = main_est_perf_metrics(preds_2, y_test_2)
            ACC3[0, k], SEN3[0, k], SPE3[0, k] = main_est_perf_metrics(preds_3, y_test_3)
            ACC4[0, k], SEN4[0, k], SPE4[0, k] = main_est_perf_metrics(preds_4, y_test_4)
            ACC5[0, k], SEN5[0, k], SPE5[0, k] = main_est_perf_metrics(preds_5, y_test_5)
            ACC6[0, k], SEN6[0, k], SPE6[0, k] = main_est_perf_metrics(preds_6, y_test_6)
            ACC7[0, k], SEN7[0, k], SPE7[0, k] = main_est_perf_metrics(preds_7, y_test_7)
            ACC8[0, k], SEN8[0, k], SPE8[0, k] = main_est_perf_metrics(preds_8, y_test_8)
            ACC9[0, k], SEN9[0, k], SPE9[0, k] = main_est_perf_metrics(preds_9, y_test_9)
            ACC10[0, k], SEN10[0, k], SPE10[0, k] = main_est_perf_metrics(preds_10, y_test_10)
            ACC11[0, k], SEN11[0, k], SPE11[0, k] = main_est_perf_metrics(preds_11, y_test_11)
            ACC12[0, k], SEN12[0, k], SPE12[0, k] = main_est_perf_metrics(preds_12, y_test_12)

        ACC[0, i], SEN[0, i], SPE[0, i] = mean_parameter(ACC1, SEN1, SPE1)
        ACC[1, i], SEN[1, i], SPE[1, i] = mean_parameter(ACC1, SEN1, SPE1)
        ACC[2, i], SEN[2, i], SPE[2, i] = mean_parameter(ACC1, SEN1, SPE1)
        ACC[3, i], SEN[3, i], SPE[3, i] = mean_parameter(ACC1, SEN1, SPE1)
        ACC[4, i], SEN[4, i], SPE[4, i] = mean_parameter(ACC1, SEN1, SPE1)
        ACC[5, i], SEN[5, i], SPE[5, i] = mean_parameter(ACC1, SEN1, SPE1)
        ACC[7, i], SEN[7, i], SPE[7, i] = mean_parameter(ACC1, SEN1, SPE1)
        ACC[8, i], SEN[8, i], SPE[8, i] = mean_parameter(ACC1, SEN1, SPE1)
        ACC[9, i], SEN[9, i], SPE[9, i] = mean_parameter(ACC1, SEN1, SPE1)
        ACC[10, i], SEN[10, i], SPE[10, i] = mean_parameter(ACC1, SEN1, SPE1)
        ACC[11, i], SEN[11, i], SPE[11, i] = mean_parameter(ACC1, SEN1, SPE1)

    np.save("Acc.npy", ACC)
    np.save("Sen.npy", SEN)
    np.save("Spe.npy", SPE)

def Complete_Figure_com(perf, val, str_1, xlab, ylab):
    perf = np.sort(perf.T).T
    np.savetxt('Results/TP/Comp_Analysis\\' + '_' + str(val) + '_' + 'Graph.csv', perf, delimiter=",")
    n_groups = 5
    index = np.arange(n_groups)
    bar_width = 0.085
    opacity = 0.6
    plt.figure(val)
    plt.bar(index, perf[0, :], bar_width, alpha=opacity, edgecolor="black", color='#00ccff', label=str_1[0])
    plt.bar(index + bar_width, perf[1, :], bar_width, alpha=opacity, edgecolor="black", color='#0000ff', label=str_1[1])
    plt.bar(index + 2 * bar_width, perf[2, :], bar_width, alpha=opacity, edgecolor="black", color='#ff00ff', label=str_1[2])
    plt.bar(index + 3 * bar_width, perf[3, :], bar_width, alpha=opacity, edgecolor="black", color='#99ffcc', label=str_1[3])
    plt.bar(index + 4 * bar_width, perf[4, :], bar_width, alpha=opacity, edgecolor="black", color='#ff6600', label=str_1[4])
    plt.bar(index + 5 * bar_width, perf[5, :], bar_width, alpha=opacity, edgecolor="black", color='#00ffff', label=str_1[5])
    plt.bar(index + 6 * bar_width, perf[6, :], bar_width, alpha=opacity, edgecolor="black", color='#ff9999', label=str_1[6])
    plt.bar(index + 7 * bar_width, perf[7, :], bar_width, alpha=opacity, edgecolor="black", color='#00ff00', label=str_1[7])

    plt.xlabel(xlab, fontsize=10, fontweight='bold')
    plt.ylabel(ylab, fontsize=10, fontweight='bold')
    plt.xticks(index + 0.1, ('40', '50', '60', '70', '80'))
    plt.title("COMPARATIVE ANALYSIS")
    plt.legend(loc='lower right')
    plt.savefig('Results/TP/Comp_Analysis\\' + str(val) + '_' + 'Graph.png', dpi=800)
    plt.show(block=False)
    plt.clf()

def Complete_Figure_perf(perf, val, str_1, xlab, ylab):
    perf = np.sort(perf.T).T
    np.savetxt('Results/TP/Perf_Analysis\\' + '_' + str(val) + '_' + 'Graph.csv', perf, delimiter=",")
    n_groups = 5
    index = np.arange(n_groups)
    bar_width = 0.085
    opacity = 0.6
    plt.figure(val)
    plt.bar(index, perf[0, :], bar_width, alpha=opacity, edgecolor="black", color='#0099ff', label=str_1[0])
    plt.bar(index + bar_width, perf[1, :], bar_width, alpha=opacity, edgecolor="black", color='#ff00ff', label=str_1[1])
    plt.bar(index + 2 * bar_width, perf[2, :], bar_width, alpha=opacity, edgecolor="black", color='#ff9900', label=str_1[2])
    plt.bar(index + 3 * bar_width, perf[3, :], bar_width, alpha=opacity, edgecolor="black", color='#ff33cc', label=str_1[3])
    plt.bar(index + 4 * bar_width, perf[4, :], bar_width, alpha=opacity, edgecolor="black", color='#00ff99', label=str_1[4])

    plt.xlabel(xlab, fontsize=10, fontweight='bold')
    plt.ylabel(ylab, fontsize=10, fontweight='bold')
    plt.xticks(index + 0.1, ('40', '50', '60', '70', '80'))
    plt.title("PERFORMANCE ANALYSIS")
    plt.legend(loc='lower right')
    plt.savefig('Results/TP/Perf_Analysis\\' + str(val) + '_' + 'Graph.png', dpi=800)
    plt.show(block=False)
    plt.clf()

def load_parameters():
    A = np.load("Accuracy.npy")*100
    B = np.load("Sensitivity.npy")*100
    C = np.load("Specificity.npy")*100
    perf_A = A[[0, 1, 2, 3, 4, 5, 6, 11], :]
    perf_B = B[[0, 1, 2, 3, 4, 5, 6, 11], :]
    perf_C = C[[0, 1, 2, 3, 4, 5, 6, 11], :]
    A1 = A[[7, 8, 9, 10, 11], :]
    B1 = B[[7, 8, 9, 10, 11], :]
    C1 = C[[7, 8, 9, 10, 11], :]
    return [perf_A, perf_B, perf_C, A1, B1, C1]

def complete_plot(ii):
    [A, B, C, A1, B1, C1] = load_parameters()
    legends = ["KNN", "DT","RF", "SVC", "DeepCNN", "HHO-DeepCNN", "SSA-DeepCNN", "Falcon Finch DeepCNN"]
    xlab = "Training(%)"
    ylab = "Accuracy(%)"
    Complete_Figure_com(A, ii, legends, xlab, ylab)
    ii += 1
    ylab = "Sensitivity(%)"
    Complete_Figure_com(B, ii, legends, xlab, ylab)
    ii += 1
    ylab = "Specificity(%)"
    Complete_Figure_com(C, ii, legends, xlab, ylab)
    ii += 1
    legends_pef = ["Falcon Finch DeepCNN with epochs = 20", "Falcon Finch DeepCNN with epochs = 40",
                   "Falcon Finch DeepCNN with epochs = 60", "Falcon Finch DeepCNN with epochs = 80",
                   "Falcon Finch DeepCNN with epochs = 100"]
    ylab = "Accuracy(%)"
    Complete_Figure_perf(A1, ii, legends_pef, xlab, ylab)
    ii += 1
    ylab = "Sensitivity(%)"
    Complete_Figure_perf(B1, ii, legends_pef, xlab, ylab)
    ii += 1
    ylab = "Specificity(%)"
    Complete_Figure_perf(C1, ii, legends_pef, xlab, ylab)
    ii += 1
    [A, B, C, A1, B1, C1] = KF_load_parameters()
    xlab = "Kfold"
    ylab = "Accuracy(%)"
    KF_Complete_Figure_com(A, ii, legends, xlab, ylab)
    ii += 1
    ylab = "Sensitivity(%)"
    KF_Complete_Figure_com(B, ii, legends, xlab, ylab)
    ii += 1
    ylab = "Specificity(%)"
    KF_Complete_Figure_com(C, ii, legends, xlab, ylab)
    ii += 1
    ylab = "Accuracy(%)"
    KF_Complete_Figure_perf(A1, ii, legends_pef, xlab, ylab)
    ii += 1
    ylab = "Sensitivity(%)"
    KF_Complete_Figure_perf(B1, ii, legends_pef, xlab, ylab)
    ii += 1
    ylab = "Specificity(%)"
    KF_Complete_Figure_perf(C1, ii, legends_pef, xlab, ylab)
    ii += 1


def KF_Complete_Figure_com(perf, val, str_1, xlab, ylab):
    perf = np.sort(perf.T).T
    np.savetxt('Results/KF/Comp_Analysis\\' + '_' + str(val) + '_' + 'Graph.csv', perf, delimiter=",")
    n_groups = 5
    index = np.arange(n_groups)
    bar_width = 0.085
    opacity = 0.6
    plt.figure(val)
    plt.bar(index, perf[0, :], bar_width, alpha=opacity, edgecolor="black", color='#00ccff', label=str_1[0])
    plt.bar(index + bar_width, perf[1, :], bar_width, alpha=opacity, edgecolor="black", color='#0000ff', label=str_1[1])
    plt.bar(index + 2 * bar_width, perf[2, :], bar_width, alpha=opacity, edgecolor="black", color='#ff00ff', label=str_1[2])
    plt.bar(index + 3 * bar_width, perf[3, :], bar_width, alpha=opacity, edgecolor="black", color='#99ffcc', label=str_1[3])
    plt.bar(index + 4 * bar_width, perf[4, :], bar_width, alpha=opacity, edgecolor="black", color='#ff6600', label=str_1[4])
    plt.bar(index + 5 * bar_width, perf[5, :], bar_width, alpha=opacity, edgecolor="black", color='#00ffff', label=str_1[5])
    plt.bar(index + 6 * bar_width, perf[6, :], bar_width, alpha=opacity, edgecolor="black", color='#ff9999', label=str_1[6])
    plt.bar(index + 7 * bar_width, perf[7, :], bar_width, alpha=opacity, edgecolor="black", color='#00ff00', label=str_1[7])

    plt.xlabel(xlab, fontsize=10, fontweight='bold')
    plt.ylabel(ylab, fontsize=10, fontweight='bold')
    plt.xticks(index + 0.1, ('4', '5', '6', '7', '8'))
    plt.title("COMPARATIVE ANALYSIS")
    plt.legend(loc='lower right')
    plt.savefig('Results/KF/Comp_Analysis\\' + str(val) + '_' + 'Graph.png', dpi=800)
    plt.show(block=False)
    plt.clf()

def KF_Complete_Figure_perf(perf, val, str_1, xlab, ylab):
    perf = np.sort(perf.T).T
    np.savetxt('Results/KF/Perf_Analysis\\' + '_' + str(val) + '_' + 'Graph.csv', perf, delimiter=",")
    n_groups = 5
    index = np.arange(n_groups)
    bar_width = 0.085
    opacity = 0.6
    plt.figure(val)
    plt.bar(index, perf[0, :], bar_width, alpha=opacity, edgecolor="black", color='#0099ff', label=str_1[0])
    plt.bar(index + bar_width, perf[1, :], bar_width, alpha=opacity, edgecolor="black", color='#ff00ff', label=str_1[1])
    plt.bar(index + 2 * bar_width, perf[2, :], bar_width, alpha=opacity, edgecolor="black", color='#ff9900', label=str_1[2])
    plt.bar(index + 3 * bar_width, perf[3, :], bar_width, alpha=opacity, edgecolor="black", color='#ff33cc', label=str_1[3])
    plt.bar(index + 4 * bar_width, perf[4, :], bar_width, alpha=opacity, edgecolor="black", color='#00ff99', label=str_1[4])

    plt.xlabel(xlab, fontsize=10, fontweight='bold')
    plt.ylabel(ylab, fontsize=10, fontweight='bold')
    plt.xticks(index + 0.1, ('4', '5', '6', '7', '8'))
    plt.title("PERFORMANCE ANALYSIS")
    plt.legend(loc='lower right')
    plt.savefig('Results/KF/Perf_Analysis\\' + str(val) + '_' + 'Graph.png', dpi=800)
    plt.show(block=False)
    plt.clf()

def KF_load_parameters():
    A = np.load("Acc.npy") * 100
    B = np.load("Sen.npy") * 100
    C = np.load("Spe.npy") * 100
    perf_A = A[[0, 1, 2, 3, 4, 5, 6, 11], :]
    perf_B = B[[0, 1, 2, 3, 4, 5, 6, 11], :]
    perf_C = C[[0, 1, 2, 3, 4, 5, 6, 11], :]
    A1 = A[[7, 8, 9, 10, 11], :]
    B1 = B[[7, 8, 9, 10, 11], :]
    C1 = C[[7, 8, 9, 10, 11], :]
    return [perf_A, perf_B, perf_C, A1, B1, C1]
