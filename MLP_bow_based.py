#!/usr/bin/env python

import os
import pickle
import random as rn
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from keras.callbacks import EarlyStopping, Callback
from keras.callbacks import ModelCheckpoint
from keras.layers.core import Dense, Activation
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import IOM_NN.constants as ctx

session_conf = tf.ConfigProto(intra_op_parallelism_threads=ctx.N_THREADS, inter_op_parallelism_threads=ctx.N_THREADS)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)


def set_random_seeds(seed):
    # set random seeds for reproducibility
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(seed)
    rn.seed(seed)
    tf.set_random_seed(seed)


def compute_per_class_accuracy(confusion_matrix):
    acc = np.array([0.] * ctx.N_CLASSES)
    for e in range(0, ctx.N_CLASSES):
        TP = confusion_matrix[e][e]
        FP = float(np.sum(confusion_matrix[:, e])) - TP
        FN = float(np.sum(confusion_matrix[e])) - TP
        N = np.sum(confusion_matrix)
        TN = N - (FP + FN + TP)
        acc[e] = (TP + TN) / N
    return acc


def get_weights(y_train):
    classes = np.array([0] * ctx.N_CLASSES)
    # revert one-hot encoding
    for lab_v in y_train:
        classes = np.append(classes, np.argmax(lab_v))
    counter = Counter(classes)
    majority = max(counter.values())
    return {cls: float(majority / count) for cls, count in counter.items()}


def train_uniform_test_split(dataset, perc_split):
    np.random.shuffle(dataset)
    sentences = dataset[:, 0]
    classes = dataset[:, 1]
    df = pd.DataFrame({'sentences': sentences, 'classes': classes})
    groups = df.groupby(['classes'])
    lengths = groups.count()
    min_len = np.amin(lengths["sentences"])
    split_size_per_class = int(min_len * perc_split)
    s_train = np.array([])
    s_test = np.array([])
    c_train = np.array([])
    c_test = np.array([])
    # first n% examples of each group build the test set
    for group in groups:
        s_test = np.append(s_test, group[1]["sentences"][:split_size_per_class])
        s_train = np.append(s_train, group[1]["sentences"][split_size_per_class:])
        c_test = np.append(c_test, group[1]["classes"][:split_size_per_class])
        c_train = np.append(c_train, group[1]["classes"][split_size_per_class:])
    # shuffle, slice and convert from pandas to numpy types
    train = np.column_stack((s_train, c_train))
    np.random.shuffle(train)
    x_train = train[:, 0].astype('U')
    y_train = train[:, 1].astype('f')
    test = np.column_stack((s_test, c_test))
    np.random.shuffle(test)
    x_test = test[:, 0].astype('U')
    y_test = test[:, 1].astype('f')
    return x_train, x_test, y_train, y_test


# custom callback
class EarlyStoppingByAccVal(Callback):
    def __init__(self, monitor='val_accuracy', value=ctx.MAX_ACC, verbose=ctx.SILENT):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)
        if current >= self.value:
            if self.verbose > 0:
                print("Epoch %05d: enough acc reached --> early stopping" % epoch)
            self.model.stop_training = True



def learn_from_positives(annotated_tweets, iteration, seed, output_log, balancing_method=None):
    if (iteration == 1):
        MAX_EPOCHS = int(np.ceil(ctx.MAX_EPOCHS*1.5))
    else:
        MAX_EPOCHS = ctx.MAX_EPOCHS
    sentences = []
    classes = []
    for tweet in annotated_tweets:
        hashtags = np.array(" ".join(tweet["hashtags"]))
        sentences.append(hashtags)
        classes.append(tweet["class"])
    sentences = np.array(sentences)
    classes = np.array(classes)
    print("dataset shape:", sentences.shape)
    output_log.write("dataset shape: " + str(sentences.shape) + "\n")
    # class balancing
    balancer = RandomUnderSampler()
    if (balancing_method == None):
        output_log.write("balancing_method: None\n")
        # create resampled dictionary
        vectorizer = CountVectorizer(min_df=0, lowercase=False)
        vectorizer.fit(sentences)
        # save vectorizer on disk
        pickling_on = open(
            "results/test_" + str(seed) + "/iter_" + str(iteration) + "/save/vectorizer_" + str(
                iteration) + ".pk", "wb")
        pickle.dump(vectorizer, pickling_on)
        pickling_on.close()
        print("dictionary saved")
        # convert resampled input into a BoW matrix
        bows = vectorizer.transform(sentences).toarray()
        # split train test
        hashtags_train, hashtags_test, y_train, y_test = train_test_split(bows, classes,
                                                                          test_size=ctx.TEST_SET_PERC_SIZE,
                                                                          random_state=seed)
    elif (balancing_method == ctx.RANDOM_OVER_SAMPLING):
        output_log.write("balancing_method: ROS\n")
        balancer = RandomOverSampler()
        dataset = np.column_stack((sentences, classes))
        # split train test with uniform per class test support
        sentences_train, sentences_test, classes_train, y_test = train_uniform_test_split(dataset,
                                                                                          ctx.TEST_SET_PERC_SIZE)
        # reshape sentences for balancing
        sentences_train = sentences_train.reshape(-1, 1)
        # resample only training sentences
        res_train_sentences, y_train = balancer.fit_resample(sentences_train, classes_train)
        # restore shape of resampled sentences
        res_train_sentences.shape = (len(res_train_sentences))
        # create resampled dictionary
        vectorizer = CountVectorizer(min_df=0, lowercase=False)
        vectorizer.fit(res_train_sentences)
        # save vectorizer on disk
        pickling_on = open(
            "results/test_" + str(seed) + "/iter_" + str(iteration) + "/save/vectorizer_" + str(
                iteration) + ".pk", "wb")
        pickle.dump(vectorizer, pickling_on)
        pickling_on.close()
        print("dictionary saved")
        # convert resampled input into a BoW matrix
        hashtags_train = vectorizer.transform(res_train_sentences).toarray()
        hashtags_test = vectorizer.transform(sentences_test).toarray()
    else:
        if (balancing_method != ctx.RANDOM_UNDER_SAMPLING):
            warn = "WARNING - Unknown balancing method was given: using default (RUS)"
            output_log.write(warn + "\n")
            print(warn)
        else:
            output_log.write("balancing_method: RUS\n")
        # reshape sentences for balancing
        sentences = sentences.reshape(-1, 1)
        res_sentences, res_classes = balancer.fit_resample(sentences, classes)
        # restore shape of resampled sentences
        res_sentences.shape = (len(res_sentences))
        # create resampled dictionary
        vectorizer = CountVectorizer(min_df=0, lowercase=False)
        vectorizer.fit(res_sentences)
        # save vectorizer on disk
        pickling_on = open(
            "results/test_" + str(seed) + "/iter_" + str(iteration) + "/save/vectorizer_" + str(
                iteration) + ".pk", "wb")
        pickle.dump(vectorizer, pickling_on)
        pickling_on.close()
        print("dictionary saved")
        # convert resampled input into a BoW matrix
        bows = vectorizer.transform(res_sentences).toarray()
        # split train test
        hashtags_train, hashtags_test, y_train, y_test = train_test_split(bows, res_classes,
                                                                          test_size=ctx.TEST_SET_PERC_SIZE,
                                                                          random_state=seed)
    # convert class vector to a binary matrix
    y_train = np_utils.to_categorical(y_train, ctx.N_CLASSES)
    y_test = np_utils.to_categorical(y_test, ctx.N_CLASSES)
    print("training shape:", hashtags_train.shape, ", ", "test shape:", hashtags_test.shape)
    output_log.write(
        "training shape: " + str(hashtags_train.shape) + ", " + "test shape: " + str(hashtags_test.shape) + "\n")
    # create MLP (multi layer perceptron) model
    model = Sequential()
    n_input = len(hashtags_train[0])
    n_hidden = int(n_input * ctx.SCALE_FACTOR)
    model.add(Dense(n_hidden, input_dim=n_input))
    model.add(Activation('relu'))
    # add a softmax classifier
    model.add(Dense(ctx.N_CLASSES, activation='softmax'))
    # display network structure informations
    print(model.summary())
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # set callbacks for early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=ctx.LOG_LEVEL, patience=ctx.PATIENCE)
    es_by_acc = EarlyStoppingByAccVal(monitor='val_accuracy', value=ctx.MAX_ACC, verbose=ctx.PROGRESS)
    best_weights_file = "results/test_" + str(seed) + "/iter_" + str(iteration) + "/save/MLP_" + str(
        iteration) + "_weights.h5"
    mc = ModelCheckpoint(best_weights_file, monitor='val_loss', mode='min', verbose=ctx.ONE_LINE_PER_EPOCH,
                         save_best_only=True)
    # train model testing it on each epoch
    history = model.fit(hashtags_train, y_train, validation_data=(hashtags_test, y_test), batch_size=ctx.BATCH_SIZE,
                        callbacks=[es, es_by_acc, mc], epochs=MAX_EPOCHS, verbose=ctx.ONE_LINE_PER_EPOCH)
    # generate metrics plot:
    # accuracy on validation
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig("results/test_" + str(seed) + "/iter_" + str(iteration) + "/results/MLP_" + str(
        iteration) + "_accuracy.png")
    plt.gcf().clear()  # clear
    # loss on validation
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig("results/test_" + str(seed) + "/iter_" + str(iteration) + "/results/MLP_" + str(
        iteration) + "_loss.png")
    plt.gcf().clear()  # clear
    # final metrics
    print("\nFinal metrics")
    # train acc and loss
    print("\n%s: %.2f%%" % ("train accuracy", history.history['accuracy'][len(history.history['accuracy']) - 1] * 100))
    print("%s: %.2f" % ("train loss", history.history['loss'][len(history.history['loss']) - 1]))
    output_log.write("model MPL_" + str(iteration) + " metrics:\n")
    output_log.write(
        "train accuracy: " + str(format(history.history['accuracy'][len(history.history['accuracy']) - 1], '.3f')) + "\n")
    output_log.write(
        "train loss: " + str(format(history.history['loss'][len(history.history['loss']) - 1], '.3f')) + "\n")
    # validation acc and loss
    print(
        "\n%s: %.2f%%" % ("validation accuracy", history.history['val_accuracy'][len(history.history['val_accuracy']) - 1] * 100))
    print("%s: %.2f" % ("validation loss", history.history['val_loss'][len(history.history['val_loss']) - 1]))
    # test acc and loss
    # load the best saved model
    model.load_weights(best_weights_file)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    test_metrics = model.evaluate(hashtags_test, y_test, batch_size=ctx.BATCH_SIZE)
    print("\n%s: %.2f%%" % ("test " + model.metrics_names[1], test_metrics[1] * 100))
    print("%s: %.2f" % ("test " + model.metrics_names[0], test_metrics[0]))
    output_log.write("test accuracy: " + str(format(test_metrics[1], '.3f')) + "\n")
    output_log.write("test loss: " + str(format(test_metrics[0], '.3f')) + "\n")
    # test acc and loss per class
    y_test = y_test.astype(int)
    real_class = np.argmax(y_test, axis=1)
    pred_class = np.argmax(model.predict(hashtags_test), axis=-1)
    report = classification_report(real_class, pred_class)
    print(report)
    output_log.write("classification report:\n" + str(report) + "\n")
    cm = confusion_matrix(real_class, pred_class)
    print("confusion_matrix:")
    print(cm)
    output_log.write("confusion_matrix:\n" + str(cm) + "\n")
    # ciascuna classe e' rappresentata
    if (len(cm) == ctx.N_CLASSES):
        print("test accuracy: ")
        per_class_acc = compute_per_class_accuracy(cm)
        print(per_class_acc)
        output_log.write("per class accuracy: " + str(per_class_acc) + "\n\n")
    # save neural network on disk:
    # serialize model to JSON
    model_json = model.to_json()
    with open("results/test_" + str(seed) + "/iter_" + str(iteration) + "/save/MLP_" + str(
            iteration) + "_model.json",
              "w") as json_file:
        json_file.write(model_json)
    print("model MLP_" + str(iteration) + " saved")
    print()
