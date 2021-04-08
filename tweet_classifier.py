#!/usr/bin/env python

import json
import os
import pickle

import numpy as np
from keras.models import model_from_json

import IOM_NN.constants as ctx
from IOM_NN.NumpyJSONEncoder import NumpyEncoder as NJenc


def load_dictionary(iteration, seed):
    path = "results/test_" + str(seed) + "/iter_" + str(iteration) + "/save/vectorizer_" + str(iteration) + ".pk"
    if (os.path.exists(path)):
        print("loading dictionary...")
        pickle_off = open(path, "rb")
        vectorizer = pickle.load(pickle_off)
        print("dictionary loaded")
        return vectorizer
    raise Exception("ERROR - Dictionary not found!")


# load model and weights
def load_model(iteration, seed):
    json_file = open(
        "results/test_" + str(seed) + "/iter_" + str(iteration) + "/save/MLP_" + str(iteration) + "_model.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(
        "results/test_" + str(seed) + "/iter_" + str(iteration) + "/save/MLP_" + str(iteration) + "_weights.h5")
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return loaded_model


def prepareLinePos(tweet, indicePartito):
    hashtags = tweet["hashtags"]
    tweet_id = str(tweet["tweet_id"])
    user_id = str(tweet["user_id"])
    favourites = str(tweet["favourites"])
    retweets = str(tweet["retweets"])
    is_retweet = str(tweet["is_retweet"])
    line = {"tweet_id": tweet_id, "user_id": user_id, "hashtags": hashtags, "class": indicePartito,
            "favourites": favourites, "retweets": retweets, "is_retweet": is_retweet}
    return line


def prepareLineOth(tweet):
    hashtags = tweet["hashtags"]
    tweet_id = str(tweet["tweet_id"])
    user_id = str(tweet["user_id"])
    favourites = str(tweet["favourites"])
    retweets = str(tweet["retweets"])
    is_retweet = str(tweet["is_retweet"])
    line = {"tweet_id": tweet_id, "user_id": user_id, "hashtags": hashtags, "favourites": favourites,
            "retweets": retweets, "is_retweet": is_retweet}
    return line


def classify_unknown(prev_unannotated_tweets, iteration_index, seed, output):
    annotated_tweets = []
    unannotated_tweets = []
    res_log = output["log"]
    res_table = output["res_table"]
    positive = [0] * ctx.N_CLASSES
    neutrals = 0
    vectorizer = load_dictionary(iteration_index, seed)
    loaded_model = load_model(iteration_index, seed)
    print("Classify unknown tweets")
    res_log.write("Classify unknown tweets\n")
    network_inputs = []
    for tweet in prev_unannotated_tweets:
        htags_list = tweet["hashtags"]
        hashtags = [" ".join(htags_list)]
        # get network input
        network_input = vectorizer.transform(hashtags).toarray()
        network_input = network_input.flatten()
        network_inputs.append(network_input)
    # predict class of selected tweets
    network_inputs = np.array(network_inputs)
    print("Inference mode; input shape: ", network_inputs.shape)
    network_outputs = loaded_model.predict(network_inputs)
    for network_output, tweet in zip(network_outputs, prev_unannotated_tweets):
        max_value = np.amax(network_output)
        if (max_value >= ctx.TH_t):
            pred_class = np.argmax(network_output)
            positive[pred_class] = positive[pred_class] + 1
            line = prepareLinePos(tweet, pred_class)
            annotated_tweets.append(line)
            # insert the tweet into the result table
            line["iteration"] = iteration_index
            j_record = json.dumps(line, cls=NJenc)
            res_table.write(j_record + '\n')
        else:
            # neutrals
            neutrals = neutrals + 1
            line = prepareLineOth(tweet)
            unannotated_tweets.append(line)
    # results
    all_tweets = np.sum(positive) + neutrals
    p_positive = np.array([positive[i] / all_tweets for i in range(ctx.N_CLASSES)])
    p_neutral = neutrals / all_tweets
    print("Results:")
    print("- new positives:", positive, "perc: ", p_positive)
    print("- new neutrals:", neutrals, "perc: ", p_neutral)
    res_log.write("new positive tweets: " + str(positive) + "; perc: " + str(p_positive) + "\n")
    res_log.write("new neutrals: " + str(neutrals) + "; perc: " + str(format(p_neutral, '.3f')) + "\n")
    class_model_output = {"number_of_neutrals": neutrals, "annotated_tweets": annotated_tweets,
                          "unannotated_tweets": unannotated_tweets, "classified_tweets_per_class": positive}
    return class_model_output


# Experimental version of the classification procedure.
# Try it in case of strict memory requirements tuning the batch size.
def batch_classify_unknown(prev_unannotated_tweets, iteration_index, seed, output, batch_size=1000):
    annotated_tweets = []
    unannotated_tweets = []
    res_log = output["log"]
    res_table = output["res_table"]
    positive = [0] * ctx.N_CLASSES
    neutrals = 0
    vectorizer = load_dictionary(iteration_index, seed)
    loaded_model = load_model(iteration_index, seed)
    print("Classify unknown tweets")
    res_log.write("Classify unknown tweets\n")
    parts = np.ceil(len(prev_unannotated_tweets) / batch_size)
    j = 0
    i = 1
    while i <= parts:
        network_inputs = []
        k = j
        while j < batch_size * i and j < len(prev_unannotated_tweets):
            tweet = prev_unannotated_tweets[j]
            htags_list = np.array(tweet["hashtags"])
            hashtags = np.array([" ".join(htags_list)])
            # get network input
            network_input = vectorizer.transform(hashtags).toarray()
            network_inputs.append(network_input)
            j += 1
        # predict class of selected tweets
        network_inputs = np.array(network_inputs)
        network_outputs = loaded_model.predict(network_inputs)
        for network_output in network_outputs:
            tweet = prev_unannotated_tweets[k]
            max_value = np.amax(network_output)
            if (max_value >= ctx.TH_t):
                pred_class = np.argmax(network_output)
                positive[pred_class] = positive[pred_class] + 1
                line = prepareLinePos(tweet, pred_class)
                annotated_tweets.append(line)
                # inserisce il tweet nella result table
                line["iteration"] = iteration_index
                j_record = json.dumps(line, cls=NJenc)
                res_table.write(j_record + '\n')
            else:
                # neutrals
                neutrals = neutrals + 1
                line = prepareLineOth(tweet)
                unannotated_tweets.append(line)
            k += 1
        i += 1
    # results
    all_tweets = np.sum(positive) + neutrals
    p_positive = np.array([positive[i] / all_tweets for i in range(ctx.N_CLASSES)])
    p_neutral = neutrals / all_tweets
    print("Results:")
    print("- new positives:", positive, "perc: ", p_positive)
    print("- new neutrals:", neutrals, "perc: ", p_neutral)
    res_log.write("new positive tweets: " + str(positive) + "; perc: " + str(p_positive) + "\n")
    res_log.write("new neutrals: " + str(neutrals) + "; perc: " + str(format(p_neutral, '.3f')) + "\n")
    class_model_output = {"number_of_neutrals": neutrals, "annotated_tweets": annotated_tweets,
                          "unannotated_tweets": unannotated_tweets, "classified_tweets_per_class": positive}
    return class_model_output
