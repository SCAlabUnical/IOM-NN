#!/usr/bin/env python

import json
import os
import pickle

import numpy as np

import IOM_NN.constants as ctx
from IOM_NN.NumpyJSONEncoder import NumpyEncoder as NJenc

# neutral hashtags
neutralKeys = ctx.NEUTRAL_KEYS
# faction hashtags
keys = ctx.KEYS
MIN_FREQ = ctx.MIN_FREQ


def compute_frequencies(file_path):
    d= {}
    with open(file_path, 'r', encoding="utf8") as input_file:
        i = 0
        for line in input_file:
            if i % 5000 == 0:
                print(str(i) + " processed")
            i = i + 1
            line = line.strip()
            try:
                tweet = json.loads(line)
                for h in tweet["hashtags"]:
                    h = h.lower()
                    d[h] = d.get(h, 0) + 1
            except:
                print('line json error')
    pickle.dump(d, open("dict.pckl", 'wb'))


def read_hashtags(tweet, dict):
    hashtags = tweet["hashtags"]
    res = []
    for h in hashtags:
        h = h.lower()
        if dict[h] > MIN_FREQ:
            res.append(h)
    return res


def prepareLinePos(tweet, indicePartito, dict):
    positives = keys[indicePartito]
    hashtags = read_hashtags(tweet, dict)
    # rimozione hashtags di base (positivi o neutri)
    htags = []
    for tag in hashtags:
        tag = tag.lower()
        if (not (tag in positives or tag in neutralKeys)):
            htags.append(tag)
    if len(htags) == 0:
        return None
    tweet_id = str(tweet["id"])
    user_id = str(tweet["user"]["id"])
    favourites = str(tweet["favoutites"])
    retweets = tweet["retweets"]
    is_retweet = str(retweets > 0)
    retweets = str(retweets)
    line = {"tweet_id": tweet_id, "user_id": user_id, "hashtags": htags, "class": indicePartito,
            "favourites": favourites, "retweets": retweets, "is_retweet": is_retweet}
    return line


def prepareLineOth(tweet, dict):
    hashtags = read_hashtags(tweet, dict)
    # rimozione hashtags di base (neutri)
    htags = []
    for tag in hashtags:
        tag = tag.lower()
        if (not (tag in neutralKeys)):
            isBase = False
            for indicePartito in range(ctx.N_CLASSES):
                if (tag in keys[indicePartito]):
                    isBase = True
                    break
            if (not (isBase)):
                htags.append(tag)
    if len(htags) == 0:
        return None
    tweet_id = str(tweet["id"])
    user_id = str(tweet["user"]["id"])
    favourites = str(tweet["favoutites"])
    retweets = tweet["retweets"]
    is_retweet = str(retweets > 0)
    retweets = str(retweets)
    line = {"tweet_id": tweet_id, "user_id": user_id, "hashtags": htags, "favourites": favourites, "retweets": retweets,
            "is_retweet": is_retweet}
    return line


def isPositive(indicePartito, hashtags):
    positives = keys[indicePartito]
    for ht in hashtags:
        ht = ht.lower()
        if (ht in positives):
            return True
    return False


def prepareTableRecord(tweet, iteration, indicePartito, dict):
    # get fields
    tweet_id = str(tweet["id"])
    user_id = str(tweet["user"]["id"])
    favourites = str(tweet["favoutites"])
    retweets = tweet["retweets"]
    is_retweet = str(retweets > 0)
    retweets = str(retweets)

    hashtags = read_hashtags(tweet, dict)

    line = {"tweet_id": tweet_id, "user_id": user_id, "hashtags": hashtags, "class": indicePartito,
            "favourites": favourites, "retweets": retweets, "is_retweet": is_retweet, "iteration": iteration}
    return line


def classify_input(file_path, def_model_out_files):
    annotated_tweets = []
    unannotated_tweets = []
    results_log = def_model_out_files["log"]
    results_table = def_model_out_files["res_table"]
    print("ITERATION: 0 - Apply default rules")
    results_log.write("ITERATION: 0 - Apply default rules\n\n")
    positive = [0] * ctx.N_CLASSES
    neutrals = 0
    # 'base_tweets' do not take part to the mining process
    base_positive = [0] * ctx.N_CLASSES
    base_neutrals = 0
    if not os.path.exists("dict.pckl"):
        compute_frequencies(file_path)
    dict = pickle.load(open("dict.pckl", 'rb'))
    with open(file_path, 'r', encoding="utf8") as input_file:
        i = 0
        for line in input_file:
            if i % 5000 == 0:
                print(str(i) + " processed")
            i = i + 1
            line = line.strip()
            tweet = json.loads(line)
            hashtags = read_hashtags(tweet, dict)
            # skip tweet without hashtags
            if len(hashtags) == 0:
                continue
            isNeutral = True
            indicePartito = -1
            for indiceP in range(ctx.N_CLASSES):
                if isPositive(indiceP, hashtags):
                    if isNeutral:
                        indicePartito = indiceP
                        isNeutral = False
                    else:
                        isNeutral = True
                        break
            if not (isNeutral):
                table_record = prepareTableRecord(tweet, 0, indicePartito, dict)
                j_record = json.dumps(table_record, cls=NJenc)
                results_table.write(j_record + '\n')
                contentLine = prepareLinePos(tweet, indicePartito, dict)
                if (not (contentLine == None)):
                    positive[indicePartito] = positive[indicePartito] + 1
                    annotated_tweets.append(contentLine)
                else:
                    base_positive[indicePartito] = base_positive[indicePartito] + 1
            else:
                contentLine = prepareLineOth(tweet, dict)
                if not contentLine == None:
                    neutrals = neutrals + 1
                    unannotated_tweets.append(contentLine)
                else:
                    base_neutrals = base_neutrals + 1
    # results
    all_tweets = np.sum(positive) + neutrals + np.sum(base_positive) + base_neutrals
    p_positive = np.array([positive[i] / all_tweets for i in range(ctx.N_CLASSES)])
    p_base_positive = np.array([base_positive[i] / all_tweets for i in range(ctx.N_CLASSES)])
    p_neutral = neutrals / all_tweets
    p_base_neutral = base_neutrals / all_tweets
    print("Results:")
    print("- new positives:", positive, "perc: ", p_positive)
    print("- base positives:", base_positive, "perc: ", p_base_positive)
    print("- new neutrals:", neutrals, "perc: ", p_neutral)
    print("- base neutrals:", base_neutrals, "perc: ", p_base_neutral)
    results_log.write("Analisi dei tweet che possiedono solo hashtag di base:\n")
    results_log.write("base positive tweets: " + str(base_positive) + "; perc: " + str(p_base_positive) + "\n")
    results_log.write("base neutrals: " + str(base_neutrals) + "; perc: " + str(format(p_base_neutral, '.3f')) + "\n")
    results_log.write("\nnew positive tweets: " + str(positive) + "; perc: " + str(p_positive) + "\n")
    results_log.write("new neutrals: " + str(neutrals) + "; perc: " + str(format(p_neutral, '.3f')) + "\n")

    default_model_output = {"number_of_initial_tweets": all_tweets, "number_of_neutrals": neutrals,
                            "annotated_tweets": annotated_tweets,
                            "unannotated_tweets": unannotated_tweets, "classified_tweets_per_class": positive}
    return default_model_output
