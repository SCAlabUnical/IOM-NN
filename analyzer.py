#!/usr/bin/env python
import json

import numpy as np
from sklearn.metrics import explained_variance_score as expl_var
from sklearn.metrics import r2_score as r2

import IOM_NN.constants as ctx

np.set_printoptions(precision=2)


def mape(real, pred):
    return round(np.sum(np.abs((real - pred) / real)) / len(real), 2)


def log_acc_ratio(real, pred):
    eps = np.array([1e-10] * len(pred))  # log(x): R+ -> R
    return round(1 - np.sum(np.abs(np.log((pred + eps) / real))) / len(real), 2)


def evaluate(table_path, level=-1):
    table = open(table_path, 'r')
    users_info = {}
    if level < 0: # analyze all levels
        lev = "ALL"
        for line in table:
            line = line.strip()
            tweet = json.loads(line)
            usr = tweet["user_id"]
            # likes = int(tweet["favourites"])
            # retweets = int(tweet["retweets"])
            if not usr in users_info:
                users_info[usr] = np.array([0.] * ctx.N_CLASSES)
            partito = tweet["class"]
            users_info[usr][partito] = users_info[usr][partito] + 1
    else:
        lev = str(level)
        for line in table:
            line = line.strip()
            tweet = json.loads(line)
            usr = tweet["user_id"]
            it = int(tweet["iteration"])
            if it > level:
                break
            # likes = int(tweet["favourites"])
            # retweets = int(tweet["retweets"])
            if not usr in users_info:
                users_info[usr] = np.array([0.] * ctx.N_CLASSES)
            partito = tweet["class"]
            users_info[usr][partito] = users_info[usr][partito] + 1
    # compute heuristics and stats
    res_1 = np.array([0.] * int(ctx.N_CLASSES))
    res_2 = np.array([0.] * int(ctx.N_CLASSES))
    res_3 = np.array([0.] * int(ctx.N_CLASSES))
    n_users = len(users_info.keys())
    n_tweets = 0
    for usr, preferences in users_info.items():
        p = np.amax(preferences)
        s = np.sum(preferences)
        n_tweets = n_tweets + s
        user_contribution = p / s
        if user_contribution >= ctx.TH_u and s >= ctx.MIN_TWEETS and s <= ctx.MAX_TWEETS:
            partito = np.argmax(preferences)
            # h1: sum 1 statically
            res_1[partito] = res_1[partito] + 1
            # h2: sum dominant class contribution percentage
            res_2[partito] = res_2[partito] + user_contribution
            # h3: sum all contribution percentages
            contributions = preferences / s
            res_3 = res_3 + contributions
    norm_val = ctx.NORM_VAL
    total_1 = np.sum(res_1)
    if total_1 > 0:
        res_1 = [round((p / total_1 * norm_val), 2) for p in res_1]
    total_2 = np.sum(res_2)
    if total_2 > 0:
        res_2 = [round((p / total_2 * norm_val), 2) for p in res_2]
    total_3 = np.sum(res_3)
    if total_3 > 0:
        res_3 = [round((p / total_3 * norm_val), 2) for p in res_3]
    res_4 = np.add(res_2, res_3) / 2  # mean of H2 and H3
    res_4 = [round(p, 2) for p in res_4]
    values = {"h1": res_1, "h2": res_2, "h3": res_3, "h_mean": res_4}
    RIS_REALI = ctx.REAL_RES
    r2s = [round(i, 2) for i in
           [r2(RIS_REALI, res_1), r2(RIS_REALI, res_2), r2(RIS_REALI, res_3), r2(RIS_REALI, res_4)]]
    expl_vars = [round(i, 2) for i in
                 [expl_var(RIS_REALI, res_1), expl_var(RIS_REALI, res_2), expl_var(RIS_REALI, res_3),
                  expl_var(RIS_REALI, res_4)]]
    mapes = [mape(RIS_REALI, res_1), mape(RIS_REALI, res_2), mape(RIS_REALI, res_3), mape(RIS_REALI, res_4)]
    lars = [log_acc_ratio(RIS_REALI, res_1), log_acc_ratio(RIS_REALI, res_2), log_acc_ratio(RIS_REALI, res_3),
            log_acc_ratio(RIS_REALI, res_4)]
    scores = {"r2": r2s, "mape": mapes, "expl_var": expl_vars, "log_acc_ratio": lars}
    return {"results": {"predicted_values": values, "h_scores": scores},
            "stats": {"n_tweets": n_tweets, "n_users": n_users, "level": lev}}
