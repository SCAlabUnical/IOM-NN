#!/usr/bin/env python

import os
import pprint as pp
import shutil
import time
from datetime import datetime as dt

import numpy as np

import IOM_NN.MLP_bow_based as mlp
import IOM_NN.analyzer as analyzer
import IOM_NN.constants as ctx
import IOM_NN.default_model as default_model
import IOM_NN.tweet_classifier as classifier
import operator
from functools import reduce
import copy



# UTILITY METHODS
def create_res_files(seed, def_model_res_table):
    results = os.path.dirname("results/test_" + str(seed) + "/")
    if not os.path.exists(results):
        os.makedirs(results)
    shutil.copyfile(def_model_res_table.name, "results/test_" + str(seed) + "/results_table")
    open("results/test_" + str(seed) + "/results_log", 'w').close()
    return {"res_table": open("results/test_" + str(seed) + "/results_table", 'a', encoding="utf-8"),
            "log": open("results/test_" + str(seed) + "/results_log", 'a', encoding="utf-8",
                        buffering=ctx.BUFFER_SIZE)}  # outputstream results map


def create_def_model_res_files():
    results = os.path.dirname("results/iter_0/")
    if not os.path.exists(results):
        os.makedirs(results)
    open("results/iter_0/results_table", 'w').close()
    open("results/iter_0/results_log", 'w').close()
    return {"res_table": open("results/iter_0/results_table", 'a', encoding="utf-8"),
            "log": open("results/iter_0/results_log", 'a', encoding="utf-8",
                        buffering=ctx.BUFFER_SIZE)}  # outputstream results map


def init_iteration(iteration, res_log, seed):
    print("\nITERATION: " + str(iteration))
    res_log.write("ITERATION: " + str(iteration) + "\n\n")
    results = os.path.dirname("results/test_" + str(seed) + "/iter_" + str(iteration) + "/results/")
    save = os.path.dirname("results/test_" + str(seed) + "/iter_" + str(iteration) + "/save/")
    if not os.path.exists(results):
        os.makedirs(results)
    if not os.path.exists(save):
        os.makedirs(save)


def current_milli_time():
    return int(round(time.time() * 1000))


def nowTimeStamp():
    year = dt.now().year
    month = dt.now().month
    day = dt.now().day
    hour = dt.now().hour
    minutes = dt.now().minute
    seconds = dt.now().second
    return str(year) + "." + str(month) + "." + str(day) + "@" + str(hour) + "_" + str(minutes) + "_" + str(seconds)


def stop(ratio, eps, iteration, max_iter, ann_tweets_per_class):
    # exit condition
    if ratio >= 1 - eps or ratio <= eps or iteration >= max_iter:
        return True
    # current iteration must classify at least one tweet for each class
    for c_index in range(ctx.N_CLASSES):
        if ann_tweets_per_class[c_index] == 0:
            return True
    return False


def merge(previous, current):
    for key in previous:
        if key == "annotated_tweets":
            print("merge annotated tweets with previous iteration results")
            ann_tweets = []
            ann_tweets.append(previous[key])
            ann_tweets.append(current[key])
            # flatten list of annotated tweets
            previous[key] = reduce(operator.concat, ann_tweets)
        else:
            print("merge keys")
            previous[key] = current[key]


# OPINION MINING METHOD
def mine_opinion(file_path, seed_list, eps=ctx.DEF_EPS, max_iter=ctx.DEF_MAX_ITER):
    # First iteration with base faction hashtags (Apply default rules)
    start_time_first_iter = current_milli_time()
    def_model_out_files = create_def_model_res_files()
    def_model_output = default_model.classify_input(file_path, def_model_out_files)  # iter. 0
    end_time_first_iter = current_milli_time()
    def_model_out_files["log"].write("Elapsed_time: " + str(end_time_first_iter - start_time_first_iter) + "\n\n")
    initial_tweets = def_model_output["number_of_initial_tweets"]
    del def_model_output["number_of_initial_tweets"]  # unnecessary key
    first_iter_neutrals = def_model_output["number_of_neutrals"]
    first_iter_ann_tweets_per_class = def_model_output["classified_tweets_per_class"]
    ratio = 1 - (first_iter_neutrals / initial_tweets)
    if stop(ratio, eps, 1, max_iter, first_iter_ann_tweets_per_class):
        def_model_out_files["log"].write("Definitive neutrals: " + str(first_iter_neutrals) + "\n\n")
        return
    # Iterative process (incremental annotation phase)
    for seed in seed_list:
        iteration = 1
        start_time = current_milli_time()
        print("CURRENT SEED: " + str(seed))
        output_files = create_res_files(seed, def_model_out_files["res_table"])
        mlp.set_random_seeds(seed)
        neutral_tweets = first_iter_neutrals
        prev_iter_output = copy.deepcopy(def_model_output)
        while True:
            init_iteration(iteration, output_files["log"], seed)
            if iteration == 1:
                bm = ctx.RANDOM_OVER_SAMPLING
            else:
                bm = ctx.RANDOM_UNDER_SAMPLING
            mlp.learn_from_positives(prev_iter_output["annotated_tweets"], iteration, seed,
                                     output_files["log"],
                                     balancing_method=bm)
            current_iteration_output = classifier.classify_unknown(prev_iter_output["unannotated_tweets"],
                                                                   iteration,
                                                                   seed, output_files)
            new_neutrals = current_iteration_output["number_of_neutrals"]
            new_ann_tweets_per_class = current_iteration_output["classified_tweets_per_class"]
            ratio = 1 - (new_neutrals / neutral_tweets)
            output_files["log"].write("Ratio: " + str(format(ratio, '.3f')) + "\n\n")
            print("Ratio: " + str(format(ratio, '.3f')))
            neutral_tweets = new_neutrals
            merge(prev_iter_output, current_iteration_output)
            iteration = iteration + 1
            if stop(ratio, eps, iteration, max_iter, new_ann_tweets_per_class):
                break
        output_files["log"].write("Definitive neutrals: " + str(neutral_tweets) + "\n\n")
        end_time = current_milli_time()
        output_files["log"].write("Elapsed_time: " + str(end_time - start_time) + "\n\n")
        print("SEED " + str(seed) + "  TEST FINISHED - *** See results_log and results_table for details ***\n")




##### Run the algorithm! #####

SEED_LIST = [ctx.SEED_BASE + s * ctx.SEED_VARIATION_STEP for s in range(0, ctx.N_TEST)]
dir_name = 'input'
print("Directory: ", dir_name)
file_to_analyze = os.listdir(dir_name)[0]
file_path = os.path.abspath(os.path.join(dir_name, file_to_analyze))
date_time = nowTimeStamp()
print("\nAnalyzing flie: " + file_to_analyze + "; date: " + date_time + "\n")
# annotation phase
start = current_milli_time()
mine_opinion(file_path, SEED_LIST)
end = current_milli_time()
ann_elapsed_time = end - start
# estimate polarization
start = current_milli_time()
res_1 = np.array([0.] * int(ctx.N_CLASSES))
res_2 = np.array([0.] * int(ctx.N_CLASSES))
res_3 = np.array([0.] * int(ctx.N_CLASSES))
res_4 = np.array([0.] * int(ctx.N_CLASSES))
r2s = np.array([0.] * int(ctx.N_HEURISTICS))
expl_vars = np.array([0.] * int(ctx.N_HEURISTICS))
mapes = np.array([0.] * int(ctx.N_HEURISTICS))
lars = np.array([0.] * int(ctx.N_HEURISTICS))
final_values = {"h1": res_1, "h2": res_2, "h3": res_3, "h_mean": res_4}
final_scores = {"r2": r2s, "mape": mapes, "expl_var": expl_vars, "log_acc_ratio": lars}
open("results/polarization estimate", 'w').close()
pol_estimates = open("results/polarization estimate", 'w')
real_results = ctx.REAL_RES
pol_estimates.write("REAL RESULTS: " + str(real_results) + "\n\n")
seed_base_test_dir = "results/test_" + str(ctx.SEED_BASE)
if not os.path.exists(seed_base_test_dir):  # the process ended just after iter. 0
    table_path = "results/iter_0/results_table"
    dict = analyzer.evaluate(table_path)
    res_dict = dict["results"]
    stats_dict = dict["stats"]
    pp.pprint(res_dict, pol_estimates)
    final_values = {k: final_values[k] + res_dict["predicted_values"][k] for k in final_values.keys()}
    final_scores = {k: final_scores[k] + res_dict["h_scores"][k] for k in final_scores.keys()}
    n_users = stats_dict["n_users"]
    n_tweets = stats_dict["n_tweets"]
    levels = stats_dict["level"]
else:
    n_users = 0
    n_tweets = 0
    for seed in SEED_LIST:
        table_path = "results/test_" + str(seed) + "/results_table"
        dict = analyzer.evaluate(table_path)
        res_dict = dict["results"]
        stats_dict = dict["stats"]
        pol_estimates.write("SEED: " + str(seed) + "\n")
        pp.pprint(res_dict, pol_estimates)
        pol_estimates.write("\n")
        final_values = {k: final_values[k] + res_dict["predicted_values"][k] for k in final_values.keys()}
        final_scores = {k: final_scores[k] + res_dict["h_scores"][k] for k in final_scores.keys()}
        n_users = n_users + stats_dict["n_users"]
        n_tweets = n_tweets + stats_dict["n_tweets"]
    final_values = {k: final_values[k] / ctx.N_TEST for k in final_values.keys()}
    final_scores = {k: final_scores[k] / ctx.N_TEST for k in final_scores.keys()}
    n_users = n_users / ctx.N_TEST
    n_tweets = n_tweets / ctx.N_TEST
    levels = stats_dict["level"]  # take last seed depth
final_results_dict = {"mean_predicted_values": final_values}
pol_estimates.write("FINAL RESULTS: \n")
pp.pprint(final_results_dict, pol_estimates)
pol_estimates.write("\n\nRISULTATI REALI: " + str(real_results) + "\n\n")
print("POLARIZATION ESTIMATE:")
pp.pprint(final_results_dict)
end = current_milli_time()
est_elapsed_time = end - start
info_file = open("results/info", 'w')
info_file.write("Stats:\n")
info_file.write("-- # iteration analyzed: " + str(levels) + "\n")
info_file.write("-- # users: " + str(n_users) + "\n")
info_file.write("-- # tweets: " + str(n_tweets) + "\n")
info_file.write("Params:\n")
info_file.write("-- # seeds: " + str(ctx.N_TEST) + "\n")
info_file.write("-- th users:" + str(ctx.TH_u) + "\n")
info_file.write("-- min tweets:" + str(ctx.MIN_TWEETS) + "\n")
info_file.write("-- max tweets:" + str(ctx.MAX_TWEETS) + "\n")
info_file.write("Elapsed time (ms):\n")
info_file.write("-- annotation phase: " + str(ann_elapsed_time) + "\n")
info_file.write("-- polarization estimation phase: " + str(est_elapsed_time))
print("ELAPSED TIME (ms):")
print("-- annotation phase: " + str(ann_elapsed_time))
print("-- polarization estimation phase: " + str(est_elapsed_time))
pol_estimates.close()
info_file.close()
