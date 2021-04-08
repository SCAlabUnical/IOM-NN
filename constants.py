import numpy as np

# factions
HILLARY = 0
TRUMP = 1
N_CLASSES = 2
LABELS = ["HILLARY", "TRUMP"]

# results for comparison
REAL_RES = np.asarray([48.2, 43.3])
NORM_VAL = np.sum(REAL_RES)

# hashtag mincount
MIN_FREQ = 10

# base neutral hashtags
NEUTRAL_KEYS = np.array(["election2016", "elections2016", "uselections", "uselection", "earlyvote", "ivoted", "potus"])

# base positive hashtags
KEYS = {}
KEYS[HILLARY] = np.array(["clinton", "clintokaine16", "democrats", "dems", "dnc", "dumpfortrump", "factcheck",
                          "hillary16", "hillary2016", "hillary", "hillaryclinton", "hillarysupporter", "hrc",
                          "imwithher", "lasttimetrumppaidtaxes", "nevertrump", "ohhillyes", "p2", "strongertogether",
                          "trumptape", "uniteblu", "notmypresident"])
KEYS[TRUMP] = np.array(["americafirst", "benghazi", "crookedhillary", "draintheswamp", "lockherup", "maga3x", "maga",
                        "makeamericagreatagain", "neverhillary", "podestaemails", "projectveritas", "riggedetection",
                        "tcot", "trump", "trump2016", "trumppence16", "trumptrain", "voterfraud", "votetrump",
                        "wakeupamerica", "lockherup", "hillarysemail", "weinergate"])
# user polarization tresholds
TH_u = 2.0 / 3.0
MIN_TWEETS = 5
MAX_TWEETS = 100

# tweet polarization treshold
TH_t = 0.9

# class balancing method
RANDOM_OVER_SAMPLING = "ROS"
RANDOM_UNDER_SAMPLING = "RUS"

# neural network
N_THREADS = 64
BATCH_SIZE = 32
TEST_SET_PERC_SIZE = 0.3
# log levels
SILENT = 0
PROGRESS = 1
ONE_LINE_PER_EPOCH = 2
DROP_RATE = 0.05
LOG_LEVEL = 1
PATIENCE = 3
MAX_EPOCHS = 25
MAX_ACC = 0.99
# scale the number of neurons in subsequent FC layers
SCALE_FACTOR = 2 / 3

# main method
N_HEURISTICS = 4
SEED_BASE = 1900
SEED_VARIATION_STEP = 10
N_TEST = 5
BUFFER_SIZE = 100
DEF_EPS = 0.05
DEF_MAX_ITER = 5
