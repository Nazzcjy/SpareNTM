
import argparse
import sys
import operator
import math
import codecs
import numpy as np
from scipy import stats
from collections import defaultdict

# parser arguments
desc = "Computes the observed coherence for a given topic and word-count file."
parser = argparse.ArgumentParser(description=desc)
#####################
# positional argument#
#####################
parser.add_argument("--num_topic",type=int, default=50,
                    help="the number of topic")
parser.add_argument("--topic_file", default="",
                    help="file that contains the topics")
parser.add_argument("--metric", default='npmi',
                    help="type of evaluation metric",
                    choices=["pmi", "npmi", "lcp", "npmi_Cv"])
parser.add_argument("--wordcount_file", default="",
                    help="file that contains the word counts")
parser.add_argument("--des_file", default="",
                    help="file that contains the word counts")
parser.add_argument("--num_experiments", type=int, default=1,
                    help="file that contains the word counts")
####################
# optional argument#
####################
parser.add_argument("-t", "--topns", nargs="+", type=int, default=[5, 10, 15],
                    help="list of top-N topic words to consider for computing coherence; e.g. '-t 5 10' means it " + \
                         " will compute coherence over top-5 words and top-10 words and then take the mean of both values." + \
                         " Default = [10]")
args = parser.parse_args()

# parameters
colloc_sep = "_"  # symbol for concatenating collocations

# constants
WTOTALKEY = "utf-8utf-8utf-8utf-8utf-8utf-8utf-8!!<TOTAL_WINDOWS>!!"  # key name for total number of windows (in word count file)

# global variables
window_total = 0  # total number of windows
wordcount = {}  # a dictionary of word counts, for single and pair words
wordpos = {}  # a dictionary of pos distribution

# compute the association between two words
def calc_assoc(word1, word2):
    combined1 = word1 + "|" + word2
    combined2 = word2 + "|" + word1

    combined_count = 0
    if combined1 in wordcount:
        combined_count = wordcount[combined1]
    elif combined2 in wordcount:
        combined_count = wordcount[combined2]
    w1_count = wordcount.get(word1, 0)
    w2_count = wordcount.get(word2, 0)

    if args.metric in ['pmi', 'npmi', 'npmi_Cv']:
        if w1_count == 0 or w2_count == 0 or combined_count == 0:
            result = 0.0
        else:
            try:
                result = math.log10((float(combined_count) * float(window_total)) / float(w1_count * w2_count))
                if args.metric == "npmi" or args.metric == "npmi_Cv":
                    result = result / (-1.0 * math.log10(float(combined_count) / window_total))
            except:
                print()

    elif args.metric == "lcp":
        if combined_count == 0:
            if w2_count != 0:
                result = math.log(float(w2_count) / window_total, 10)
            else:
                result = math.log(float(1.0) / window_total, 10)
        else:
            result = math.log((float(combined_count)) / (float(w1_count)), 10)

    return result

# compute topic coherence given a list of topic words
def calc_topic_coherence(topic_words):
    topic_assoc = []
    for w1_id in range(0, len(topic_words) - 1):
        target_word = topic_words[w1_id]
        # remove the underscore and sub it with space if it's a collocation/bigram
        w1 = " ".join(target_word.split(colloc_sep))
        for w2_id in range(w1_id + 1, len(topic_words)):
            topic_word = topic_words[w2_id]
            # remove the underscore and sub it with space if it's a collocation/bigram
            w2 = " ".join(topic_word.split(colloc_sep))
            if target_word != topic_word:
                topic_assoc.append(calc_assoc(w1, w2))

    return float(sum(topic_assoc)) / len(topic_assoc)

# compute topic uniqueness
def calc_topic_uniqueness(top_words_file):
    # compute the mean of TU
    with open(top_words_file, 'r') as f:
        lines = f.readlines()
    top_words = [l.strip().split() for l in lines]
    top_words = [l[:15] for l in top_words]
    cnt = np.zeros([len(top_words), len(top_words[0])])
    for j in range(len(top_words)):
        for k in range(len(top_words[0])):
            cnt_jk = 0
            for j2 in range(len(top_words)):
                cnt_jk += top_words[j2].count(top_words[j][k])
            cnt[j, k] = cnt_jk
    TU_z = np.mean(1. / cnt, 1)
    TU = np.mean(TU_z)
    return TU

# compute Cv given a list of topic words
def calc_Cv(topic_words):
    t_w = list(set(topic_words))
    T = len(t_w)
    npmi_metric = np.zeros([T, T])
    for w1_id in range(0, len(topic_words) - 1):
        target_word = topic_words[w1_id]
        # remove the underscore and sub it with space if it's a collocation/bigram
        w1 = " ".join(target_word.split(colloc_sep))
        for w2_id in range(w1_id + 1, len(topic_words)):
            topic_word = topic_words[w2_id]
            # remove the underscore and sub it with space if it's a collocation/bigram
            w2 = " ".join(topic_word.split(colloc_sep))
            np_tem = calc_assoc(w1, w2)
            npmi_metric[w1_id, w2_id] = np_tem
            npmi_metric[w2_id, w1_id] = np_tem
    # npmi_metric[np.where(npmi_metric<0.)] = 0.
    npmi_1_T = np.sum(npmi_metric, axis=1)

    def cos_one_word(x):
        a_norm = np.linalg.norm(x)
        if a_norm == 0.:
            return 0.
        return np.sum(x * npmi_1_T) / (a_norm * np.linalg.norm(npmi_1_T))

    return np.mean(np.nan_to_num(np.apply_along_axis(cos_one_word, 1, npmi_metric)))

#######
# main#
#######

# input
if args.num_experiments == 1:
    topic_fs = [args.topic_file]
    des_fs = [args.des_file]
else:
    topic_fs = [args.topic_file.format(str(i)) for i in range(1,args.num_experiments+1)]
    des_fs = [args.des_file.format(str(i)) for i in range(1,args.num_experiments+1)]

topics_lines = []
for tf in topic_fs:
    topic_file = codecs.open(tf, "r", "utf-8")
    topics_lines.append(topic_file.readlines())
# topic_file = codecs.open(args.topic_file, "r", "utf-8")

topics_all = []
for tls in topics_lines:
    for line in tls:
        topics_all.extend(line.strip().split('\t'))
topics_all.append(WTOTALKEY)
topic_set = list(set(topics_all))

wc_file = codecs.open(args.wordcount_file, "r", "utf-8")

# process the word count file(s)
for line in wc_file:
    line = line.strip()
    data = line.split("|")
    if len(data) == 2:
        if data[0] not in topic_set:
            continue
        wordcount[data[0]] = int(data[1])
    elif len(data) == 3:
        if data[0] not in topic_set or data[1] not in topic_set:
            continue
        if data[0] < data[1]:
            key = data[0] + "|" + data[1]
        else:
            key = data[1] + "|" + data[0]
        wordcount[key] = int(data[2])
    else:
        print("ERROR: wordcount format incorrect. Line =", line)
        raise SystemExit

# get the total number of windows
if WTOTALKEY in wordcount:
    window_total = wordcount[WTOTALKEY]

# read the topic file and compute the observed coherence
for i,tls in enumerate(topics_lines):
    mean_TU = calc_topic_uniqueness(topic_fs[i])
    topic_coherence = defaultdict(list)  # {topicid: [tc]}
    topic_tw = {}  # {topicid: topN_topicwords}
    if args.metric == 'npmi_Cv':
        metric_fun = calc_Cv
    else:
        metric_fun = calc_topic_coherence

    for topic_id, line in enumerate(tls): #tfs
        topic_list = line.split()[:max(args.topns)]
        topic_tw[topic_id] = " ".join(topic_list)
        for n in args.topns:
            topic_coherence[topic_id].append(metric_fun(topic_list[:n]))

    # sort the topic coherence scores in terms of topic id
    tc_items = sorted(topic_coherence.items())
    mean_coherence_list = []
    mean_uniqueness_list = []
    with open(des_fs[i],'w',encoding='utf-8') as fw: #
        for item in tc_items:
            topic_words = topic_tw[item[0]].split()
            mean_coherence = np.mean(item[1])
            mean_coherence_list.append(mean_coherence)
            fw.write("[%.2f] (" % mean_coherence+'\n')
            for i in item[1]:
                fw.write("%.2f;" % i + '\n')
            fw.write(")"+ topic_tw[item[0]]+'\n')

        # print the overall topic coherence for all topics
        fw.write("==========================================================================\n")
        fw.write("Average Topic Uniqueness = %.3f" % mean_TU+"\n")
        fw.write("Average Topic Coherence = %.3f" % np.mean(mean_coherence_list)+"\n")
        fw.write("Median Topic Coherence = %.3f" % np.median(mean_coherence_list)+"\n")
print("Finished the topic coherence and uniqueness computing!")

# compute the mean of all experiment
TU = []
co = []
for i in range(len(des_fs)):
    # compute the mean of NPMI and TU
    with open(des_fs[i], 'r', encoding='utf-8') as f:
        lines = f.readlines()
        co.append(float(lines[-2].strip().split('=')[-1]))
        TU.append(float(lines[-3].strip().split('=')[-1]))

if args.num_experiments == 1:
    co_deviation = 0
    TU_deviation = 0
else:
    co_deviation = stats.sem(co)*1.96
    TU_deviation = stats.sem(TU)*1.96

print("The number of topic:", args.num_topic)
print("NPMI", co)
print("the mean and deviation:", sum(co)/len(co), co_deviation, "\n")
print("TU:", TU)
print("the mean and deviation:", sum(TU)/len(TU), TU_deviation, "\n")