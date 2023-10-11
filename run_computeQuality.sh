#!/bin/bash

#script that computes the observed coherence (pointwise mutual information, normalised pmi or log 
#conditional probability)
#steps:
#1. sample the word counts of the topic words based on the reference corpus
#2. compute the observed coherence using the chosen metric


#parameters

metric="npmi" #evaluation metric: pmi, npmi or lcp npmi_Cv
ref_corpus_dir="./ref_corpus"
data_name="20ng"
K=50

#compute the word occurrences
echo "Computing word occurrence..."

topic_file="./output/${data_name}.topWords"
wordcount_file="./output/${data_name}_wc.txt"
python ComputeWordCount.py --topic_file=$topic_file --ref_corpus_dir=$ref_corpus_dir > $wordcount_file

#compute the topic observed coherence
echo "Computing the observed coherence..."
topic_file="D./output/${data_name}.topWords"
wordcount_file="./output/${data_name}_wc.txt"
oc_file="./output/${data_name}_quality.txt"
python ./ComputeTopicQuality.py --topic_file=$topic_file --num_topic=${K} --metric=$metric\
--wordcount_file=$wordcount_file --des_file=$oc_file --num_experience=1 -t 5 10 15

