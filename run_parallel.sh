#!/bin/bash

CORES=$1

DATASET="MNIST"

NCLIENTS=25

NROUNDS=100

NCLUSTERS="1 5 10 20"

CLUSTERING="True"

CLUSTERROUND=10

NONIID="True"

SELECTIONMETHOD="All POC Random Less_Selected"

CLUSTERMETRIC="CKA weights"

METRICLAYER="-1"

POCPERCOFCLIENTS=0.5

CLUSTERMETHOD="HC Random KCenter"

parallel --bar -j $CORES "python3 simulation.py --dataset {1} --nclients {2} \
 --nrounds {3} --nclusters {4} --clustering {5} --clusterround {6} --noniid {7} \
 --selectionmethod {8} --clustermetric {9} --metriclayer {10} --pocpercofclients {11} \
 --clustermethod {12} " ::: $DATASET ::: $NCLIENTS ::: $NROUNDS ::: \
 $NCLUSTERS ::: $CLUSTERING ::: $CLUSTERROUND ::: $NONIID :::\
 $SELECTIONMETHOD ::: $CLUSTERMETRIC ::: $METRICLAYER :::\
 $POCPERCOFCLIENTS ::: $CLUSTERMETHOD
