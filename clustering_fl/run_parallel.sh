#!/bin/bash

CORES=$1

DATASET="MotionSense"

NCLIENTS=12

NROUNDS=5

NCLUSTERS="2 5"

CLUSTERING="True"

CLUSTERROUND=2

NONIID="True"

SELECTIONMETHOD="All POC Random"

CLUSTERMETRIC="CKA weights"

METRICLAYER="-1 -2 1"

POCPERCOFCLIENTS=0.5

CLUSTERMETHOD="KCenter Affinity HC Random"

parallel --bar -j $CORES "python3 simulation.py --dataset {1} --nclients {2} \
 --nrounds {3} --nclusters {4} --clustering {5} --clusterround {6} --noniid {7} \
 --selectionmethod {8} --clustermetric {9} --metriclayer {10} --pocpercofclients {11} \
 --clustermethod {12} " ::: $DATASET ::: $NCLIENTS ::: $NROUNDS ::: \
 $NCLUSTERS ::: $CLUSTERING ::: $CLUSTERROUND ::: $NONIID :::\
 $SELECTIONMETHOD ::: $CLUSTERMETRIC ::: $METRICLAYER :::\
 $POCPERCOFCLIENTS ::: $CLUSTERMETHOD