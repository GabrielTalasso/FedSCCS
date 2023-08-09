#!/bin/bash

CORES=$1

SCENARIO="cologne"

INTERVAL=5

BEGIN=100

START_PROCESS=100

STEP_LENGHT=0.1

RESOURCES_VCS=1

TOTAL_SEEDS="1 2 3 4 5"

SEED_SUMO=1

TASK_RATE="1 2 3 4 5 10 15"

TASK_SIZE=10

CPU_CYCLE=30

SIMULATION_TIME=302

ALGORITHMS="FCFS RANDOM FARID FARIDNEW KIRA EFESTO NANCY TOVEC CRATOS"

DEADLINES="0.5 0.8 1 3 5 7"

# CORES=$1
# INICIO=$2
# FINAL=$3
# ALGORITMOS="MARINA CRATOS FIFO UNC AHP PSO"

# echo "VEC Simulation"
# echo "* Inicio: $INICIO"
# echo "* Final: $FINAL"
# echo "* Numero de cores usados: $CORES"
# echo "* Algoritmos: $ALGORITMOS"

parallel --bar -j $CORES "python3 run.py \
--scenario {1} \
--interval {2} \
--tasks 10 \
--begin $BEGIN \
--end $SIMULATION_TIME \
--radius 2000 \
--resources {3} \
--weight {4} \
--taskrate {5} \
--cpucycle {6} \
--algorithm {7} \
--seed_sumo {8} \
--seed_task {9} \
--deadline {10} \
--startprocess {11} \
--steplenght {12}" ::: $SCENARIO ::: $INTERVAL ::: $RESOURCES_VCS ::: \
$TASK_SIZE ::: $TASK_RATE ::: $CPU_CYCLE ::: \
$ALGORITHMS ::: $SEED_SUMO ::: $TOTAL_SEEDS ::: $DEADLINES ::: $START_PROCESS ::: $STEP_LENGHT