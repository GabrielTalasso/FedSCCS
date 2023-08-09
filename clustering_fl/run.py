#!/usr/bin/python3
# title: run.py
# author: Joahannes Costa <joahannes@lrc.ic.unicamp.br
# date: 22.09.2021

import argparse
import sys
import os

parser = argparse.ArgumentParser(description='Simulator for Task Scheduling in Vehicular Edge Computing ')
parser.add_argument("-a", "--dataset", dest="dataset", help="", metavar="DATASET")
parser.add_argument("-b", "--nclients", dest="nclients", default=10, help="", metavar="NCLIENTS")
parser.add_argument("-c", "--nrounds", dest="nrounds", default=1, help="", metavar="NROUNDS")
parser.add_argument("-d", "--nclusters", dest="nclusters", default=1, help="", metavar="NCLUSTERS")
parser.add_argument("-e", "--clustering", dest="clustering", default=3600, help="", metavar="CLUSTERING")
parser.add_argument("-f", "--clusterround", dest="clusterround", default=2000, help="", metavar="CLUSTEROUND")
parser.add_argument("-g", "--noniid", dest="noniid", default=1, help="", metavar="NONIID")


options = parser.parse_args()

print("+-------------------------------+")
print("+        VEC Simulation         +")
print("+-------------------------------+")
print(" Scenario:",options.scenario)
print(" Interval:",options.interval)
print(" Tasks:",options.tasks)
print(" Begin:",options.begin)
print(" End:",options.end)
print(" Radius:",options.radius)
print(" Resources:",options.resources)
print(" Weight:",options.weight)
print(" Rate:",options.taskrate)
print(" CPU:",options.cpucycle)
print(" Algorithm:",options.algorithm)
print(" SeedSUMO:",options.seed_sumo)
print(" SeedTask:",options.seed_task)
print(" Deadline:",options.deadline)
print(" StartProcess:",options.start_process)
print(" StepLenght:",options.step_lenght)
print("+-------------------------------+")

# simulation log
current_config = str(options.algorithm) + '_TASKS_' + str(options.tasks) + '_RESOURCES_' + str(options.resources) + '_SIZE_' + str(options.weight) + '_RATE_' + str(options.taskrate) + '_CPU_' + str(options.cpucycle) + '_DEADLINE_' + str(options.deadline) + '_SEED-SUMO_' + str(options.seed_sumo) + '_SEED-TASK_' + str(options.seed_task) + '.txt'

# replace file
if os.path.exists('log/' + current_config):
	os.system('rm log/' + current_config)

os.system('python src/main.py --scenario scenario/cologne/cologne100.sumocfg --network scenario/cologne/cologne2.net.xml --interval ' + str(options.interval) + ' --requests ' + str(options.tasks) + ' --begin ' + str(options.begin) + ' --end ' + str(options.end) + ' --output output/cologne-output.xml --logfile log/' + current_config + ' --summary summary/cologne.summary.xml --radius ' + str(options.radius) + ' --resource ' + str(options.resources) + ' --weight ' + str(options.weight) + ' --rate ' + str(options.taskrate) + ' --megacycles ' + str(options.cpucycle) + ' --algorithm ' + str(options.algorithm) + ' --seed_sumo ' + str(options.seed_sumo) + ' --seed_task ' + str(options.seed_task) + ' --deadline ' + str(options.deadline) + ' --start_process ' + str(options.start_process) + ' --step_lenght ' + str(options.step_lenght))

# parallel -j2 'python run.py -a {1} -b {2} -c {3} -d {4} -e {5} -f {6} -g {7} -i {8} -j {9} -k {10} -l {11} -m {12}' ::: cologne ::: 10 ::: 10 ::: 20 ::: 30 ::: 999 ::: 1 ::: 10 ::: 1 ::: 10 ::: MARINA ::: 1 2

# testes
# python run.py --scenario cologne --interval 5 --tasks 10 --begin 100 --end 110 --radius 2000 --resources 1 --weight 10 --taskrate 1 --cpucycle 10 --algorithm MARINA --seed_sumo 1 --seed_task 1 --deadline 7 --startprocess 110 --steplenght 1