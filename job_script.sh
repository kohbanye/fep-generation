#!/bin/sh
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=24:00:00
#$ -N fegrow
#$ -M kohbanye@gmail.com

DATA_IDX=$1

. /etc/profile.d/modules.sh
source $HOME/.zshrc
conda activate fegrow

python main.py --data_index $DATA_IDX
