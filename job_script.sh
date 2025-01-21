#!/bin/sh
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=10:00:00
#$ -N fegrow
#$ -M kohbanye@gmail.com

. /etc/profile.d/modules.sh
source $HOME/.zshrc
conda activate fegrow

python main.py
