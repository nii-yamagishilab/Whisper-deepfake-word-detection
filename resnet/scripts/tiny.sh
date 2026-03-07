#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=24:00:00

ID=`pwd | xargs -I{} basename {}`

expdir=$PWD
DATADIR=$PWD/../data/tiny
CONFIG=$PWD/hparams/tiny.yaml

source ~/.bashrc
conda activate partialwhispeer

cd ${expdir}

# training
echo "###"
echo "# training"
echo "###"
COM="python main.py ${CONFIG} train --data_base_dir ${DATADIR}"
echo ${COM}
eval ${COM}

# inference
echo "###"
echo "# inference"
echo "###"
COM="python main.py ${CONFIG} inference --data_base_dir ${DATADIR}"
echo ${COM}
eval ${COM}
