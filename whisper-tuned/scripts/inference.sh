#!/usr/bin/bash

#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=24:00:00

ID=`pwd | xargs -I{} basename {}`

expdir=$PWD
# The data directory
DATADIR=/home/smg2/share/shared-data/Antideepfake-vocoded/Whisper-vocoded
# The configuration file
CONFIG=$PWD/hparams/exp_voc_mls.yaml
# The checkpoint folder
CHECKPOINT_FOLDER=/home/smg2/share/paper-materials/wang-interspeech-submitted-2026-whisper/experiments/spoofing-loc-vocoded-aug-2/cp_whisper_hparams_exp_voc_mls.yaml/artifacts/checkpoint

# load the env
source ~/.bashrc
conda activate partialwhispeer

cd ${expdir}

# inference
echo "###"
echo "# inference partial av1m"
echo "###"
COM="python main.py ${CONFIG} inference --data_base_dir ${DATADIR}/av1m
	    --eval_list ${DATADIR}/av1m/test.csv
	    --eval_set_name av1m
            --checkpoint_folder ${CHECKPOINT_FOLDER}"
echo ${COM}
eval ${COM}


echo "###"
echo "# inference partial mls_coded"
echo "###"
COM="python main.py ${CONFIG} inference --data_base_dir ${DATADIR}/mls_coded
	    --eval_list ${DATADIR}/mls_coded/csv/v1/eval.csv
	    --eval_set_name mls_coded
            --checkpoint_folder ${CHECKPOINT_FOLDER}"
echo ${COM}
eval ${COM}

echo "###"
echo "# inference partial llampartial"
echo "###"
COM="python main.py ${CONFIG} inference --data_base_dir ${DATADIR}/llamapartialspoof
	    --eval_list ${DATADIR}/llamapartialspoof/eval_subset3000.lst
	    --eval_set_name llamapartial
            --checkpoint_folder ${CHECKPOINT_FOLDER}"
echo ${COM}
eval ${COM}

echo "###"
echo "# inference partial partialedit"
echo "###"
COM="python main.py ${CONFIG} inference --data_base_dir ${DATADIR}/partialedit
	    --eval_list ${DATADIR}/partialedit/eval_subset3000.lst
	    --eval_set_name partialedit
            --checkpoint_folder ${CHECKPOINT_FOLDER}"
echo ${COM}
eval ${COM}

