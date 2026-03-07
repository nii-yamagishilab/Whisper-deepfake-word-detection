#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=24:00:00

ID=`pwd | xargs -I{} basename {}`

expdir=$PWD

source ~/.bashrc
conda activate partialwhispeer

cd ${expdir}
if [[ -z "$3" ]]; then
    echo "Skip training"
else	
    python main.py $1 train
fi

com="python main.py $1 inference \
	    --eval_list ${WORK}/data/other_testsets/mls_coded/csv/v1/eval.csv \
	    --eval_set_name mls_coded \
	    --data_base_dir ${WORK}/data/other_testsets/mls_coded"
echo "${com}"
eval "${com}"

com="python main.py $1 inference \
	    --eval_list ${WORK}/data/protocols_v3/eval.csv \
	    --eval_set_name tedx_v3 \
	    --data_base_dir ${WORK}/data"
echo "${com}"
eval "${com}"

command="python main.py $1 inference \
	    --eval_list ${WORK}/data/other_testsets/partialedit/eval_subset3000.lst \
	    --eval_set_name partialedit_sub \
	    --data_base_dir ${WORK}/data/other_testsets/partialedit"

echo ${command}
eval ${command}

com="python main.py $1 inference \
	    --eval_list ${WORK}/data/other_testsets/av1m/test.csv \
	    --eval_set_name av1m_sub \
	    --data_base_dir ${WORK}/data/other_testsets"
echo "${com}"
eval "${com}"

com="python main.py $1 inference \
	    --eval_list ${WORK}/data/other_testsets/lavdf/test_3k.csv \
	    --eval_set_name lavdf_sub \
	    --data_base_dir ${WORK}/data/other_testsets"
echo "${com}"
eval "${com}"

com="python main.py $1 inference --checkpoint $2 \
	    --eval_list ${WORK}/data/other_testsets/llamapartialspoof/eval_subset3000.lst \
	    --eval_set_name llamapartialspoof_sub \
	    --data_base_dir ${WORK}/data/other_testsets/llamapartialspoof"
echo "${com}"
eval "${com}"
