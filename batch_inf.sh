#!/usr/bin/bash

tag=$(echo $1 | xargs -I{} basename {} .yaml)
echo ${tag}

if [ -z "$2" ]; then
    # If it is unset or empty, set the local variable 'arg2' to an empty string.
    # Note: $2 itself cannot be directly reassigned, so we use a new variable.
    arg2="null"
    modelname="null"
else
    arg2="$2"
    modelname=`echo ${arg2} | xargs -I{} basename {}`
fi


qsub -g ${gid} -o log_${tag}_inf_${modelname} -e log_${tag}_inf_${modelname}_err qsub_inf.sh $1 ${arg2}
