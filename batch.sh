#!/usr/bin/bash

tag=$(echo $1 | xargs -I{} basename {} .yaml)
echo ${tag}

qsub -g ${gid} -o log_${tag} -e log_${tag}_err qsub.sh $1
