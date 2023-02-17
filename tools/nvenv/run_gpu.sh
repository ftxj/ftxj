#! /bin/bash
cluster="computelab"
gpu="a100"
timeout="4:00:00"


while getopts 'I:DG:' OPT; do
    case $OPT in
        I) img="$OPTARG";;
        D) cluster="dlcluster";;
        G) gpu="$OPTARG";;
    esac
done

ssh_cluster_cmd=""

binary=""
if [ "$cluster" == "dlcluster" ]; then
  ssh_cluster_cmd="ssh dlcluster"
  binary="/usr/bin/srun"
else
  binary="/home/scratch.svc_compute_arch/release/crun/0.1.220707084558.1/crun/crun"
fi

if [ "$cluster" == "computelab" ];then
  chip="g"${gpu}
  #crun -i -s bash -ex -d /home/builds/release/display/x86_64/510.02/NVIDIA-Linux-x86_64-510.02.run -q 'node=ipp1-2159' -t 240
  # cmd=$binary' -i -s bash -q "chip='$chip' and partition=all" -t '$timeout
  cmd=$binary' -i -q chip='$chip' -t '$timeout' '
elif [ "$cluster" == "dlcluster" ];then
  timeout="8:00:00"
  partition="a100"
  if [ "$gpu" == "a100" ];then
    partition="a100"
  else
    partition="dgx1v"
  fi
  cmd=$binary' -p '$partition' -t '$timeout' --pty /bin/bash'
else
  echo "unknow cluster name"
fi


if [ "$ssh_cluster_cmd" != "" ];then
    echo $ssh_cluster_cmd
    eval $ssh_cluster_cmd
fi
echo $cmd

eval $cmd