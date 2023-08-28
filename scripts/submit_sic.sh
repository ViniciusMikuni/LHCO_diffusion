#!/bin/sh
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -n 16
#SBATCH --ntasks-per-node 4
#SBATCH --gpus-per-task 1
#SBATCH -t 02:00:00
#SBATCH -A m3246
#SBATCH --gpu-bind=none
#SBATCH --array=500,1000,2000,3000,4000,5000,6000,7000,10000

module load tensorflow
echo python classify.py --SR --nsig ${SLURM_ARRAY_TASK_ID} --nid $1
srun python classify.py --SR  --nsig ${SLURM_ARRAY_TASK_ID} --nid $1
