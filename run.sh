#!/bin/bash
#
#SBATCH --job-name=herdingbias
#SBATCH --array=401-600
#SBATCH -p sched_mit_sloan_batch
## SBATCH -p sched_mit_sloan_sinana
#SBATCH -o "slurm-%j.out"
#SBATCH -e "slurm-%j.out"
#SBATCH --time=3-0
#SBATCH --mem-per-cpu=30000
#SBATCH --mail-type=END
#SBATCH --mail-user=jhays@mit.edu

module load /home/software/modulefiles/python/3.9.4
python3.9 -m pip install -r requirements.txt --user
#python3.9 run_sims.py --n_runs=200 --n_listings=100 --crtime --output_postfix $SLURM_ARRAY_TASK_ID
#python3.9 run_sims.py --n_runs=200 --n_listings=100 --lrtime --output_postfix $SLURM_ARRAY_TASK_ID
python3.9 run_sims.py --n_runs=50 --n_listings=100 --herding --output_postfix $SLURM_ARRAY_TASK_ID

