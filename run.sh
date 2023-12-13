#!/bin/bash
#
#SBATCH --job-name=herdingbias
#
#SBATCH --ntasks=1
#SBATCH -p sched_mit_sloan_sinana
#SBATCH -o "slurm-%j.out"
#SBATCH -e "slurm-%j.out"
#SBATCH --time=5-0
#SBATCH --mem-per-cpu=10000
#SBATCH --mail-type=END
#SBATCH --mail-user=jhays@mit.edu

module load /home/software/modulefiles/python/3.9.4
python3.9 -m pip install -r requirements.txt --user
python3.9 run_sims.py --n_runs=5000 --n_listings=100 --crtime

