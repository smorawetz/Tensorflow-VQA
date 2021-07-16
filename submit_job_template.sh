#!/bin/bash
#SBATCH --job-name=do_tensorflow_neural_annealing_LREPLACE_WITH_SIZE_TREPLACE_WITH_TEMP_NhREPLACE_WITH_HIDDEN_NsREPLACE_WITH_SAMPLES_NaREPLACE_WITH_ANNEALS_seedREPLACE_WITH_SEED_maxREPLACE_WITH_MAXCHANGE
#SBATCH --account=rrg-rgmelko-ab
#SBATCH --time=REPLACE_WITH_TIME
#SBATCH --mem=4G
#SBATCH --output=/home/stewmo/fall_2020_phys437/slurm_output/%x.out
#SBATCH --error=/home/stewmo/fall_2020_phys437/slurm_errors/%x.out

module load python/3.6
#virtualenv --no-download tensorflow
#source tensorflow/bin/activate
#pip install --no-index --upgrade pip
#pip install --no-index tensorflow_cpu=1.15

# Prepare environment
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r scripts/requirements.txt

# Run data generation
python scripts/run_tensorflow.py REPLACE_WITH_SIZE REPLACE_WITH_HIDDEN REPLACE_WITH_SAMPLES REPLACE_WITH_ANNEALS REPLACE_WITH_SEED REPLACE_WITH_MAXCHANGE REPLACE_WITH_TEMP
