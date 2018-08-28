#!/bin/bash

#SBATCH --mail-user=xiaoge.zhang@vanderbilt.edu
#SBATCH --mail-type=all
#SBATCH --nodes=5
#SBATCH --ntasks=10
#SBATCH --time=100:00:00
#SBATCH --mem=100G
#SBATCH --output=model.out

module restore default_module
python hybrid_model.py

