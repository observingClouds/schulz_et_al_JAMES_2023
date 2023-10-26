#!/bin/bash
#SBATCH --partition=compute
#SBATCH --account=mh0010
#SBATCH --time=08:00:00
#SBATCH --exclusive

python calc_TOAfluxes.py

