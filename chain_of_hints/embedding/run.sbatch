#!/bin/bash
#SBATCH --job-name=en
#SBATCH --partition=general
#SBATCH --cpus-per-task=2      
#SBATCH --mem=60G
#SBATCH --time=6:00:00        
#SBATCH --mail-type=ALL
python pair.py
