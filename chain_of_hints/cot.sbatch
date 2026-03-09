#!/bin/bash
#SBATCH --job-name=cot
#SBATCH --partition=general 
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --time=12:00:00        
#SBATCH --mail-type=ALL
#SBATCH --error=logs/cot.err
python cot.py
