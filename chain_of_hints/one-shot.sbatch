#!/bin/bash
#SBATCH --job-name=few-shot
#SBATCH --partition=general    
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --time=12:00:00        
#SBATCH --mail-type=ALL
python few-shot.py
python one-shot.py
