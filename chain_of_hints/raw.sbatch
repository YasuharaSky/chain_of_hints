#!/bin/bash
#SBATCH --job-name=raw
#SBATCH --partition=general    
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --time=12:00:00        
#SBATCH --mail-type=ALL 
#SBATCH --error=logs/raw.err
python raw.py
