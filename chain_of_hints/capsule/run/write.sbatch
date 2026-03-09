#!/bin/bash
#SBATCH --job-name=write
#SBATCH --partition=general    
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --time=12:00:00        
#SBATCH --mail-type=ALL 
python write.py
