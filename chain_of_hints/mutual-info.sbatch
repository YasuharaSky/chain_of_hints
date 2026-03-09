#!/bin/bash
#SBATCH --job-name=mutual
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1       
#SBATCH --cpus-per-task=2
#SBATCH --mem=80G
#SBATCH --time=12:00:00        
#SBATCH --mail-type=ALL
python mutual-info.py
