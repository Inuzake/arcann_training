#!/bin/bash
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2024 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
# Created: 2026/02/25
# Last modified: 2026/02/25
#----------------------------------------------
# You must keep the _R_VARIABLES_ in the file.
# You must keep the name file as job_deepmd_train_ARCHTYPE_myHPCkeyword.sh.
#----------------------------------------------
# Project/Account
#SBATCH --account=_R_PROJECT_@_R_ALLOC_
# QoS/Partition/SubPartition
#SBATCH --qos=_R_QOS_
#SBATCH --partition=_R_PARTITION_
#SBATCH -C _R_SUBPARTITION_
# Number of Nodes/MPIperNodes/OpenMPperMPI/GPU
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 10
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
# Walltime
#SBATCH -t _R_WALLTIME_
# Merge Output/Error
#SBATCH -o MACE_Train.%j
#SBATCH -e MACE_Train.%j
# Name of job
#SBATCH -J MACE_Train
# Email
#SBATCH --mail-type FAIL,BEGIN,END,ALL
#SBATCH --mail-user _R_EMAIL_
#

#----------------------------------------------
# Files / Variables - They should not be changed
#----------------------------------------------

FRANKEN_IN_FILE="_R_FRANKEN_INPUT_FILE_"
FRANKEN_OUT_FILE="_R_FRANKEN_OUTPUT_FILE_"
MACE_DATA_DIR="../data"

MACE_CONDA_INSTALL="" #If you don't want to use a specific version of MACE, but rather your own you installed on a conda env, specify the path of this env here

#----------------------------------------------
# Adapt the following lines to your HPC system
#----------------------------------------------

# Go where the job has been launched
cd "${SLURM_SUBMIT_DIR}" || { echo "Could not go to ${SLURM_SUBMIT_DIR}. Aborting..."; exit 1; }

# Check
[ -f "${FRANKEN_IN_FILE}" ] || { echo "${FRANKEN_IN_FILE} does not exist. Aborting..."; exit 1; }

# This part copy the data from the MACE_DATA_DIR to the job folder (because they are one up and they should be in the same folder)
[ -d ${MACE_DATA_DIR} ] || { echo "${MACE_DATA_DIR} does not exist. Aborting..."; exit 1; }
mkdir -p "${SLURM_SUBMIT_DIR}"/data || { echo "Could not create ${SLURM_SUBMIT_DIR}/data. Aborting..."; exit 1; }
{ cp -r ${MACE_DATA_DIR}/* "${SLURM_SUBMIT_DIR}"/data && echo "${MACE_DATA_DIR} copied successfully"; } || { echo "Could not copy ${MACE_DATA_DIR}. Aborting..."; exit 1; }

module purge
module load arch/h100
module load pytorch-gpu/py3/
conda activate franken

franken.autotune --train-path data/training_dataset.extxyz --val-path data/validation_dataset.extxyz --l2-penalty="(-11,-6,5,log)" --force-weight="(0.01,0.99,5,linear)" --metrics energy_MAE forces_MAE energy_RMSE forces_RMSE --seed 42 --ms-gaussian.rng-seed 1337 --jac-chunk-size "10" --run-dir "./results" --backbone=mace --mace.path-or-id "/lustre/fsn1/projects/rech/ihj/use32lq/NNP/MACE/FUND_MOD/mace-omat-0-medium.model" --mace.interaction-block 2 --rf=ms-gaussian --ms-gaussian.num-rf 16384 --ms-gaussian.length-scale-low 8.0 --ms-gaussian.length-scale-high 32.0 --ms-gaussian.length-scale-num 4


sleep 5
exit 0

