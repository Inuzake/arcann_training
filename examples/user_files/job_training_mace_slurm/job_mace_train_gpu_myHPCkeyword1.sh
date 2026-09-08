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

MACE_MODEL_VERSION="_R_MACE_VERSION_"
MACE_IN_FILE="_R_MACE_INPUT_FILE_"
MACE_LOG_FILE="_R_MACE_LOG_FILE_"
MACE_OUT_FILE="_R_MACE_OUTPUT_FILE_"
MACE_DATA_DIR="../data"

MACE_CONDA_INSTALL="" #If you don't want to use a specific version of MACE, but rather your own you installed on a conda env, specify the path of this env here

#----------------------------------------------
# Adapt the following lines to your HPC system
#----------------------------------------------

# Go where the job has been launched
cd "${SLURM_SUBMIT_DIR}" || { echo "Could not go to ${SLURM_SUBMIT_DIR}. Aborting..."; exit 1; }

# Check
[ -f "${MACE_IN_FILE}" ] || { echo "${MACE_IN_FILE} does not exist. Aborting..."; exit 1; }

# This part copy the data from the MACE_DATA_DIR to the job folder (because they are one up and they should be in the same folder)
[ -d ${MACE_DATA_DIR} ] || { echo "${MACE_DATA_DIR} does not exist. Aborting..."; exit 1; }
mkdir -p "${SLURM_SUBMIT_DIR}"/data || { echo "Could not create ${SLURM_SUBMIT_DIR}/data. Aborting..."; exit 1; }
{ cp -r ${MACE_DATA_DIR}/* "${SLURM_SUBMIT_DIR}"/data && echo "${MACE_DATA_DIR} copied successfully"; } || { echo "Could not copy ${MACE_DATA_DIR}. Aborting..."; exit 1; }

# Example to use the DeepMD_MODEL_VERSION variable
if [ ${MACE_MODEL_VERSION} == "0.3.14" ]; then
    # Load the MACE module
    MACE_INSTALL="/lustre/fsn1/worksf/projects/rech/nvs/uht29vt/LAMMPS-SYMMETRIX-PLUMED/env"
    module purge
    module load arch/h100
    module load gcc/12.2.0 cuda/12.8.0 openmpi/4.1.6-cuda
    module load cudnn/9.21.0.82-cuda fftw/3.3.10-mpi-cuda bzip2/1.0.8 ffmpeg/8.1-cuda hdf5/1.12.0-mpi-cuda libpng/1.6.37 netcdf-c/4.7.4-mpi-cuda libjpeg-turbo/2.1.3 gsl/2.7.1 openblas/0.3.20

    conda activate $MACE_INSTALL
    export LD_LIBRARY_PATH="$MACE_INSTALL/lib":$LD_LIBRARY_PATH
elif [ -n "$MACE_CONDA_INSTALL" ]; then
    # Activate the conda environment
    module load conda
    source ${CONDA_PREFIX}/bin/activate
    conda activate ${MACE_CONDA_INSTALL}
else
    echo "MACE version ${MACE_MODEL_VERSION} is not available. Aborting..."
    exit 1
fi

# Run the MACE train
echo "# [$(date)] Running MACE train..."
mace_run_train --config=${MACE_IN_FILE} 1> ${MACE_LOG_FILE} 2> ${MACE_OUT_FILE}
echo "# [$(date)] MACE train finished."

# This are useless files, so we remove them
if [ -f out.json ]; then rm out.json; fi
if [ -f input_v2_compat.json ]; then rm input_v2_compat.json; fi

sleep 2
exit

module purge
module load arch/h100
module load pytorch-gpu/py3/
conda activate franken

franken.autotune\
        --train-path data/training_dataset.extxyz \
        --val-path data/validation_dataset.extxyz \
        --l2-penalty="(-11,-6,5,log)" \
        --force-weight="(0.01,0.99,5,linear)"\
        --metrics energy_MAE forces_MAE energy_RMSE forces_RMSE \
        --seed 42 --ms-gaussian.rng-seed 1337 \
        --jac-chunk-size "10" \
        --run-dir "./results" \
        --backbone=mace --mace.path-or-id "/lustre/fsn1/projects/rech/ihj/use32lq/NNP/MACE/FUND_MOD/mace-omat-0-medium.model" --mace.interaction-block 2 \
        --rf=ms-gaussian --ms-gaussian.num-rf 16384 --ms-gaussian.length-scale-low 8.0 --ms-gaussian.length-scale-high 32.0 --ms-gaussian.length-scale-num 4


sleep 5
exit 0

