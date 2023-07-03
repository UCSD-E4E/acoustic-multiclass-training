#!/bin/bash

# This script assumes the user is running Ubuntu on AWS EC2 instance
# Assuming the user has run the following commands

# Ensure system is up to date
echo "Updating system"
sudo apt update -y
sudo apt upgrade -y

# Install miniconda
echo "Installing miniconda"
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Linux-x86_64.sh
if [[ $(sha256sum Miniconda3-py310_23.3.1-0-Linux-x86_64.sh) != aef279d6baea7f67940f16aad17ebe5f6aac97487c7c03466ff01f4819e5a651 ]]; then
  echo "File sha does not match expected sha :P  Exiting"
  exit 1
fi
bash Miniconda3-py310_23.3.1-0-Linux-x86_64.sh
conda env create -f environment.yml
conda activate asid

# Install s3fs
echo "Installing AWS and s3fs"
sudo apt install awscli
sudo apt install s3fs

# Configure aws
echo "Starting aws configure. Please enter your access key and secret key. Leave region and output format default.\n"
aws configure

echo "System setup done. Please run mount.sh to mount your s3 bucket."