#!/bin/bash

# This script assumes the user is running Ubuntu on AWS EC2 instance
# Assuming the user has run the following commands

# Ensure system is up to date
echo "Updating system"
sudo apt-get update
sudo apt-get upgrade -y

# Install miniconda
echo "Installing miniconda"

miniconda_file="Miniconda3-py310_23.3.1-0-Linux-x86_64.sh"
miniconda_checksum="aef279d6baea7f67940f16aad17ebe5f6aac97487c7c03466ff01f4819e5a651"
echo "$miniconda_file $miniconda_checksum"

rm "$miniconda_file"
wget "https://repo.anaconda.com/miniconda/$miniconda_file"
if ! echo "$miniconda_checksum  $miniconda_file" | sha256sum --check ; then
    echo "SHA 256 checksum failed" >&2
    exit 1
fi

bash $miniconda_file
# Temporarily update PATH, it will be updated qutomatically in .bashrc
export PATH="$HOME/miniconda3/bin:$PATH"
conda env create -f environment.yml
conda activate asid

# Install s3fs
echo "Installing AWS and s3fs"
sudo apt-get install awscli
sudo apt-get install s3fs

# Initialize repository install
pip install -e .
python -m pyha_analyzer.config

echo "System setup done."
