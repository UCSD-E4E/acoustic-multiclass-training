# AWS Setup

This guide provides an explanation on how to setup Amazon Web Services S3 storage and EC2 cloud compute for machine learning uses.

Before doing anything, log into the aws console at [https://aws.amazon.com/console/](https://aws.amazon.com/console).


## Creating S3

S3 in the Amazon world stands for Amazon Simple Storage Service. It is basically a place to store files.

To create an S3 instance, search for S3 at the top of the page. Create a new S3 bucket and give it a name. Leave all other settings as they are. On the S3 dashboard, you can upload any files, but it will likely be more useful to mount it to an EC2 filesystem as we will do later.


## Creating EC2

EC2 in the Amazon world stands for Elastic Container 2. It is basically a virtual machine.

To create an EC2 instance, search for EC2 at the top of the page. Create a new EC2 instance by giving it a name, selecting an operating system (preferably Ubuntu), and an instance type. While on this page, create a key pair so you can have an SSH key for the next step.


## Logging in through SSH 

If you didn't make an SSH key on the previous step, open the find Network & Security -> Key Pairs on the left panel. Create a new key pair with the permissions you want. 

Store the key file in a secure location and on linux run `sudo chmod 400 {filename}.pem` to make it read only for just your user. To connect run `ssh -i "path/to/key.pem" ubuntu@{servername}.compute.amazonaws.com`. You can find the domain name in the EC2 dashboard.


## Creating IAM roles

You need to create an IAM role that gives the server access to the S3 instance. Search for IAM in the AWS console, click on roles and create a new IAM role. Click AWS Service, EC2, then next. Add the policy `AmazonS3FullAccess`. Click next. Call the role `"S3_Access"` for the mounting script to run properly. Create role.

Go to the EC2 instance by searching for EC2, click running instances and find your instance. Click on `Actions -> Security -> Modify IAM Roles`. Give the `S3_Access` role to the compute server. 


## Run setup script

Before running the setup script, you need to install the repo onto the machine.
```bash
git clone https://github.com/UCSD-E4E/acoustic-multiclass-training.git
cd acoustic-multiclass-training
```

Run the setup and mount scripts using
```bash
bash documentation/aws/setup.sh
bash documentation/aws/mount.sh
```

This setup script will
1. Update the machine
2. Install miniconda, check its SHA checksum, and create the asid environment
3. Install aws-cli and s3 file system
4. Configure aws setup so you can enter access keys


## Manual setup

Below are the instructions for manual setup if you need to reference them.

### EC2 setup 

Before starting anything, run `sudo apt update -y` followed by `sudo apt upgrade -y` to ensure that apt's repository cache is up to date.

Install the aws tools using the command: `sudo apt install awscli s3fs`.

## IAM roles

Open up your SSH session and run the command `aws configure`. Paste in the access key from this user and the secret access key. The region and output format can be left blank.

Alternatively, you can make an IAM role to have AmazonS3FullAccess and assign it to the EC2 instance, but I kept getting permission denied errors.

### Mounting S3 in EC2

Create an empty directory for the mount to go. Such as, `mkdir ~/s3-mount`.

Run the command `s3fs {S3 bucket name} ~/s3-mount`.
