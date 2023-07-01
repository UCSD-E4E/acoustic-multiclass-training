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


## EC2 setup 

Before starting anything, run `sudo apt update -y` followed by `sudo apt upgrade -y` to ensure that apt's repository cache is up to date.

Install the aws tools using the command: `sudo apt install awscli s3fs`.


## Creating IAM roles

You need to save a valid access key on EC2 in order to mount an s3 filesystem. To do that, search for IAM in the AWS console, click on users and create a new IAM user. Give them a username. Click attach policies directly and give them `AmazonS3FullAccess`. Create new user. Next you have to get their access token. Click on the new user -> security credentials -> create access key. Give CLI access. Stay on this page.

Open up your SSH session and run the command `aws configure`. Paste in the access key from this user and the secret access key. The region and output format can be left blank.

Alternatively, you can make an IAM role to have AmazonS3FullAccess and assign it to the EC2 instance, but I kept getting permission denied errors.


## Mounting S3 in EC2

Create an empty directory for the mount to go. Such as, `mkdir ~/s3-mount`.

Run the command `s3fs {S3 bucket name} ~/s3-mount`.
