#!/bin/bash

mkdir data
echo -n "Enter the name of the bucket: "
read bucket_name
umount data
s3fs $bucket_name data -o iam_role="S3_Access"