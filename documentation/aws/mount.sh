#!/bin/bash

mkdir data
echo -n "Enter the name of the bucket: "
read bucket_name
umount data
s3fs $bucket_name data
echo "Mounting successful"