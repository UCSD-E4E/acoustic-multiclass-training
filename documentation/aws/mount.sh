#!/bin/bash

mkdir data
echo -n "Enter the name of the bucket: "
read bucket_name
s3fs $bucket_name data
echo "Mounting successful"