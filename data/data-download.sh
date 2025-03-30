#!/bin/bash
# Download directories vars
root_dl="k400"
root_dl_targz="k400_targz"
# Make root directories
[ ! -d $root_dl ] && mkdir $root_dl
[ ! -d $root_dl_targz ] && mkdir $root_dl_targz

# Download train first item only
curr_dl=${root_dl_targz}/train
url=https://s3.amazonaws.com/kinetics/400/train/k400_train_path.txt
[ ! -d $curr_dl ] && mkdir -p $curr_dl
# Download the list file first
curl -s $url -o /tmp/train_list.txt
# Get only the first URL and download it
first_url=$(head -n 1 /tmp/train_list.txt)
echo "Downloading first train item: $first_url"
curl -C - "$first_url" -o "$curr_dl/$(basename $first_url)"

# Download validation first item only
curr_dl=${root_dl_targz}/val
url=https://s3.amazonaws.com/kinetics/400/val/k400_val_path.txt
[ ! -d $curr_dl ] && mkdir -p $curr_dl
# Download the list file first
curl -s $url -o /tmp/val_list.txt
# Get only the first URL and download it
first_url=$(head -n 1 /tmp/val_list.txt)
echo "Downloading first validation item: $first_url"
curl -C - "$first_url" -o "$curr_dl/$(basename $first_url)"

# Download test first item only
curr_dl=${root_dl_targz}/test
url=https://s3.amazonaws.com/kinetics/400/test/k400_test_path.txt
[ ! -d $curr_dl ] && mkdir -p $curr_dl
# Download the list file first
curl -s $url -o /tmp/test_list.txt
# Get only the first URL and download it
first_url=$(head -n 1 /tmp/test_list.txt)
echo "Downloading first test item: $first_url"
curl -C - "$first_url"