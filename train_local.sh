#!/bin/sh

image=$1

mkdir -p test_dir/model
mkdir -p test_dir/output

rm -f -r test_dir/model/*
rm -f -r test_dir/output/*

docker run -v $(pwd)/test_dir:/opt/ml --rm ${image} train
