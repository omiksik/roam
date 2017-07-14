#!/bin/bash
#date            :20160908




GROUNDT_LOCATION=$1
RESULTS_LOCATION=$2
NAME_OUT=$3


find "$RESULTS_LOCATION" -name *.png > "~list_results.txt"
sort -V "~list_results.txt" > "~list_results_ordered.txt" 

find "$GROUNDT_LOCATION" -name *.png > "~list_groundtruth.txt"
sort -V "~list_groundtruth.txt" > "~list_groundtruth_ordered.txt" 

eval_cli -gt="~list_groundtruth_ordered.txt" -res="~list_results_ordered.txt" -out="$NAME_OUT"

rm "~list_results.txt"
rm "~list_groundtruth.txt"
rm "~list_results_ordered.txt" 
rm "~list_groundtruth_ordered.txt"

cat "$NAME_OUT"




