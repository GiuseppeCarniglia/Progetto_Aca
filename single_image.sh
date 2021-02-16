#!/bin/bash


PROGRAM="$1"

IMAGE_NAME="$2"

N_OF_LEVELS="$3"

ITERATION_NUMBER="$4"

OUTPUT_FILE="$5"

if [ $# -ne 5 ]
then 
    echo "Usage: bash single_image.sh program-to-execute image-to-process number_of_levels number-of-iterations output-file"
fi

if [ -e "$OUTPUT_FILE" ]
then
    rm "$OUTPUT_FILE"
fi

touch "$OUTPUT_FILE"
 
echo "Program name: "$PROGRAM

echo "Image name: "$IMAGE_NAME

echo "Number of levels: "$N_OF_LEVELS

echo "Iterations: "$ITERATION_NUMBER

for ((i=0;i<$ITERATION_NUMBER;i++));do
    ((iteration = $i+1));
    echo "Executing: $PROGRAM _____________________________Iteration: $iteration";
    ./$PROGRAM "$IMAGE_NAME" "$N_OF_LEVELS" >> "$OUTPUT_FILE";  
done





