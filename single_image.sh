#!/bin/bash


PROGRAM="$1"

IMAGE_NAME="$2"

ITERATION_NUMBER="$3"

OUTPUT_FILE="$4"

if [ $# -ne 4 ]
then 
    echo "Usage: bash single_image.sh program-to-execute image-to-process number-of-iterations output-file"
fi

if [ -e "$OUTPUT_FILE" ]
then
    rm "$OUTPUT_FILE"
fi

touch "$OUTPUT_FILE" 
echo "Program name: "$PROGRAM

echo "Image name: "$IMAGE_NAME

echo "Iterations: "$ITERATION_NUMBER

for ((i=0;i<$ITERATION_NUMBER;i++));do
    ((iteration = $i+1));
    echo "Executing: $PROGRAM _____________________________Iteration: $iteration";
    ./$PROGRAM "$IMAGE_NAME" >> "$OUTPUT_FILE";  
done





