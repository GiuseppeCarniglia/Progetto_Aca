#!/bin/bash

PROGRAM="$1"

IMAGE_DIR="$2"

OUTPUT_FILE="$3"

if [ $# -ne 3 ]
then
    echo "Usage: bash multiple_images.sh program_to_run directory_of_images output_file"
fi


if [ -e "$OUTPUT_FILE" ]
then
    rm "$OUTPUT_FILE"
fi

touch "$OUTPUT_FILE"

for filename in $IMAGE_DIR/*.jpg;do
    [[ -e "$filename" ]] || continue;
    echo "Executing: $PROGRAM on image: "$filename"";
    ./$PROGRAM "$filename" >> "$OUTPUT_FILE";
done



