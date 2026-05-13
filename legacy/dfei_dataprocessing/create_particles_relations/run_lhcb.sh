#!/bin/bash

# Usage: ./run_analysis.sh <dataset_type> <start> <end>
# Example: ./run_analysis.sh inclusive 0 5000
# This will process files from 0-999, 1000-1999, 2000-2999, 3000-3999, 4000-4999

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <dataset_type> <start> <end>"
    echo "Example: $0 inclusive 0 5000"
    echo "This will process files in steps of 1000"
    exit 1
fi

DATASET_TYPE=$1
START=$2
END=$3
IDENTIFIER=$4
DIR="/afs/cern.ch/work/b/bsevilla/yukai_dfei_ift/dfei_repo/output"
SIM="LHCb"

# Loop through the range in steps of 1000
for ((i=$START; i<$END; i+=1000)); do
    # Calculate the end of this range (i + 999)
    range_end=$((i + 999))
    
    # Format numbers with leading zeros (8 digits)
    start_formatted=$(printf "%08d" $i)
    end_formatted=$(printf "%08d" $range_end)
    
    # Construct the filename
    infile="GeneratedParticles_${DATASET_TYPE}_${start_formatted}_${end_formatted}.root"
    
    echo "Processing: $infile"
    echo main.py --infile "$infile" --dir "$DIR" --sim "$SIM" --identifier "$IDENTIFIER"
    python main.py --infile "$infile" --dir "$DIR" --sim "$SIM" --identifier "$IDENTIFIER"
    
    # Optional: Check if the command was successful
    if [ $? -ne 0 ]; then
        echo "Error processing $infile"
        exit 1
    fi
done

echo "All files processed successfully!"


#bash run_lhcb.sh inclusive 0 83000 0001 