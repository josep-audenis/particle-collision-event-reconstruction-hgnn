#!/bin/bash

# Configuration parameters
INDIR="/eos/lhcb/user/y/yukaiz/DFEI_data/DFEI_LHCb_particles_relations/"
OUTDIR="/eos/user/y/yukaiz/DFEI_IFT/cached_data/"

# File settings
SAMPLE="Bs_Jpsiphi"
identifier=00338678
mode="tst_only"

# Loop range configuration
LOOP_START=0000
LOOP_END=99000
STEP_SIZE=1000

echo "Processing sample: $SAMPLE"

# Loop from LOOP_START to LOOP_END with STEP_SIZE
for i in $(seq $LOOP_START $STEP_SIZE $LOOP_END); do
    # Format numbers with leading zeros (8 digits)
    START=$(printf "%08d" $i)
    END=$(printf "%08d" $((i + STEP_SIZE - 1)))

    FILENAME="Event_data_particles_${SAMPLE}_LHCbcollision_${START}_to_${END}.root"

    echo main.py --indir "$INDIR" --outdir "$OUTDIR" --file "$FILENAME" --identifier "$identifier" --normed --mode "$mode"
    python main.py --indir "$INDIR" --outdir "$OUTDIR" --file "$FILENAME" --identifier "$identifier" --normed --mode "$mode"

    # Optional: check if command succeeded
    if [ $? -ne 0 ]; then
        echo "Error processing $FILENAME"
    fi
done

echo "All processing complete!"
