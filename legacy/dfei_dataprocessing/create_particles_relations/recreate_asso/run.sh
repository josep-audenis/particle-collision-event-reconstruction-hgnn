#!/bin/bash

start=77000
end=78000
bunch=1000
max_start=79000

while [ $start -le $max_start ]
do
    echo "Running: ./main -d Bd_Dpi -s $start -e $end -b $bunch -m true" # ture or miasso
      ./main -d Bd_Dpi -s $start -e $end -b $bunch -m true

    start=$((start + bunch))
    end=$((end + bunch))
done



