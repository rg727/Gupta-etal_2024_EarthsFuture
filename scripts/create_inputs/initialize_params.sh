#!/bin/bash
for T in 1 2 3 
do 
for i in {1..50}
  do
    cd /home/fs02/pmr82_0001/rg727/calfews_input/${T}T_1CC/$i/6
    cp /home/fs02/pmr82_0001/rg727/calfews_input/base_inflows_6.json . 
    mv base_inflows_6.json base_inflows.json
    replacement="sacsma_paleo\": \"calfews_input/${T}T_1CC/$i/6/6.csv\""
    sed -i "s#sacsma_paleo\": \"calfews_src/data/input/sacsma_data_paleo_no_snow.csv\"#$replacement#" base_inflows.json
done
done    