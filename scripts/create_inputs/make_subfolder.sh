#!/bin/bash
for T in $(seq 4)
do
  for C in 0 0.5 0.75 1  
  do
    for i in $(seq 1 50)
      do
      mkdir /home/fs02/pmr82_0001/rg727/calfews_input/${T}T_${C}CC
      mkdir /home/fs02/pmr82_0001/rg727/calfews_input/${T}T_${C}CC/$i
      cd /home/fs02/pmr82_0001/rg727/calfews_input/${T}T_${C}CC/$i
      mkdir /home/fs02/pmr82_0001/rg727/calfews_input/${T}T_${C}CC/$i/{1..22}
      cd /home/fs02/pmr82_0001/rg727/calfews_input/
    done
  done
done 


