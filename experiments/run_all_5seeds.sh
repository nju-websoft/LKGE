for dataset in ENTITY RELATION FACT HYBRID
  do
    for method in Snapshot retraining finetune MEAN LAN PNN CWR SI EWC EMR GEM LKGE
      do
        for seed in 11 22 33 44 55
          do
            note=$seed
            python -u main.py -dataset $dataset -gpu 0 -lifelong_name $method -learning_rate 0.0001 -batch_size 2048 -seed $seed -note $note
          done
      done
  done