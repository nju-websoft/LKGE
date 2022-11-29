for dataset in FACT
  do
    for method in retraining finetune PNN CWR SI EMR GEM Snapshot EWC LKGE
      do
        for seed in 11
          do
            note=$seed
            python -u main.py -dataset $dataset -gpu 0 -lifelong_name $method -learning_rate 0.0001 -batch_size 2048 -regular_weight 0.01 -reconstruct_weight 1.0 -seed $seed -note $note
          done
      done
  done