for dataset in ENTITY RELATION FACT HYBRID
  do
    for method in LKGE
      do
        for regular_weight in 0.01 0.1 1.0
          do
            for reconstruct_weight in 0.01 0.1 1.0
              do
                note=$regular_weight$reconstruct_weight
                python -u main.py -dataset $dataset -gpu 0 -lifelong_name $method -learning_rate 0.0001 -batch_size 2048 -seed 1 -note $note -regular_weight $regular_weight -reconstruct_weight $reconstruct_weight
              done
          done
      done
  done