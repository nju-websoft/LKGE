for method in EWC SI
  do
    for dataset in ENTITY RELATION FACT HYBRID
      do
        for seed in 11 22 33 44 55
          do
            for weight in 0.01 0.1 1.0
              do
                note=$weight$seed
                python -u main.py -dataset $dataset -gpu 1 -lifelong_name $method -learning_rate 0.0001 -batch_size 2048 -seed $seed -note $note -regular_weight $weight
              done
          done
      done
    done