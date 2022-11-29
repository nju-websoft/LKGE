for dataset in ENTITY RELATION FACT HYBRID
  do
    for lr in 0.0001 0.0005 0.001
      do
        for bs in 1024 2048
          do
            for emb_dim in 100 200
              do
                note=$dataset$lr$bs$emb_dim
                python -u main.py -dataset $dataset -gpu 0 -lifelong_name retraining -skip_previous True -learning_rate $lr -batch_size $bs -emb_dim $emb_dim -seed 1 -note $note
              done
          done
      done
  done