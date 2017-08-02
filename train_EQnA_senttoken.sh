CUDA_VISIBLE_DEVICES=3 nohup python -m basic.cli --batch_size=20 --val_num_batches=20 --cluster --len_opt --data_dir=data/EQnA/sent_token --model_name=EQnA --sentece_token=True --mode train --noload  1>logs/EQnA_senttoken.out 2>&1 &

