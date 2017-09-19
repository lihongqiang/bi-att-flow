#CUDA_VISIBLE_DEVICES=0  nohup python -m basic.cli --cluster --len_opt --data_dir=data/EQnA/no_sent_token --model_name=EQnA --sentece_token=False --mode train --noload  1>logs/EQnA-0708.out 2>&1 &
# nohup python -m basic.cli --cluster --len_opt --data_dir=data/EQnA/no_sent_token/qp_data_w2v/ --out_dir=/home/t-honli/bi-att-flow/out/EQnA/no_sent_token/q2pemb --model_name=EQnA --sentece_token=False --mode train --noload  1>logs/EQnA_q2pemb.out 2>&1 &

# train
# nohup python -m basic.cli --cluster --len_opt --data_dir=data/EQnA/no_sent_token/ --out_dir=/home/t-honli/bi-att-flow/out/EQnA/no_sent_token/small --model_name=EQnA --sentece_token=False --mode train --noload  1>logs/EQnA_small.out 2>&1 &

# load and retrain
# nohup python -m basic.cli --cluster --len_opt --data_dir=data/EQnA_multiQuery --out_dir=/home/t-honli/bi-att-flow/out/EQnA_retrain --model_name=EQnA --sentece_token=False --mode train --load_path=/home/t-honli/bi-att-flow/out/EQnA_retrain/save/save-46000 --batch_size=30 --val_num_batches=0 --num_steps=30000  --shared_path=/home/t-honli/bi-att-flow/out/EQnA_retrain/shared.json 1>logs/EQnA_retain.out 2>&1 &

python -m basic.cli --cluster --len_opt --data_dir=data/EQnA_multiQuery --out_dir=/home/t-honli/bi-att-flow/out/EQnA_retrain_fact_finetune --model_name=EQnA --sentece_token=False --mode train --load_path=/home/t-honli/bi-att-flow/out/EQnA_retrain_fact_finetune/save/save-52000 --batch_size=60 --val_num_batches=0 --num_steps=120000 --retrain=True
