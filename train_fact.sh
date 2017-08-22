#CUDA_VISIBLE_DEVICES=0  nohup python -m basic.cli --cluster --len_opt --data_dir=data/EQnA/no_sent_token --model_name=EQnA --sentece_token=False --mode train --noload  1>logs/EQnA-0708.out 2>&1 &
# nohup python -m basic.cli --cluster --len_opt --data_dir=data/EQnA/no_sent_token/qp_data_w2v/ --out_dir=/home/t-honli/bi-att-flow/out/EQnA/no_sent_token/q2pemb --model_name=EQnA --sentece_token=False --mode train --noload  1>logs/EQnA_q2pemb.out 2>&1 &
nohup python -m basic.cli --cluster --len_opt --data_dir=data/fact/ --out_dir=/home/t-honli/bi-att-flow/out/fact --model_name=fact --sentece_token=False --mode train --noload  1>logs/fact.out 2>&1 &


