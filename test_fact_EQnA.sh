#python -m basic.cli --len_opt --cluster  --topk=3 --out_dir=out/EQnA/03-07-2017 --data_dir=data/EQnA --model_name=EQnA
#python -m basic.cli_test --len_opt --cluster  --topk=3 --out_dir=out/EQnA/03-07-2017 --data_dir=data/EQnA/sent_token --model_name=EQnA
#python -m basic.cli_test --len_opt --cluster  --topk=0 --out_dir=/home/t-honli/bi-att-flow/out/EQnA/no_sent_token/18-07-2017 --data_dir=data/EQnA/no_sent_token --model_name=EQnA
#python -m basic.cli_test --len_opt --cluster  --topk=0 --out_dir=/home/t-honli/bi-att-flow/out/EQnA/sent_token/18-07-2017 --data_dir=data/EQnA/sent_token --model_name=EQnA
#python -m basic.cli_test --len_opt --cluster  --topk=3 --out_dir=/home/t-honli/bi-att-flow/out/EQnA/sent_token/18-07-2017 --data_dir=data/EQnA/sent_token --model_name=EQnA
#python -m basic.cli_test --len_opt --cluster  --topk=3 --out_dir=/home/t-honli/bi-att-flow/out/EQnA/no_sent_token/18-07-2017 --data_dir=data/EQnA/no_sent_token --model_name=EQnA
#python -m basic.cli_test --len_opt --cluster  --topk=0 --out_dir=/home/t-honli/bi-att-flow/out/EQnA/03-07-2017 --data_dir=data/EQnA/no_sent_token --model_name=EQnA
 python -m basic.cli_test --len_opt --cluster  --topk=0 --out_dir=/home/t-honli/bi-att-flow/out/fact_EQnA --data_dir=data/fact_EQnA/ --load_path=/home/t-honli/bi-att-flow/out/fact_EQnA/save/save-45000 --model_name=fact_EQnA

