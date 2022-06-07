cd ~/code/CogGCN/GF_CF-main/Table3\&4

# lgcn别人的复现
nohup python3 main.py --dataset="aminer" --topks="[20,5,10,50]" --model "lgn" --gpu_id 7  > ./log/gfcf_lgn_aminer.log 2>&1 &
nohup python3 main.py --dataset="citeulike" --topks="[20]" --model "lgn" --gpu_id 3 > ./log/gfcf_lgn_citeulike.log 2>&1 &

nohup python3 main.py --dataset="aminer" --topks="[20,5,10,50]" --simple_model "gf-cf" --gpu_id 0 > ./log/gfcf_aminer.log 2>&1 &
nohup python3 main.py --dataset="amazon" --topks="[20,5,10,50]" --simple_model "gf-cf" --gpu_id 0 > ./log/gfcf_amazon-book.log 2>&1 &
nohup python3 main.py --dataset="gowalla" --topks="[20,5,10,50]" --simple_model "gf-cf" --gpu_id 0 > ./log/gfcf_hgowalla.log 2>&1 &
nohup python3 main.py --dataset="yelp2018" --topks="[20,5,10,50]" --simple_model "gf-cf" --gpu_id 0 > ./log/gfcf_telp2018.log 2>&1 &
nohup python3 main.py --dataset="ml-1m" --topks="[20,5,10,50]" --simple_model "gf-cf" --gpu_id 1 > ./log/gfcf_ml1m.log 2>&1 &


# TO 宇翔 and 嫣嫣
# run lightGCN：
python3 main.py --dataset="aminer" --topks="[20,5,10,50]" --model "lgn" --gpu_id 0
# run GF-CF
python3 main.py --dataset="gowalla" --topks="[20,5,10,50]" --simple_model "gf-cf" --gpu_id 0
