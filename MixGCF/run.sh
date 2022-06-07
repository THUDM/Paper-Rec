# lighgcn gowalla
nohup python main.py --dataset gowalla --context_hops 1 --gpu_id 0 --gnn lightgcn --pool mean > ./training_log/train_0120_gowalla_1_lightgcn.log 2>&1 &
nohup python main.py --dataset gowalla --context_hops 2 --gpu_id 0 --gnn lightgcn --pool mean > ./training_log/train_0120_gowalla_2_lightgcn.log 2>&1 &
nohup python main.py --dataset gowalla --context_hops 3 --gpu_id 6 --gnn lightgcn --pool mean > ./training_log/train_0120_gowalla_3_lightgcn.log 2>&1 &
nohup python main.py --dataset gowalla --context_hops 4 --gpu_id 6 --gnn lightgcn --pool mean > ./training_log/train_0120_gowalla_4_lightgcn.log 2>&1 &

## yelp
nohup python main.py --dataset yelp2018 --context_hops 1 --gpu_id 4 --gnn lightgcn --pool mean > ./training_log/train_0120_yelp2018_1_lightgcn.log 2>&1 &
nohup python main.py --dataset yelp2018 --context_hops 2 --gpu_id 5 --gnn lightgcn --pool mean > ./training_log/train_0120_yelp2018_2_lightgcn.log 2>&1 &
nohup python main.py --dataset yelp2018 --context_hops 3 --gpu_id 6 --gnn lightgcn --pool mean > ./training_log/train_0120_yelp2018_3_lightgcn.log 2>&1 &
nohup python main.py --dataset yelp2018 --context_hops 4 --gpu_id 7 --gnn lightgcn --pool mean > ./training_log/train_0120_yelp2018_4_lightgcn.log 2>&1 &
