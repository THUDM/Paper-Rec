nohup python main.py --cuda --gpu_id 3 --data aminer --batch_size 2048 > ./log/VAE_aminer.log 2>&1 &
nohup python main.py --cuda --gpu_id 3 --data amazon --batch_size 2048 > ./log/VAE_amazon.log 2>&1 &
nohup python main.py --cuda --gpu_id 3 --data yelp2018 --batch_size 2048 > ./log/VAE_yelp.log 2>&1 &
nohup python main.py --cuda --gpu_id 3 --data gowalla --batch_size 2048 > ./log/VAE_gowalla.log 2>&1 &
nohup python main.py --cuda --gpu_id 3 --data ml-1m --batch_size 2048 > ./log/VAE_ml1m.log 2>&1 &