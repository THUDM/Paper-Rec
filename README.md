# Paper-Rec

# Environment
- Python 3.7
- ```pip install -r requirements.txt```


## How to Run

* VAE:
We apply the VAE model implemented by [younggyoseo's version](https://github.com/younggyoseo/vae-cf-pytorch).

```shell
cd ./vae-cf-pytorch
python main.py --cuda --gpu_id 0 --data aminer --batch_size 2048
```

* GF-CF:

We apply the GF-CF model implemented by [yshenaw's version](https://github.com/yshenaw/GF_CF).

```shell
cd ./GF_CF-main/Table3&4/
python3 main.py --dataset="aminer" --topks="[20,5,10,50]" --simple_model "gf-cf" --gpu_id 0
```

* LightGCN & NGCF
```shell
cd MixGCF
python main.py --dataset aminer --context_hops 2 --gpu_id 1 --gnn lightgcn --pool mean
python main.py --dataset aminer --context_hops 2 --gpu_id 1 --gnn ngcf --pool mean
```
