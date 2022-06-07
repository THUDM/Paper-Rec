# Paper-Rec


## codes
Before running the code, please make sure you have installed all required packages in `requirements.txt` in each model folder.

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
