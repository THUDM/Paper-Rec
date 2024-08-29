# Paper-Rec

# Environment
- Python 3.7
- torch 1.10.0+cu111
- ```pip install -r requirements.txt``` (Note: sparsesvd can be installed from source)


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
cd GF_CF-main/Table3\&4
python3 main.py --dataset="aminer" --topks="[20,5,10,50]" --simple_model "gf-cf" --gpu_id 0
```

* LightGCN & NGCF
```shell
cd MixGCF
python main.py --dataset aminer --context_hops 2 --gpu_id 1 --gnn lightgcn --pool mean
python main.py --dataset aminer --context_hops 2 --gpu_id 1 --gnn ngcf --pool mean
```


## References
🌟 If you find our work helpful, please leave us a star and cite our paper.
```
@inproceedings{zhang2024oag,
  title={OAG-bench: a human-curated benchmark for academic graph mining},
  author={Fanjin Zhang and Shijie Shi and Yifan Zhu and Bo Chen and Yukuo Cen and Jifan Yu and Yelin Chen and Lulu Wang and Qingfei Zhao and Yuqing Cheng and Tianyi Han and Yuwei An and Dan Zhang and Weng Lam Tam and Kun Cao and Yunhe Pang and Xinyu Guan and Huihui Yuan and Jian Song and Xiaoyan Li and Yuxiao Dong and Jie Tang},
  booktitle={Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={6214--6225},
  year={2024}
}
```
