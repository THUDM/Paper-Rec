import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tensorboardX import SummaryWriter
from scipy import sparse
import models
import data
import metric
import wandb

parser = argparse.ArgumentParser(description='PyTorch Variational Autoencoders for Collaborative Filtering')
parser.add_argument('--data', type=str, default='ml-20m',
                    help='Movielens-20m dataset location')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate')
parser.add_argument('--wd', type=float, default=0.00,
                    help='weight decay coefficient')
parser.add_argument('--batch_size', type=int, default=500,
                    help='batch size')
parser.add_argument('--epochs', type=int, default=200,
                    help='upper epoch limit')
parser.add_argument('--total_anneal_steps', type=int, default=200000,
                    help='the total number of gradient updates for annealing')
parser.add_argument('--anneal_cap', type=float, default=0.2,
                    help='largest annealing parameter')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--gpu_id', type=int, default=2,
                    help='random seed')
args = parser.parse_args()

# Set the random seed manually for reproductibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")
torch.cuda.set_device(args.gpu_id)
topk = [20,5,10,50]
###############################################################################
# Load data
###############################################################################
t_start = time.time()
print("Staring loading data...")
if args.data == 'ml-20m':
    loader = data.DataLoader(args.data)
    train_data = loader.load_data('train')
    vad_data_tr, vad_data_te = loader.load_data('validation')
    test_data_tr, test_data_te = loader.load_data('test')
    n_items = loader.load_n_items()
else:
    loader = data.BenchmarkDataLoader(args.data)
    train_data = loader.load_data('train')
    vad_data_tr, vad_data_te = loader.load_data('valid')
    test_data_tr, test_data_te = loader.load_data('test')
    n_items = loader.load_n_items()




print("train shape:{}, vad_tr:{}, test_te:{}".format(train_data.shape, vad_data_tr.shape, test_data_te.shape))

N = train_data.shape[0]
idxlist = list(range(N))

print("Done, used {}s.".format(time.time()-t_start))
###############################################################################
# Build the model
###############################################################################

p_dims = [200, 600, n_items]
model = models.MultiVAE(p_dims).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=args.wd)
criterion = models.loss_function

###############################################################################
# Training code
###############################################################################

# TensorboardX Writer

writer = SummaryWriter()

def sparse2torch_sparse(data):
    """
    Convert scipy sparse matrix to torch sparse tensor with L2 Normalization
    This is much faster than naive use of torch.FloatTensor(data.toarray())
    https://discuss.pytorch.org/t/sparse-tensor-use-cases/22047/2
    """
    samples = data.shape[0]
    features = data.shape[1]
    coo_data = data.tocoo()
    indices = torch.LongTensor([coo_data.row, coo_data.col])
    row_norms_inv = 1 / np.sqrt(data.sum(1))
    row2val = {i : row_norms_inv[i].item() for i in range(samples)}
    values = np.array([row2val[r] for r in coo_data.row])
    t = torch.sparse.FloatTensor(indices, torch.from_numpy(values).float(), [samples, features])
    return t

def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray())


def train():
    # Turn on training mode
    model.train()
    train_loss = 0.0
    start_time = time.time()
    global update_count

    np.random.shuffle(idxlist)
    
    for batch_idx, start_idx in enumerate(range(0, N, args.batch_size)):
        end_idx = min(start_idx + args.batch_size, N)
        data = train_data[idxlist[start_idx:end_idx]]

        #print("train data size:{}".format(data.shape))
        data = naive_sparse2tensor(data).to(device)
        #print("after train data size:{}".format(data.shape))

        if args.total_anneal_steps > 0:
            anneal = min(args.anneal_cap, 
                            1. * update_count / args.total_anneal_steps)
        else:
            anneal = args.anneal_cap

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        
        loss = criterion(recon_batch, data, mu, logvar, anneal)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        update_count += 1

        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:4d}/{:4d} batches | ms/batch {:4.2f} | '
                    'loss {:4.2f}'.format(
                        epoch, batch_idx, len(range(0, N, args.batch_size)),
                        elapsed * 1000 / args.log_interval,
                        train_loss / args.log_interval))
            
            # Log loss to tensorboard
            n_iter = (epoch - 1) * len(range(0, N, args.batch_size)) + batch_idx
            writer.add_scalars('data/loss', {'train': train_loss / args.log_interval}, n_iter)

            start_time = time.time()
            train_loss = 0.0


def evaluate(data_tr, data_te):
    # Turn on evaluation mode
    model.eval()
    total_loss = 0.0
    global update_count
    e_idxlist = list(range(data_tr.shape[0]))
    e_N = data_tr.shape[0]
    recall = {}
    ndcg = {}
    for each_k in topk:
        recall[each_k] = []
        ndcg[each_k] = []
    # n100_list = []
    # r20_list = []
    # r50_list = []
    
    with torch.no_grad():
        for start_idx in range(0, e_N, args.batch_size):
            end_idx = min(start_idx + args.batch_size, N)
            data = data_tr[e_idxlist[start_idx:end_idx]]
            heldout_data = data_te[e_idxlist[start_idx:end_idx]]
            #heldout_data = []

            data_tensor = naive_sparse2tensor(data).to(device)

            if args.total_anneal_steps > 0:
                anneal = min(args.anneal_cap, 
                               1. * update_count / args.total_anneal_steps)
            else:
                anneal = args.anneal_cap

            recon_batch, mu, logvar = model(data_tensor)

            loss = criterion(recon_batch, data_tensor, mu, logvar, anneal)
            total_loss += loss.item()

            # Exclude examples from training set
            recon_batch = recon_batch.cpu().numpy()
            #recon_batch[data.nonzero()] = -np.inf

            for each_k in topk:
                ndcg[each_k].append(metric.NDCG_binary_at_k_batch(recon_batch, heldout_data, each_k))
                recall[each_k].append(metric.Recall_at_k_batch(recon_batch, heldout_data, each_k))
            # n100 = metric.NDCG_binary_at_k_batch(recon_batch, heldout_data, 100)
            # r20 = metric.Recall_at_k_batch(recon_batch, heldout_data, 20)
            # r50 = metric.Recall_at_k_batch(recon_batch, heldout_data, 50)

            # n100_list.append(n100)
            # r20_list.append(r20)
            # r50_list.append(r50)
 
    total_loss /= len(range(0, e_N, args.batch_size))
    for each_k in topk:
        ndcg[each_k] = np.mean(np.concatenate(ndcg[each_k]))
        recall[each_k] = np.mean(np.concatenate(recall[each_k]))

    return total_loss, ndcg, recall


best_n100 = -np.inf
update_count = 0
print("Prepare to train, before this we have spent {}s.".format(time.time()-t_start))
# At any point you can hit Ctrl + C to break out of training early.
try:
    tag = []
    note = ''
    tag.append('MultVAE')
    note += 'MultVAE-'
    tag.append(args.data)
    note += args.data
    note += '; only last log is for test set!!!'
    wandb.login(key="ed022f13b9f1b9e155450b10e06c563b40452b07")
    wandb.init(project="CogGCN-baselines", entity="keg-aminer-rec", notes=note, tags=tag)
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train()
        val_loss, ndcg, recall = evaluate(test_data_tr, test_data_te)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:4.2f}s | valid loss {:4.2f} | '
                'n20 {:5.3f} | r20 {:5.3f} | '.format(
                    epoch, time.time() - epoch_start_time, val_loss,
                    ndcg[topk[0]], recall[topk[0]]))
        print('-' * 89)

        n_iter = epoch * len(range(0, N, args.batch_size))
        wandb_log = {}
        wandb_log["Time"] = time.time() - epoch_start_time
        wandb_log["Epoch"] = epoch
        wandb_log["TrainLoss"] = val_loss
        for each_k in topk:
            wandb_log["recall@"+str(each_k)] = recall[each_k]
            wandb_log["ndcg@"+str(each_k)] = ndcg[each_k]
        # writer.add_scalars('data/loss', {'valid': val_loss}, n_iter)
        # writer.add_scalar('data/n100', n100, n_iter)
        # writer.add_scalar('data/r20', r20, n_iter)
        # writer.add_scalar('data/r50', r50, n_iter)
        wandb.log(wandb_log)

        # Save the model if the n100 is the best we've seen so far.
        if recall[topk[0]] > best_n100:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_n100 = recall[topk[0]]

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss, ndcg, recall = evaluate(test_data_tr, test_data_te)
print('=' * 89)
print('| End of training | test loss {:4.2f} | n20 {:4.2f} | r20 {:4.2f} | '.format(test_loss, ndcg[topk[0]], recall[topk[0]]))
print('=' * 89)