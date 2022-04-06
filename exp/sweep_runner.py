#! /usr/bin/env python
# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
This script will allow wandb to run hyperparameter sweeps and log the results from individual folds.
NOTE: IT WILL NOT RUN IN AN IDE DUE THE MULTIPROCESSING
"""
import collections
import sys, os
import multiprocessing

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import torch.nn.functional as F
import git
import numpy as np

from exp.parser import get_parser
from models.cont_models import DiagSheafDiffusion
from models.disc_models import DiscreteDiagSheafDiffusion, DiscreteBundleSheafDiffusion
from data.heterophilic import get_dataset, get_fixed_splits
from torch_geometric.utils import degree
from tqdm import tqdm
import wandb

# wandb stuff
Worker = collections.namedtuple("Worker", ("queue", "process"))
WorkerInitData = collections.namedtuple(
    "WorkerInitData", ("num", "sweep_id", "sweep_run_name", "config", "dataset", "model_cls")
)
WorkerDoneData = collections.namedtuple("WorkerDoneData", ("best_val_acc", "best_test_acc"))


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x)[data.train_mask]
    nll = F.nll_loss(out, data.y[data.train_mask])
    loss = nll
    loss.backward()

    optimizer.step()
    del out


def test(model, data):
    model.eval()
    with torch.no_grad():
        logits, accs, losses, preds = model(data.x), [], [], []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

            loss = F.nll_loss(logits[mask], data.y[mask])

            preds.append(pred.detach().cpu())
            accs.append(acc)
            losses.append(loss.detach().cpu())
        return accs, preds, losses


def run_fold(sweep_q, worker_q):
    worker_data = worker_q.get()
    dataset = worker_data.dataset
    model_cls = worker_data.model_cls
    fold = worker_data.num
    reset_wandb_env()
    run_name = f"{worker_data.sweep_run_name}-{worker_data.num}"
    args = worker_data.config
    run = wandb.init(
        group=worker_data.sweep_id,
        job_type=worker_data.sweep_run_name,
        name=run_name,
        config=args,
    )
    data = dataset[0]
    data = get_fixed_splits(data, args['dataset'], fold)
    data = data.to(args['device'])

    model = model_cls(data.edge_index, args)
    model = model.to(args['device'])

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    epoch = 0
    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []
    best_epoch = 0
    bad_counter = 0

    for epoch in range(args['epochs']):
        train(model, optimizer, data)

        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss] = test(model, data)
        res_dict = {'train_acc': train_acc, 'val_acc': val_acc, 'tmp_test_acc': tmp_test_acc, 'train_loss': train_loss}
        run.log(res_dict, step=epoch)
        if val_loss < best_val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args['early_stopping']:
            break

    run.log({'best_test_acc': test_acc, 'best_val_acc': best_val_acc, 'best_epoch': best_epoch})
    run.finish()
    sweep_q.put(WorkerDoneData(best_val_acc=best_val_acc, best_test_acc=test_acc))

    # return test_acc, best_val_acc


def main(args):
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    if args.model == 'DiagSheafODE':
        model_cls = DiagSheafDiffusion
    elif args.model == 'DiagSheaf':
        model_cls = DiscreteDiagSheafDiffusion
    elif args.model == 'BundleSheaf':
        model_cls = DiscreteBundleSheafDiffusion
    else:
        raise ValueError(f'Unknown model {args.model}')

    dataset = get_dataset(args.dataset)

    # Add extra arguments
    args.sha = sha
    args.graph_size = dataset[0].x.size(0)
    args.input_dim = dataset.num_features
    args.output_dim = dataset.num_classes
    args.device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    assert args.normalised or args.deg_normalised

    deg = degree(dataset[0].edge_index[0], num_nodes=dataset[0].x.size(0))
    print("Isolated nodes:", (dataset[0].x.size(0) - torch.count_nonzero(deg)).item())
    print(args)

    results = []

    # wandb stuff
    sweep_run = wandb.init(project="sheaf", config=args, entity=args.entity)
    # sweep_run = wandb.init(config=args)
    sweep_id = sweep_run.sweep_id or "unknown"
    sweep_url = sweep_run.get_sweep_url()
    project_url = sweep_run.get_project_url()
    sweep_group_url = "{}/groups/{}".format(project_url, sweep_id)
    sweep_run.notes = sweep_group_url
    # sweep_run.save()
    sweep_run_name = sweep_run.name or sweep_run.id or "unknown"

    # group_id = wandb.util.generate_id()
    # Spin up workers before calling wandb.init()
    # Workers will be blocked on a queue waiting to start
    sweep_q = multiprocessing.Queue()
    workers = []
    for num in range(args.folds):
        q = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=run_fold, kwargs=dict(sweep_q=sweep_q, worker_q=q)
        )
        p.start()
        workers.append(Worker(queue=q, process=p))

    for fold in tqdm(range(args.folds)):
        worker = workers[fold]
        # start worker
        worker.queue.put(
            WorkerInitData(
                sweep_id=sweep_id,
                num=fold,
                sweep_run_name=sweep_run_name,
                config=dict(sweep_run.config),
                dataset=dataset,
                model_cls=model_cls
            )
        )
        # test_acc, best_val_acc = run_fold(args, dataset, model_cls, fold, worker_data)
        # get metric from worker
        result = sweep_q.get()
        # wait for worker to finish
        worker.process.join()
        # log metric to sweep_run
        results.append([result.best_test_acc, result.best_val_acc])

    test_acc_mean, val_acc_mean = np.mean(results, axis=0) * 100
    test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100
    print(f'{args.model} on {args.dataset} | SHA: {sha}')
    print(f'Test acc: {test_acc_mean:.4f} +/- {test_acc_std:.4f} | Val acc: {val_acc_mean:.4f}')
    wandb_results = {'test_acc': test_acc_mean, 'val_acc': val_acc_mean, 'test_acc_std': test_acc_std}
    sweep_run.log(wandb_results)
    wandb.finish()


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    main(args)
