import torch
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from utils.TimeSeriesDataSet import TimeSeriesDataset


def get_loader(local_rank,train_batch_size,eval_batch_size,data_source_dir):
    # 当进程不是主进程（local_rank不为0）且在分布式环境中（local_rank不为-1）时，等待其他进程
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    # 只有当进程是主进程时才执行barrier。这通常意味着等待所有其他进程到达这一点，然后由主进程执行一些操作，比如保存模型或输出日志
    if local_rank == 0:
        torch.distributed.barrier()

    file_extension = '.csv'
    dataset = TimeSeriesDataset(data_source_dir, file_extension)
    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.8)
    test_size = dataset_size - train_size
    trainset, testset = random_split(dataset, [train_size, test_size])
    trainset.AUG = True
    train_sampler = RandomSampler(trainset) if local_rank == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=train_batch_size,
                              collate_fn=TimeSeriesDataset.collate_fn,
                              drop_last=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=eval_batch_size,
                             collate_fn=TimeSeriesDataset.collate_fn,
                             drop_last=True) if testset is not None else None

    return train_loader, test_loader
