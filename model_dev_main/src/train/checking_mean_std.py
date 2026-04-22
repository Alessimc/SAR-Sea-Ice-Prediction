from torch.utils.data import DataLoader
import torch
import argparse
from model_dev_main.src.dataloader.DriftWindDataset import DriftWindDataset

_ap = argparse.ArgumentParser(description="Inspect dataset mean/std (debug utility).")
_ap.add_argument("--index", required=True, help="Path to index JSONL")
_ap.add_argument("--norm_yaml", default=None, help="Path to normalisation YAML")
_args = _ap.parse_args()

train_ds = DriftWindDataset(
    index_jsonl=_args.index,
    include_wspd=False,
    return_meta=False,
    norm_yaml_path=_args.norm_yaml,
    normalize_y=True,
)

train_loader = DataLoader(
    train_ds,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True,
)


@torch.no_grad()
def global_mean_std(loader, key="x", n_batches=100):
    s = None
    ss = None
    n = 0

    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        x = batch[key].float()  # (B,C,H,W)
        B,C,H,W = x.shape
        x = x.view(B, C, -1)

        bs = x.sum(dim=(0,2))
        bss = (x*x).sum(dim=(0,2))

        s = bs if s is None else s + bs
        ss = bss if ss is None else ss + bss
        n += B*H*W

    mean = s / n
    var = (ss / n) - mean**2
    std = torch.sqrt(torch.clamp(var, min=1e-12))
    return mean, std

m, s = global_mean_std(train_loader, key="x", n_batches=100)
print("Global mean (100 batches):", m)
print("Global std  (100 batches):", s)
