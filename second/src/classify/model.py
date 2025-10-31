#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Name   : model.py
Author      : wzw
Date Created: 2025/10/31
Description : 以ResNet为基础的模型，加入自定义卷积层
"""

import torchvision
from torchvision import datasets, models
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms, InterpolationMode

import torch.backends.cudnn as cudnn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNetSimCLR(nn.Module):
    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {
            "resnet18": models.resnet18(weights=None),
            "resnet34": models.resnet34(weights=None),   # backbone baru
            "resnet50": models.resnet50(weights=None),
        }
        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features

 # Projection head baru: Linear -> BN -> ReLU -> Linear(out_dim)
        self.backbone.fc = nn.Sequential(
        nn.Linear(dim_mlp, dim_mlp*2),
        nn.BatchNorm1d(dim_mlp*2),
        nn.ReLU(inplace=True),
        nn.Linear(dim_mlp*2, args.out_dim)
    )


    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError("Backbone invalid. Gunakan resnet18, resnet34, atau resnet50.")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)


class SimCLR:
    def __init__(self, model, optimizer, scheduler, args):
        self.model = model.to(args.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args

    def nt_xent_loss(self, z1, z2, temperature=0.5):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        N = z1.size(0)
        z = torch.cat([z1, z2], dim=0)  # [2N, D]
        sim = torch.matmul(z, z.t())    # [2N, 2N]
        mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        sim = sim[~mask].view(2 * N, 2 * N - 1)
        pos = torch.sum(z1 * z2, dim=-1)
        pos = torch.cat([pos, pos], dim=0)
        sim = sim / temperature
        denom = torch.logsumexp(sim, dim=1)
        loss = -pos / temperature + denom
        return loss.mean()

    def train(self, train_loader):
        self.model.train()
        loss_history = []
        for epoch in range(1, self.args.epochs + 1):
            running = 0.0
            for step, (views, _) in enumerate(train_loader, start=1):
                x1, x2 = views
                x1, x2 = x1.to(self.args.device), x2.to(self.args.device)
                z1 = self.model(x1)
                z2 = self.model(x2)
                loss = self.nt_xent_loss(z1, z2, temperature=self.args.temperature)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running += loss.item()
                if step % self.args.log_every_n_steps == 0:
                    print(f"Epoch {epoch}/{self.args.epochs} Step {step}/{len(train_loader)} Loss {running/self.args.log_every_n_steps:.4f}")
                    running = 0.0
            self.scheduler.step()
            loss_history.append(loss.item())
            print(f"[Epoch {epoch}] Loss: {loss.item():.4f}")
        return loss_history

import torch
from types import SimpleNamespace
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms

# ======================
# Setup Argumen
# ======================
args = SimpleNamespace()
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Ganti path dataset Tiny-ImageNet kalau berbeda
# Struktur: tiny-imagenet-200/{train, val}
args.data = '/kaggle/input/tiny-imagenet'

cudnn.deterministic = True
cudnn.benchmark = True

args.dataset_name = 'tinyimagenet'
args.n_views = 2
args.batch_size = 512
args.out_dim = 256
args.lr = 0.0003
args.weight_decay = 1e-4
args.arch = 'resnet50'   # bisa diganti 'resnet18'
args.workers = 2
args.gpu_index = 0
args.log_dir = './logs/simclr'
args.fp16_precision = True
args.epochs = 10
args.temperature = 0.5
args.seed = 1
args.log_every_n_steps = 50

# ======================
# Dataset
# ======================
dataset = ContrastiveLearningDataset(args.data)

# Train dataset (self-supervised pretraining)
train_dataset = dataset.get_dataset(args.dataset_name, args.n_views)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=min(args.batch_size, 512),   # 512 disarankan
    shuffle=True,
    num_workers=max(2, args.workers, 4),    # 4–8 biasanya oke di P100
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
    drop_last=True
)

# Validation dataset (evaluasi setelah training)
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Path val Tiny-ImageNet
val_dataset = datasets.ImageFolder(root=f"{args.data}/tiny-imagenet-200/val", transform=val_transform)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.workers,
    pin_memory=True
)

# ======================
# Model + Optimizer + Scheduler
# ======================
model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)
optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

# Cosine LR scheduler
total_steps = args.epochs * len(train_loader)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)


# ======================
# Training SimCLR
# ======================
simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)

# Pretraining di train_loader
loss_history = simclr.train(train_loader)

# ======================
# (Optional) Evaluasi pakai val_loader
# ======================
# Contoh evaluasi: ambil embedding dan hitung loss / akurasi kNN
# val_loss = simclr.evaluate(val_loader)
