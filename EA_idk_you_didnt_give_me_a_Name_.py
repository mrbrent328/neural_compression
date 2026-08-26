#!/usr/bin/env python3

"""
R5 — Parameter matched local interaction ablation

Question:
Does interaction improve representation quality beyond simply adding parameters?

Compare:

A:
local representations + shared interaction

B:
same parameter budget, no interaction
"""

from pathlib import Path
import json
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

# Import your existing utilities
from R4_cifar100_no_interaction_ablation import (
    load_cifar100,
    make_subset,
    evaluate,
    train_one,
    save_reconstruction_examples,
    save_global_state_plot,
    save_token_position_plot,
    local_geometry,
    covariance_geometry,
    CIFARParquetDataset,
)


ROOT = Path(__file__).resolve().parent

DATA_ROOT = Path(
    "/home/xetal/Desktop/latent_model_test/cifar100/cifar100"
)

OUT = ROOT / "R5_parameter_matched_ablation"


SEED = 12345

SIZES = [
    500,
    1000,
    5000,
    10000,
    25000,
    40000,
]


PATCH_SIZE = 8
IMAGE_SIZE = 32
CHANNELS = 3

TOKENS = 16
PATCH_DIM = 192


BATCH = 128
EPOCHS = 100

LOCAL_DIM = 32


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# -------------------------------------------------
# Parameter matched models
# -------------------------------------------------


class InteractionModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = nn.Linear(
            PATCH_DIM,
            LOCAL_DIM
        )

        self.position = nn.Parameter(
            torch.zeros(
                1,
                TOKENS,
                LOCAL_DIM
            )
        )

        self.attn = nn.MultiheadAttention(
            LOCAL_DIM,
            4,
            batch_first=True
        )

        self.decoder = nn.Linear(
            LOCAL_DIM,
            PATCH_DIM
        )


    def forward(self,x):

        # patch extraction assumed from your existing code
        patches = patchify(x)

        z = self.encoder(patches)

        z = z + self.position

        y,_ = self.attn(
            z,z,z,
            need_weights=False
        )

        z = z+y

        out=self.decoder(z)

        return unpatchify(out),z,z.mean(1)



class NoInteractionMatched(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(
                PATCH_DIM,
                128
            ),
            nn.GELU(),

            nn.Linear(
                128,
                LOCAL_DIM
            ),
        )


        self.position = nn.Parameter(
            torch.zeros(
                1,
                TOKENS,
                LOCAL_DIM
            )
        )


        # extra capacity replaces attention

        self.local_extra = nn.Sequential(
            nn.Linear(
                LOCAL_DIM,
                128
            ),
            nn.GELU(),

            nn.Linear(
                128,
                LOCAL_DIM
            )
        )


        self.decoder = nn.Linear(
            LOCAL_DIM,
            PATCH_DIM
        )


    def forward(self,x):

        patches=patchify(x)

        z=self.encoder(patches)

        z=z+self.position

        z=self.local_extra(z)

        out=self.decoder(z)

        return unpatchify(out),z,z.mean(1)



# -------------------------------------------------
# Main experiment
# -------------------------------------------------

def main():

    seed_everything(SEED)

    OUT.mkdir(
        exist_ok=True
    )

    device=torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    train,test=load_cifar100(
        DATA_ROOT
    )

    pool=np.arange(
        len(train)
    )


    for name,Model in [
        (
            "interaction",
            InteractionModel
        ),
        (
            "no_interaction_matched",
            NoInteractionMatched
        )
    ]:

        out=OUT/name

        out.mkdir(
            exist_ok=True
        )


        for size in SIZES:

            subset=make_subset(
                train,
                pool,
                size,
                SEED
            )


            loader=DataLoader(
                subset,
                batch_size=BATCH,
                shuffle=True
            )


            model=Model().to(device)


            params=sum(
                p.numel()
                for p in model.parameters()
                if p.requires_grad
            )


            print(
                name,
                size,
                params
            )


            start=time.time()


            history=train_one(
                model,
                loader,
                device,
                EPOCHS,
                1e-3
            )


            result=evaluate(
                model,
                test,
                device,
                BATCH
            )


            summary={

                "experiment":
                    "R5_parameter_matched",

                "variant":
                    name,

                "training_size":
                    size,

                "parameters":
                    params,

                "test_mse":
                    float(
                        result["errors"].mean()
                    ),

                "pooled_PR":
                    covariance_geometry(
                        result["global_states"]
                    )["participation_ratio"],

                "local_PR":
                    local_geometry(
                        result["local_states"]
                    )["participation_ratio"],

                "time":
                    time.time()-start,

            }


            with open(
                out/f"summary_{size}.json",
                "w"
            ) as f:

                json.dump(
                    summary,
                    f,
                    indent=2
                )


if __name__=="__main__":
    main()
