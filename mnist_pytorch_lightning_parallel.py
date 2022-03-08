"""
What does this script do
========================

Show basic PyTorch Lightning features.

Use pytorch-lightning for data parallel training using
torch.nn.parallel.DistributedDataParallel via Trainer(strategy="ddp") on one
multi-GPU node.

For technical reasons, this cannot be run in Jupyter. See
https://pytorch-lightning.readthedocs.io/en/latest/accelerators/gpu.html#distributed-data-parallel
for why.

That's why we need to run in a normal interactive session on a compute node.


On the login node (example: HZDR herema cluster)
================================================

Do this once in your $HOME.

    $ ml python/3.9.6
    $ which python
    /trinity/shared/pkg/devel/python/3.9.6/bin/python

    $ python3 -m venv --system-site-packages lightning
    $ . ./lightning/bin/activate
    (lightning) $ pip install pytorch-lightning

Start interactive session, request 4 GPUs. This will change your shell
prompt if the session started.

    $ srun --time=01:00:00 -p gpu --nodes=1 --ntasks-per-node=4 --gres=gpu:4 --pty bash -l -i

On the compute node
===================

Check if we have GPUs.

    $ nvidia-smi

You may need to load that again.

    $ ml python/3.9.6
    $ which python
    /trinity/shared/pkg/devel/python/3.9.6/bin/python
    $ . ./lightning/bin/activate

Run with 1 GPU and time it. PL_TORCH_DISTRIBUTED_BACKEND=nccl is default.

    (lightning) $ time CUDA_VISIBLE_DEVICES=0 python this_script.py

Or set Trainer(gpus=[0],...) and

    (lightning) $ time python this_script.py

Note that gpus=0 means CPU!

Run with 4 GPUs

    (lightning) $ time CUDA_VISIBLE_DEVICES=0,1,2,3 python this_script.py

Or set Trainer(gpus=[0,1,2,3],...) and

    (lightning) $ time python this_script.py

Lightning features
==================

* Abstract away lots of boilerplate code (train loop, etc)
* Automatic checkpointing
  * checkpoints include global state, not just state_dict
    https://pytorch-lightning.readthedocs.io/en/latest/common/checkpointing.html
* model.eval() and torch.no_grad() are called automatically where needed
* automatic Tensor.to(<device>)
* easy logging (default in TensorBoard format)
* easy(er) multi-GPU training

Checkpoints
-----------

docs: https://pytorch-lightning.readthedocs.io/en/latest/common/checkpointing.html

    # ./checkpoints or logs/version_XY/checkpoints/ (when
    # TensorBoardLogger("./logs"))
    $ ls -1rt checkpoints
    'epoch=4-step=589.ckpt'

Logging (TensorBoard default)
-----------------------------

    # http://localhost:6006/
    $ tensorboard --logdir=logs

When working on a cluster, either start tensorboard there and forward the port
to your local machine or mount the remote log dir locally (e.g. using sshfs) and
run a local tensorboard using that log dir.
"""

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import pytorch_lightning as pl
##from pytorch_lightning.loggers import TensorBoardLogger

# lightning > v1.5
##from pytorch_lightning.strategies import DDPStrategy

# lightning v1.5
from pytorch_lightning.plugins import DDPPlugin
DDPStrategy = DDPPlugin


class Model(pl.LightningModule):
    """Each of the methods below override default behavior in LightningModule.
    This is how Lightning is supposed to be used.
    """

    def __init__(self, *, lr, loss_func, n_classes=None):
        super().__init__()
        self.lr = lr
        self.loss_func = loss_func

        # Ridicolously large model, just to give the GPU something to chew on
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(9216,  10000),
            nn.ReLU(),
            nn.Linear(10000, 10000),
            nn.ReLU(),
            nn.Linear(10000, 10000),
            nn.ReLU(),
            nn.Linear(10000, 10000),
            nn.ReLU(),
            nn.Linear(10000, n_classes),
        )

        ##self.model = nn.Sequential(
        ##    nn.Conv2d(1, 32, 3, 1),
        ##    nn.ReLU(),
        ##    nn.Conv2d(32, 64, 3, 1),
        ##    nn.ReLU(),
        ##    nn.MaxPool2d(2),
        ##    nn.Flatten(),
        ##    nn.Linear(9216,  128),
        ##    nn.ReLU(),
        ##    nn.Linear(128,  n_classes),
        ##)

        # Tiny model for CPU-only tests
        ##self.model = nn.Sequential(
        ##    nn.Flatten(),
        ##    nn.Linear(28 * 28, 50),
        ##    nn.ReLU(),
        ##    nn.Linear(50, n_classes),
        ##)

    def forward(self, x):
        # Only when self.model = Linear(...)
        ##x = x.view(x.shape[0], -1)
        return F.log_softmax(self.model(x), dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # self() calls self.forward()
        loss = self.loss_func(self(x), y)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.log(
            "val_loss", self.loss_func(self(x), y), on_step=False, on_epoch=True,
        )

    def configure_optimizers(self):
        return T.optim.Adam(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    # Seed each model on each GPU with the same random seed
    pl.seed_everything(123)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    data_path = "./pytorch_data"
    train_dataset = datasets.MNIST(
        data_path, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(data_path, train=False, transform=transform)

    common_loader_kwds = dict(pin_memory=True, num_workers=4)
    train_kwds = dict(batch_size=512, **common_loader_kwds)
    train_loader = T.utils.data.DataLoader(train_dataset, **train_kwds)

    test_kwds = train_kwds
    test_loader = T.utils.data.DataLoader(test_dataset, **test_kwds)

    model = Model(
        lr=0.01,
        loss_func=F.nll_loss,
        n_classes=10,
    )
    print(model)

    trainer = pl.Trainer(
        max_epochs=2,
        ##gpus=[0,1],
        gpus=-1 if T.cuda.is_available() else 0,
        ##strategy="ddp",
        strategy=DDPStrategy(find_unused_parameters=False),
        ##logger=TensorBoardLogger("./logs", name="best_model_ever"),
        # log every nth batch, default 50
        ##log_every_n_steps=10,
        logger=None,
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=test_loader,
    )
