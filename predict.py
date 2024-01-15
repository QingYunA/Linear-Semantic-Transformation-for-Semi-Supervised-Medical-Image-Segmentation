import os
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torchio as tio
from torchio.transforms import (
    ZNormalization,
)
from tqdm import tqdm
from utils.metric import metric
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from process_input import process_x, process_gt
import numpy as np
from logger import create_logger
from utils import yaml_read
from utils.conf_base import Default_Conf
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    MofNCompleteColumn,
    TimeRemainingColumn,
)
import hydra
from rich.logging import RichHandler
import logging
from accelerate import Accelerator
import shutil

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # ! solve warning


def get_logger(config):
    file_handler = logging.FileHandler(os.path.join(config.hydra_path, f"{config.job_name}.log"))
    rich_handler = RichHandler()

    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    log.addHandler(rich_handler)
    log.addHandler(file_handler)
    log.propagate = False
    log.info("Successfully create rich logger")

    return log


def predict(model, config, logger):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = config.cudnn_enabled
    torch.backends.cudnn.benchmark = config.cudnn_benchmark
    # init progress
    progress = Progress(
        TextColumn("[bold blue]{task.description}", justify="right"),
        MofNCompleteColumn(),
        BarColumn(bar_width=40),
        "[progress.percentage]{task.percentage:>3.1f}%",
        TimeRemainingColumn(),
    )

    # * load model
    # assert type(conf.ckpt) == str, "You must specify the checkpoint path"
    assert isinstance(config.ckpt, str), "You must specify the checkpoint path"
    logger.info(f"load model from:{os.path.join(config.ckpt, config.latest_checkpoint_file)}")
    ckpt = torch.load(os.path.join(config.ckpt, config.latest_checkpoint_file), map_location=lambda storage, loc: storage)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # * load datasetBs
    from dataloader import Dataset

    dataset = Dataset(config).subjects  # ! notice in predict.py should use Dataset(conf).subjects
    znorm = ZNormalization()

    pre_ls,rec_ls,jaccard_ls, dice_ls,hs95_ls = [], [],[],[],[]

    file_tqdm = progress.add_task("[red]Predicting file", total=len(dataset))

    # *  accelerator prepare
    accelerator = Accelerator()
    model = accelerator.prepare(model)
    # start progess
    progress.start()
    for i, item in enumerate(dataset):
        item = znorm(item)
        grid_sampler = tio.inference.GridSampler(item, patch_size=(config.patch_size), patch_overlap=(4, 4, 36))
        affine = item["source"]["affine"]
        spacing =item.spacing
        # * dist sampler
        # dist_sampler = torch.utils.data.distributed.DistributedSampler(grid_sampler, shuffle=True)

        # assert conf.batch_size == 1, 'batch_size must be 1 for inference'

        patch_loader = torch.utils.data.DataLoader(
            grid_sampler, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=True
        )
        patch_loader = accelerator.prepare(patch_loader)
        if i == 0:
            batch_tqdm = progress.add_task("[blue]file batch", total=len(patch_loader))
        else:
            progress.reset(batch_tqdm, total=len(patch_loader))

        pred_aggregator = tio.inference.GridAggregator(grid_sampler)
        gt_aggregator = tio.inference.GridAggregator(grid_sampler)
        with torch.no_grad():
            for j, batch in enumerate(patch_loader):
                locations = batch[tio.LOCATION]

                x = process_x(config, batch)
                x = x.type(torch.FloatTensor).to(accelerator.device)
                gt = process_gt(config, batch)
                gt = gt.type(torch.FloatTensor).to(accelerator.device)

                if config.network =="IS":
                    _,pred = model(x)
                pred = model(x)

                # mask = torch.sigmoid(pred.clone())
                # mask[mask > 0.5] = 1
                # mask[mask <= 0.5] = 0
                mask = pred.clone()
                mask = mask.argmax(dim=1, keepdim=True)

                pred_aggregator.add_batch(mask, locations)
                gt_aggregator.add_batch(gt, locations)
                progress.update(batch_tqdm, completed=j + 1)
                progress.refresh()
            # reset batchtqdm
            pred_t = pred_aggregator.get_output_tensor()
            gt_t = gt_aggregator.get_output_tensor()

            # * save pred mhd file
            save_mhd(pred_t, affine, i, config)
            # * calculate metrics
            pre,rec,jaccard, dice,hs95 = metric(gt_t, pred_t,spacing)
            pre_ls.append(pre)
            rec_ls.append(rec)
            jaccard_ls.append(jaccard)
            dice_ls.append(dice)
            hs95_ls.append(hs95)
            logger.info(f"File {i+1} metrics: " f"\n presicion: {pre}" f"\n recall: {rec}" f"\njaccard: {jaccard}" f"\ndice: {dice}" f"\n hs95: {hs95}")
        progress.update(file_tqdm, completed=i + 1)
    save_csv(pre_ls,rec_ls,jaccard_ls, dice_ls,hs95_ls,config)
    presion_mean = np.mean(pre_ls)
    rec_mean = np.mean(rec_ls)
    jaccard_mean = np.mean(jaccard_ls)
    dice_mean = np.mean(dice_ls)
    hs95_mean = np.mean(hs95_ls)
    # print('-' * 40)
    logger.info(f"\npresion_mean:{presion_mean}" f"\nrecall_mean:{rec_mean}" f"\njaccard_mean: {jaccard_mean}" f"\ndice_mean: {dice_mean}" f"\n hs95_mean: {hs95_mean}")

def save_csv(pre_ls,rec_ls,jaccard_ls, dice_ls,hs95_ls,config):
    import pandas as pd

    data = {"presion":pre_ls,"recall":rec_ls,"jaccard": jaccard_ls, "dice": dice_ls,"hs95":hs95_ls}
    df = pd.DataFrame(data)
    df.loc[len(df)] = [df.iloc[:, 0].mean(), df.iloc[:, 1].mean(),df.iloc[:,2].mean(),df.iloc[:,3].mean(),df.iloc[:,4].mean()]
    save_path = os.path.join(config.hydra_path, "metrics.csv")
    df.to_csv(save_path, index=False)

def save_mhd(pred, affine, index, config):
    save_base = os.path.join(config.hydra_path, "pred_file")
    os.makedirs(save_base, exist_ok=True)
    pred_data = tio.ScalarImage(tensor=pred, affine=affine)
    pred_data.save(os.path.join(save_base, f"pred-{index:04d}.mhd"))


@hydra.main(config_path="conf", config_name="config")
def main(config):
    config = config["config"]
    config.hydra_path = config.hydra_path.replace("logs", "results")

    if isinstance(config.patch_size, str):
        assert (
            len(config.patch_size.split(",")) <= 3
        ), f'patch size can only be one str or three str but got {len(config.patch_size.split(","))}'
        if len(config.patch_size.split(",")) == 3:
            config.patch_size = tuple(map(int, config.patch_size.split(",")))
        else:
            config.patch_size = int(config.patch_size)

    os.makedirs(config.hydra_path, exist_ok=True)
    # * model selection
    if config.network == "res_unet":
        from models.three_d.residual_unet3d import UNet
        model = UNet(in_channels=config.in_classes, n_classes=config.out_classes, base_n_filter=32)
    elif config.network == "unet":
        from models.three_d.unet3d import UNet3D  # * 3d unet
        model = UNet3D(in_channels=config.in_classes,out_channels=config.out_classes,init_features=32)

    elif config.network == 'vnet':
        from models.three_d.vnet3d import VNet
        model = VNet(in_channels=config.in_classes,classes= config.out_classes)

    elif config.network == 'unetr':
        from models.three_d.unetr import UNETR
        model = UNETR(img_shape=config.img_shape, input_dim=config.in_classes, output_dim=config.out_classes,
                      embed_dim=config.embed_dim, patch_size=config.unetr_patch_size, num_heads=config.num_heads, dropout=config.dropout)

    elif config.network == "ernet":
        from models.three_d.ER_net import ER_Net
        model = ER_Net(classes=config.out_classes, channels=config.in_classes)

    elif config.network == "renet":
        from models.three_d.RE_net import RE_Net
        model = RE_Net(classes=config.in_classes, channels=config.out_classes)
    elif config.network == "IS_net":
        from models.three_d.IS import UNet3D
        model = UNet3D(in_channels=config.in_classes,out_channels=config.out_classes,init_features=32)
    elif config.network == "csrnet":
        from models.three_d.csrnet import UNet3D
        model = UNet3D(in_channels=config.in_classes,out_channels=config.out_classes,init_features=32) 
    elif config.network == "IS":
        from models.three_d.IS import UNet3D
        model = UNet3D(in_channels=config.in_classes,out_channels=config.out_classes,init_features=32)

    # * create logger
    logger = get_logger(config)
    info = "\nParameter Settings:\n"
    for k, v in config.items():
        info += f"{k}: {v}\n"
    logger.info(info)

    predict(model, config, logger)
    logger.info(f"tensorboard file saved in:{config.hydra_path}")
    # TODO 取巧，通过删除文件夹的方式消除重复出现的文件夹
    shutil.rmtree(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)


if __name__ == "__main__":
    main()
