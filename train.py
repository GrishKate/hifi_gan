import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
import itertools

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture, then print to console
    generator = instantiate(config.generator).to(device)
    msd = instantiate(config.msd).to(device)
    mpd = instantiate(config.mpd).to(device)

    # get function handles of loss and metrics
    gen_loss = instantiate(config.gen_loss).to(device)
    disc_loss = instantiate(config.disc_loss).to(device)
    metrics = instantiate(config.metrics)

    # build optimizer, learning rate scheduler
    trainable_params = filter(lambda p: p.requires_grad, generator.parameters())
    gen_optimizer = instantiate(config.gen_optimizer, params=trainable_params, betas=(0.8, 0.99))
    gen_lr_scheduler = instantiate(config.gen_lr_scheduler, optimizer=gen_optimizer)
    trainable_params = filter(lambda p: p.requires_grad, (itertools.chain(msd.parameters(), mpd.parameters())))
    disc_optimizer = instantiate(config.disc_optimizer, params=trainable_params, betas=(0.8, 0.99))
    disc_lr_scheduler = instantiate(config.disc_lr_scheduler, optimizer=disc_optimizer)

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")

    trainer = Trainer(
        generator=generator,
        msd=msd,
        mpd=mpd,
        gen_loss=gen_loss,
        disc_loss=disc_loss,
        metrics=metrics,
        gen_optimizer=gen_optimizer,
        disc_optimizer=disc_optimizer,
        gen_lr_scheduler=gen_lr_scheduler,
        disc_lr_scheduler=disc_lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()


if __name__ == "__main__":
    main()
