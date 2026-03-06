# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Théo Gieruc and Marius Kästingschäfer
# ==============================================================================

import argparse
import os
import warnings

import numpy as np
import torch
from mmengine.config import Config
from tqdm import tqdm

warnings.filterwarnings("ignore")
import time

import cv2
import lpips
import matplotlib

from builder import data_builder, model_builder
from triplane_decoder.decoder import TriplaneDecoder
from triplane_decoder.losses import compute_tv_loss
from triplane_decoder.lr_scheduling import get_cosine_schedule_with_warmup
from triplane_decoder.pif import PIF
from triplane_decoder.rendering import render_rays

matplotlib.use("Agg")
import mlflow
from matplotlib import pyplot as plt


def main(local_rank, args):
    # global settings

    experiment_name = args.log_dir or "6img-to-3d"
    mlflow.set_experiment(experiment_name)
    active_run = mlflow.start_run(run_name=args.log_dir or None)
    print(f"MLFLOW_RUN_ID={active_run.info.run_id}", flush=True)
    logdir = (
        f"runs/{time.strftime('%b%d_%H-%M-%S', time.localtime())}_{args.log_dir}"
        if args.log_dir
        else f"runs/{time.strftime('%b%d_%H-%M-%S', time.localtime())}"
    )
    save_dir = os.path.join(logdir, "models")
    mlflow.log_params(
        {
            "config": args.py_config,
            "log_dir": args.log_dir,
            "manifest_train": args.manifest_train or "",
        }
    )

    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)

    os.makedirs(logdir, exist_ok=True)
    Config.dump(cfg, os.path.join(logdir, "config.py"))

    triplane_encoder = model_builder.build(cfg.model)
    triplane_decoder = TriplaneDecoder(cfg)

    pif = PIF().cuda() if cfg.pif else None

    triplane_decoder = triplane_decoder.cuda()
    triplane_encoder = triplane_encoder.cuda()

    print("done building models")

    # resume and load
    if args.ckpt_path:
        assert os.path.isfile(args.ckpt_path)
        cfg.resume_from = args.ckpt_path
        print("ckpt path:", cfg.resume_from)
        map_location = "cpu"
        ckpt = torch.load(cfg.resume_from, map_location=map_location)
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        if "model" in ckpt:
            ckpt = ckpt["model"]
        tpv_keys = triplane_encoder.state_dict().keys()
        ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
        # module_key = [key for key in ckpt.keys() if 'module.' in key]
        # if len(module_key) > 0:
        #     ckpt = revise_ckpt(ckpt)
        try:
            print(triplane_encoder.load_state_dict(ckpt, strict=False))
            print("successfully loaded ckpt")
        except Exception as e:
            print(e)

    # torch.cuda.empty_cache()
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location="cpu")
        try:
            triplane_decoder_ckpt = ckpt["ttp"]
            if (
                "module.full_net.params" in triplane_decoder_ckpt
                and "module.decoder_net.params" not in triplane_decoder_ckpt
            ):
                triplane_decoder_ckpt["module.decoder_net.params"] = triplane_decoder_ckpt["module.full_net.params"]
            triplane_decoder_ckpt = {k.replace("module.", ""): v for k, v in triplane_decoder_ckpt.items()}
            print(triplane_decoder.load_state_dict(triplane_decoder_ckpt, strict=False))
        except:
            print("no ttp in ckpt")

        triplane_generator_ckpt = ckpt["tpv"]
        triplane_generator_ckpt = {k.replace("module.", ""): v for k, v in triplane_generator_ckpt.items()}

        print(triplane_encoder.load_state_dict(triplane_generator_ckpt, strict=False))

    if args.manifest_train and args.manifest_val:
        from types import SimpleNamespace

        train_dl_cfg = SimpleNamespace(
            depth=cfg.dataset_params.train_data_loader.get("depth", False),
            phase="train",
            batch_size=cfg.dataset_params.train_data_loader.get("batch_size", 1),
            factor=cfg.dataset_params.train_data_loader.get("factor", 1.0),
            num_workers=cfg.dataset_params.train_data_loader.get("num_workers", 0),
            shuffle=cfg.dataset_params.train_data_loader.get("shuffle", True),
        )
        val_dl_cfg = SimpleNamespace(
            depth=cfg.dataset_params.val_data_loader.get("depth", False),
            phase="val",
            batch_size=cfg.dataset_params.val_data_loader.get("batch_size", 1),
            factor=cfg.dataset_params.val_data_loader.get("factor", 0.25),
            num_workers=cfg.dataset_params.val_data_loader.get("num_workers", 0),
            shuffle=False,
        )
        train_dataset_loader, val_dataset_loader = data_builder.build_from_manifests(
            train_manifest=args.manifest_train,
            val_manifest=args.manifest_val,
            config=cfg,
            train_dataset_config=train_dl_cfg,
            val_dataset_config=val_dl_cfg,
        )
    else:
        train_dataset_loader, val_dataset_loader = data_builder.build(cfg)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    optimizer = torch.optim.AdamW(
        list(triplane_encoder.parameters()) + list(triplane_decoder.parameters()), lr=cfg.optimizer.lr
    )  # 5e5 for training, 5e-6 for lpips finetuning
    mse_loss_fct = torch.nn.MSELoss()
    lpips_loss_fct = lpips.LPIPS(net="vgg").cuda()

    num_steps = len(train_dataset_loader) * cfg.optimizer.num_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=cfg.optimizer.num_training_steps,
        num_training_steps=num_steps,
    )

    if args.from_epoch > 0:
        if not args.resume_from:
            raise ValueError("--from-epoch requires --resume-from to load optimizer/scheduler state")
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])

        scheduler.step(args.from_epoch * len(train_dataset_loader))

    triplane_decoder.train()
    triplane_encoder.train()

    best_psnr = 0
    best_lpips = 1

    num_imgs = cfg.dataset_params.train_data_loader.get("num_imgs", 1)

    if args.from_epoch > 0:
        start = args.from_epoch
    else:
        start = 0

    for epoch in range(start, num_steps // len(train_dataset_loader)):
        train_dataset_loader.dataset.part_num = (epoch * num_imgs) % (80 // num_imgs)
        try:
            triplane_decoder.train()
            triplane_encoder.train()

            loss_dict = {}
            loss_dict["loss"] = 0
            loss_dict["mse_loss"] = 0
            if cfg.optimizer.tv_loss_weight > 0:
                loss_dict["tv_loss"] = 0
            if cfg.optimizer.dist_loss_weight > 0:
                loss_dict["dist_loss"] = 0
            if cfg.optimizer.lpips_loss_weight > 0:
                loss_dict["lpips_loss"] = 0
            if cfg.optimizer.depth_loss_weight > 0:
                loss_dict["depth_loss"] = 0

            print(f"step {epoch}/100")
            if args.num_scenes > 0:
                total_scenes = min(args.num_scenes, len(train_dataset_loader))
            else:
                total_scenes = len(train_dataset_loader)

            pbar = tqdm(enumerate(train_dataset_loader), total=total_scenes)
            for i_iter_val, (imgs, img_metas, batch) in pbar:
                if (args.num_scenes > 0) and i_iter_val >= total_scenes:
                    break
                batch = torch.from_numpy(batch[0])
                # batch = batch[0]

                imgs = imgs.cuda()

                triplane, features = triplane_encoder(img=imgs, img_metas=img_metas)

                triplane_decoder.update_planes(triplane)
                if pif is not None:
                    meta = img_metas[0]
                    pif.update_proj_mat(meta["K"], meta["c2w"], meta["img_shape"][0][:2], meta["num_cams"])
                    pif.update_imgs(features[0])

                # train_step
                mask = batch[:, 9].bool()

                if mask.sum() > 0:
                    batch = batch.cuda()
                    mask = mask.cuda()
                    ray_origins = batch[:, :3]
                    ray_directions = batch[:, 3:6]
                    ground_truth_px_values = batch[:, 6:9]
                    if cfg.optimizer.depth_loss_weight > 0:
                        ground_truth_depth = batch[:, 10:]

                    if cfg.decoder.whiteout:
                        ground_truth_px_values[~mask] = 1

                    regenerated_px_values, dist_loss, depth = render_rays(
                        triplane_decoder, ray_origins, ray_directions, cfg, pif=pif, training=True
                    )

                    mse_loss = mse_loss_fct(regenerated_px_values, ground_truth_px_values)

                    tv_loss = (
                        cfg.optimizer.tv_loss_weight * compute_tv_loss(triplane_decoder)
                        if cfg.optimizer.tv_loss_weight > 0
                        else 0
                    )

                    dist_loss = cfg.optimizer.dist_loss_weight * dist_loss if cfg.optimizer.dist_loss_weight > 0 else 0

                    if cfg.optimizer.lpips_loss_weight > 0:
                        lpips_loss = cfg.optimizer.lpips_loss_weight * torch.mean(
                            lpips_loss_fct(
                                regenerated_px_values.view(-1, 48, 64, 3).permute(0, 3, 1, 2) * 2 - 1,
                                ground_truth_px_values.view(-1, 48, 64, 3).permute(0, 3, 1, 2) * 2 - 1,
                            )
                        )
                    else:
                        lpips_loss = 0

                    depth_loss = (
                        cfg.optimizer.depth_loss_weight
                        * mse_loss_fct(torch.sqrt(depth / 60), torch.sqrt(torch.clip(ground_truth_depth / 60, 0, 1)))
                        if cfg.optimizer.depth_loss_weight > 0
                        else 0
                    )

                    loss = mse_loss + tv_loss + dist_loss + lpips_loss + depth_loss

                    if loss.isnan():
                        print("Loss is NaN")
                        continue

                    optimizer.zero_grad()
                    loss.backward()
                    if cfg.optimizer.clip_grad_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(triplane_decoder.parameters(), cfg.optimizer.clip_grad_norm)
                        torch.nn.utils.clip_grad_norm_(triplane_encoder.parameters(), cfg.optimizer.clip_grad_norm)
                    optimizer.step()
                    scheduler.step(epoch * len(train_dataset_loader) + i_iter_val)

                    if (i_iter_val % 100) == 0:
                        loss_dict["loss"] += loss.item()
                        loss_dict["mse_loss"] += mse_loss.item()
                        if cfg.optimizer.tv_loss_weight > 0:
                            loss_dict["tv_loss"] += tv_loss.item()
                        if cfg.optimizer.dist_loss_weight > 0:
                            loss_dict["dist_loss"] += dist_loss.item()
                        if cfg.optimizer.lpips_loss_weight:
                            loss_dict["lpips_loss"] += lpips_loss.item()
                        if cfg.optimizer.depth_loss_weight > 0:
                            loss_dict["depth_loss"] += depth_loss.item()

                        pbar.set_description(f"loss: {loss.item():.4f}")
            for key in loss_dict.keys():
                loss_dict[key] /= len(train_dataset_loader) / 100

            mlflow.log_metrics({f"train/{k}": v for k, v in loss_dict.items()}, step=epoch)

            # save models
            if epoch % 10 == 0:
                torch.save(
                    {
                        "ttp": triplane_decoder.state_dict(),
                        "tpv": triplane_encoder.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    },
                    os.path.join(save_dir, f"model_{epoch}.pth"),
                )
            torch.save(
                {
                    "ttp": triplane_decoder.state_dict(),
                    "tpv": triplane_encoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                os.path.join(save_dir, "model_latest.pth"),
            )

            optimizer.zero_grad()

            with torch.no_grad():
                psnr_list = []
                lpips_list = []

                for i_iter_val, (imgs, img_metas, val_dataset) in enumerate(val_dataset_loader):
                    val_dataset = val_dataset[0].dataset
                    W, H = val_dataset.intrinsics.width, val_dataset.intrinsics.height

                    triplane_decoder.eval()
                    triplane_encoder.eval()

                    imgs = imgs.cuda()

                    triplane, features = triplane_encoder(img=imgs, img_metas=img_metas)

                    triplane_decoder.update_planes(triplane)
                    if pif is not None:
                        meta = img_metas[0]
                        pif.update_proj_mat(meta["K"], meta["c2w"], meta["img_shape"][0][:2], meta["num_cams"])
                        pif.update_imgs(features[0])

                    for img_index in np.arange(0, len(val_dataset) // (H * W)):
                        if img_index == 19:
                            break
                        ray_origins = val_dataset[img_index * H * W : (img_index + 1) * H * W, :3].cuda()
                        ray_directions = val_dataset[img_index * H * W : (img_index + 1) * H * W, 3:6].cuda()
                        ground_truth_image = (
                            val_dataset[img_index * H * W : (img_index + 1) * H * W, 6:9].reshape(H, W, 3).cuda()
                        )
                        mask = val_dataset[img_index * H * W : (img_index + 1) * H * W, 9].reshape(H, W).numpy()

                        data = []

                        for i in range(int(np.ceil(H / cfg.decoder.testing_batch_size))):
                            ray_origins_ = ray_origins[
                                i * W * cfg.decoder.testing_batch_size : (i + 1) * W * cfg.decoder.testing_batch_size
                            ]
                            ray_directions_ = ray_directions[
                                i * W * cfg.decoder.testing_batch_size : (i + 1) * W * cfg.decoder.testing_batch_size
                            ]
                            regenerated_px_values, dist_loss, _ = render_rays(
                                triplane_decoder, ray_origins_, ray_directions_, cfg, pif=pif, training=False
                            )
                            data.append(regenerated_px_values)

                        img = torch.cat(data).reshape(H, W, 3)
                        img = torch.clip(img, 0, 1)

                        if cfg.decoder.whiteout:
                            ground_truth_image[mask == 0] = 1

                        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                        ax[0].imshow(img.cpu())
                        ax[0].axis("off")
                        ax[0].set_title("Generated Image")
                        ax[1].imshow(ground_truth_image.cpu())
                        ax[1].axis("off")
                        ax[1].set_title("Ground Truth Image")

                        # figures logged at eval time
                        plt.close()

                        lpips_metric = torch.mean(
                            lpips_loss_fct(
                                img.view(1, H, W, 3).permute(0, 3, 1, 2) * 2 - 1,
                                ground_truth_image.view(1, H, W, 3).permute(0, 3, 1, 2) * 2 - 1,
                            )
                        ).item()
                        psnr = cv2.PSNR(img.cpu().numpy(), ground_truth_image.cpu().numpy(), R=1)
                        psnr_list.append(psnr)
                        lpips_list.append(lpips_metric)

                    num_vis_cams = img_metas[0]["num_cams"]
                    ncols = min(num_vis_cams, 3)
                    nrows = (num_vis_cams + ncols - 1) // ncols
                    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3))
                    if num_vis_cams == 1:
                        axes = np.array([axes])
                    axes = axes.flatten()
                    imgs_norm = torch.clip(imgs[0].detach().cpu() / 255 + 0.5, 0, 1).permute(0, 2, 3, 1)
                    for cam_i in range(num_vis_cams):
                        axes[cam_i].imshow(imgs_norm[cam_i][:, :, [2, 1, 0]])
                        axes[cam_i].axis("off")
                        axes[cam_i].set_title(f"Cam {cam_i}")
                    for ax_i in range(num_vis_cams, len(axes)):
                        axes[ax_i].axis("off")

                    plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.01, hspace=0.01)

                    # figures logged at eval time

                mlflow.log_metrics(
                    {"val/psnr": float(np.mean(psnr_list)), "val/lpips": float(np.mean(lpips_list))}, step=epoch
                )
                print(f"{args.log_dir} PSNR : {np.mean(psnr_list):.2f}, LPIPS : {np.mean(lpips_list):.2f}")

                if np.mean(psnr_list) > best_psnr:
                    torch.save(
                        {
                            "ttp": triplane_decoder.state_dict(),
                            "tpv": triplane_encoder.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "epoch": epoch,
                        },
                        os.path.join(save_dir, "model_best_psnr.pth"),
                    )
                    best_psnr = np.mean(psnr_list)
                if np.mean(lpips_list) < best_lpips:
                    torch.save(
                        {
                            "ttp": triplane_decoder.state_dict(),
                            "tpv": triplane_encoder.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "epoch": epoch,
                        },
                        os.path.join(save_dir, "model_best_lpips.pth"),
                    )
                    best_lpips = np.mean(lpips_list)

        except RuntimeError as e:
            print(e)

            torch.cuda.empty_cache()
            ckpt = torch.load(os.path.join(save_dir, "model_latest.pth"), map_location="cpu")
            triplane_decoder.load_state_dict(ckpt["ttp"])
            triplane_encoder.load_state_dict(ckpt["tpv"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])

    mlflow.end_run()


if __name__ == "__main__":
    # Eval settings
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--py-config", default="config/config.py")
    parser.add_argument("--ckpt-path", type=str, default="")
    parser.add_argument("--resume-from", type=str, default="")
    parser.add_argument("--log-dir", type=str, default="")
    parser.add_argument("--num-scenes", type=int, default=-1)
    parser.add_argument("--from-epoch", type=int, default="-1")
    parser.add_argument("--manifest-train", type=str, default="", help="Path to train.jsonl manifest")
    parser.add_argument("--manifest-val", type=str, default="", help="Path to val.jsonl manifest")
    args = parser.parse_args()

    ngpus = torch.cuda.device_count()
    ngpus = 0
    args.gpus = ngpus
    print(args)

    # torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
    main(0, args)
