import time
import cv2
import torch
from tqdm import tqdm
import pytorch_warmup as warmup
import numpy as np
import random
import os
import pickle as pkl
from pathlib import Path

from clrnet.models.registry import build_net
from .registry import build_trainer, build_evaluator
from .optimizer import build_optimizer
from .scheduler import build_scheduler
from clrnet.datasets import build_dataloader
from clrnet.utils.recorder import build_recorder
from clrnet.utils.net_utils import save_model, load_network, resume_network
from mmcv.parallel import MMDataParallel


class Runner(object):
    def __init__(self, cfg):
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        self.cfg = cfg
        self.recorder = build_recorder(self.cfg)
        self.net = build_net(self.cfg)
        self.net = MMDataParallel(self.net, device_ids=range(self.cfg.gpus)).cuda()
        self.recorder.logger.info("Network: \n" + str(self.net))
        self.resume()
        self.optimizer = build_optimizer(self.cfg, self.net)
        self.scheduler = build_scheduler(self.cfg, self.optimizer)
        self.metric = 0.0
        self.val_loader = None
        self.test_loader = None

    def to_cuda(self, batch):
        for k in batch:
            if not isinstance(batch[k], torch.Tensor):
                continue
            batch[k] = batch[k].cuda()
        return batch

    def resume(self):
        if not self.cfg.load_from and not self.cfg.finetune_from:
            return
        load_network(
            self.net,
            self.cfg.load_from,
            finetune_from=self.cfg.finetune_from,
            logger=self.recorder.logger,
        )

    def train_epoch(self, epoch, train_loader):
        self.net.train()
        end = time.time()
        max_iter = len(train_loader)
        for i, data in enumerate(train_loader):
            if self.recorder.step >= self.cfg.total_iter:
                break
            date_time = time.time() - end
            self.recorder.step += 1
            data = self.to_cuda(data)  # data: Dict[str, Any]
            output = self.net(data)
            self.optimizer.zero_grad()
            loss = output["loss"].sum()
            loss.backward()
            self.optimizer.step()
            if not self.cfg.lr_update_by_epoch:
                self.scheduler.step()
            batch_time = time.time() - end
            end = time.time()
            self.recorder.update_loss_stats(output["loss_stats"])
            self.recorder.batch_time.update(batch_time)
            self.recorder.data_time.update(date_time)

            if i % self.cfg.log_interval == 0 or i == max_iter - 1:
                lr = self.optimizer.param_groups[0]["lr"]
                self.recorder.lr = lr
                self.recorder.record("train")

    def train(self):
        self.recorder.logger.info("Build train loader...")
        train_loader = build_dataloader(self.cfg.dataset.train, self.cfg, is_train=True)

        self.recorder.logger.info("Start training...")
        start_epoch = 0
        if self.cfg.resume_from:
            start_epoch = resume_network(
                self.cfg.resume_from,
                self.net,
                self.optimizer,
                self.scheduler,
                self.recorder,
            )
        for epoch in range(start_epoch, self.cfg.epochs):
            self.recorder.epoch = epoch
            self.train_epoch(epoch, train_loader)  # ! debug
            if (epoch + 1) % self.cfg.save_ep == 0 or epoch == self.cfg.epochs - 1:
                self.recorder.logger.info("Saving checkpoint...")
                self.save_ckpt()
            if (epoch + 1) % self.cfg.eval_ep == 0 or epoch == self.cfg.epochs - 1:
                self.recorder.logger.info("Start validating...")
                self.validate()
            # if self.recorder.step >= self.cfg.total_iter:  # TODO: remove to ignore total_iter
            #     break
            if self.cfg.lr_update_by_epoch:
                self.scheduler.step()

    def test(self):
        if not self.test_loader:
            self.test_loader = build_dataloader(
                self.cfg.dataset.test, self.cfg, is_train=False
            )
        self.net.eval()
        predictions = []
        pred_feat_list = []
        for i, data in enumerate(tqdm(self.test_loader, desc=f"Testing")):
            data = self.to_cuda(data)
            with torch.no_grad():
                output = self.net(data)
                output, lanes_feat_list = self.net.module.heads.get_lanes(output)
                predictions.extend(output)
                pred_feat_list.extend(lanes_feat_list)
            if self.cfg.view:
                self.test_loader.dataset.view(output, data["meta"])

        metric = self.test_loader.dataset.evaluate(predictions, self.cfg.work_dir)
        if metric is not None:
            self.recorder.logger.info("metric: " + str(metric))

    def validate(self):
        if not self.val_loader:
            self.val_loader = build_dataloader(
                self.cfg.dataset.val, self.cfg, is_train=False
            )
        self.net.eval()
        preds = []
        pred_meta = []
        pred_feat_list = []
        for _, data in enumerate(tqdm(self.val_loader, desc=f"Validate")):
            data = self.to_cuda(data)
            with torch.no_grad():
                output = self.net(data)
                output, lanes_feat_list = self.net.module.heads.get_lanes(output)

                preds.extend(
                    output
                )  # list of each image's lanes, each element is a list of lanes, each lane is type Lane
                pred_feat_list.extend(lanes_feat_list)
                pred_meta.extend(data["meta"].data[0])
            if self.cfg.view:
                self.val_loader.dataset.view(output, data["meta"])

        # ! save the prediction metadata
        # logit_list = []
        # prob_list = []
        # lane_list = []

        entries = []
        for i, (pred, meta, pred_feat) in enumerate(
            zip(preds, pred_meta, pred_feat_list)
        ):
            logits = []
            probs = []
            lanes = self.val_loader.dataset.get_prediction_arr(
                pred
            )  # list of [N, 2] array
            lanes_feat = [feat.cpu().detach().numpy() for feat in pred_feat]
            for lane in pred:
                logits.append(lane.metadata["logit"])  # a list of [2,] array
                probs.append(float(lane.metadata["prob"]))
            # logit_list.append(probs)
            # prob_list.append(logits)
            # lane_list.append(lanes)

            entries.append(
                {
                    "logits": np.array(logits),  # [N, 2]
                    "probs": np.array(probs),  # [N, 1]
                    "lanes": lanes,  # list of [N*, 2] array, N* is not fixed
                    "lanes_feat": lanes_feat,
                    "full_img_path": meta["full_img_path"],
                    "img_name": meta["img_name"],
                }
            )

        pkl_path = os.path.join(self.cfg.work_dir, "pred_results.pkl")

        with open(pkl_path, "wb") as f:
            # pred_metadata = {
            #     "logit": logit_list,
            #     "prob": prob_list,
            # }
            # pkl.dump(pred_metadata, f)
            pkl.dump(entries, f)
        # ! end

        metric = self.val_loader.dataset.evaluate(
            predictions=preds, output_basedir=self.cfg.work_dir
        )
        self.recorder.logger.info("metric: " + str(metric))

    def demo(self, src: str):
        dataset = build_dataloader

    def save_ckpt(self, is_best=False):
        save_model(self.net, self.optimizer, self.scheduler, self.recorder, is_best)
