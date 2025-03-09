#!/usr/bin/env python3
"""
a trainer class
"""
import datetime
import time
import torch
import torch.nn as nn
import os
import numpy as np
import pprint

from fvcore.common.config import CfgNode
from fvcore.common.checkpoint import Checkpointer

from ..engine.evaluator import Evaluator
from ..solver.lr_scheduler import make_scheduler
from ..solver.optimizer import make_optimizer
from ..utils import logging
from ..utils.train_utils import AverageMeter, gpu_mem_usage
from ..models.ppo import PPO

logger = logging.get_logger("visual_prompt")


class Trainer:
    """
    a trainer with below logics:

    1. Build optimizer, scheduler
    2. Load checkpoints if provided
    3. Train and eval at each epoch
    """
    def __init__(
        self,
        cfg: CfgNode,
        model: nn.Module,
        evaluator: Evaluator,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device

        # solver related
        logger.info("\tSetting up the optimizer...")
        if self.cfg.MODEL.PROMPT.ADAPTIVE:
            self.optimizer = make_optimizer([self.model], cfg.SOLVER, model.enc.get_prompt_params())
        else:
            self.optimizer = make_optimizer([self.model], cfg.SOLVER)
        self.scheduler = make_scheduler(self.optimizer, cfg.SOLVER)
        self.cls_criterion = nn.CrossEntropyLoss()

        self.checkpointer = Checkpointer(
            self.model,
            save_dir=cfg.OUTPUT_DIR,
            save_to_disk=True
        )

        if len(cfg.MODEL.WEIGHT_PATH) > 0:
            # only use this for vtab in-domain experiments
            checkpointables = [key for key in self.checkpointer.checkpointables if key not in ["head.last_layer.bias",  "head.last_layer.weight"]]
            self.checkpointer.load(cfg.MODEL.WEIGHT_PATH, checkpointables)
            logger.info(f"Model weight loaded from {cfg.MODEL.WEIGHT_PATH}")

        self.evaluator = evaluator
        self.cpu_device = torch.device("cpu")

    def forward_one_batch(self, inputs, targets, is_train, max_norm=-1, is_update=True):
        """Train a single (full) epoch on the model using the given
        data loader.

        Args:
            X: input dict
            targets
            is_train: bool
            max_norm
            is_update: bool
        Returns:
            loss
            outputs: output logits
        """
        # move data to device
        inputs = inputs.to(self.device, non_blocking=True)    # (batchsize, 2048)
        targets = targets.to(self.device, non_blocking=True)  # (batchsize, )

        if self.cfg.DBG:
            logger.info(f"shape of inputs: {inputs.shape}")
            logger.info(f"shape of targets: {targets.shape}")

        # forward
        with torch.set_grad_enabled(is_train):
            outputs = self.model(inputs)  # (batchsize, num_cls)
            if self.cfg.DBG:
                logger.info(
                    "shape of model output: {}, targets: {}".format(
                        outputs.shape, targets.shape))

            loss = self.cls_criterion(outputs, targets)

            if loss == float('inf'):
                logger.info(
                    "encountered infinite loss, skip gradient updating for this batch!"
                )
                return -1, -1
            elif torch.isnan(loss).any():
                logger.info(
                    "encountered nan loss, skip gradient updating for this batch!"
                )
                return -1, -1

        # =======backward and optim step only if in training phase... =========
        if is_train:
            self.optimizer.zero_grad()
            loss.backward()

            if max_norm != -1:  # for gradient clipping
                params = []
                for _, value in self.model.named_parameters():
                    if value.requires_grad:
                        params.append(value)
                if self.cfg.MODEL.PROMPT.ADAPTIVE:
                    params += self.model.enc.get_prompt_params()
                nn.utils.clip_grad_norm_(params, max_norm)

            if is_update:
                self.optimizer.step()

        return loss, outputs

    def get_input(self, data):
        if not isinstance(data["image"], torch.Tensor):
            for k, v in data.items():
                data[k] = torch.from_numpy(v)

        inputs = data["image"].float()
        labels = data["label"]
        return inputs, labels

    def train_classifier(self, train_loader, val_loader, test_loader):
        """
        Train a classifier using epoch
        """
        # save the model prompt if required before training
        self.model.eval()

        # setup training epoch params
        total_epoch = self.cfg.SOLVER.TOTAL_EPOCH
        total_data = len(train_loader)
        best_epoch = -1
        best_metric = 0
        log_interval = self.cfg.SOLVER.LOG_EVERY_N

        losses = AverageMeter('Loss', ':.4e')
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')

        patience = 0  # if > self.cfg.SOLVER.PATIENCE, stop training

        if self.cfg.MODEL.PROMPT.ADAPTIVE:
            flag = False
            num_blocks = self.model.enc.get_num_blk()
            warmup_epoch = self.cfg.MODEL.PROMPT.REWARD_WARMUP_EPOCH
            if self.cfg.MODEL.PROMPT.PPO:
                """ PPO (for RL) """
                ppo_upd_iter = 0
                ppo_agent = PPO(state_dim=3 * num_blocks,  # scores + prompt distribution + layer index
                                action_dim=num_blocks,  # no action 0
                                gamma=self.cfg.MODEL.PROMPT.PPO_DISCOUNT_FACTOR,  # discount factor
                                K_epochs=self.cfg.MODEL.PROMPT.PPO_K_EPOCH,
                                lr_actor=0.0003, lr_critic=0.001, eps_clip=0.2,
                                has_continuous_action_space=False)
            else:
                """ TS (for MAB) """
                action_dim = num_blocks
                num_successes = np.ones(action_dim)
                num_failures = np.ones(action_dim)
                reward = np.zeros(action_dim)

        for epoch in range(total_epoch):
            # reset averagemeters to measure per-epoch results
            losses.reset()
            batch_time.reset()
            data_time.reset()

            if self.scheduler is not None:
                lr = self.scheduler.get_lr()[0]
                logger.info(
                    "Training {} / {} epoch, with learning rate {}".format(
                        epoch + 1, total_epoch, lr
                    )
                )
            else:
                logger.info("Training {} / {} epoch".format(epoch + 1, total_epoch))

            # Enable training mode
            self.model.train()

            end = time.time()

            if self.cfg.MODEL.PROMPT.ADAPTIVE:
                scores = []
                avg_train_loss = 0
                if self.cfg.MODEL.PROMPT.GATE_GRADS_IDENT:
                    self.model.enc.start_gate_ident()
                    for idx, input_data in enumerate(train_loader):  # one epoch
                        X, targets = self.get_input(input_data)
                        train_loss, _ = self.forward_one_batch(X, targets, True, max_norm=self.cfg.MODEL.PROMPT.CLP_MAX_NORM, is_update=False)
                        avg_train_loss += train_loss.item() / X.shape[0]
                        prompt_gates_per_layer = self.model.enc.get_prompt_gates_per_layer()
                        for blk_idx, gates in enumerate(prompt_gates_per_layer):
                            if idx == 0:
                                scores.append(np.array([gate.grad.detach().cpu().item() for gate in gates]))
                            else:
                                scores[blk_idx] += np.array([gate.grad.detach().cpu().item() for gate in gates])
                    for blk_idx in range(num_blocks):
                        scores[blk_idx] /= len(train_loader)
                else:
                    self.model.enc.prompt_norm_ident()
                    orig_scores = []
                    for blk_idx in range(num_blocks):
                        blk_scores = self.model.enc.get_blk_prompt_norm(blk_idx)
                        orig_scores.append(blk_scores)
                        blk_scores = - blk_scores / sum(blk_scores)  # for maximum goal
                        scores.append(blk_scores)

                max_scores, max_scores_indices, scores_layerwise = [], [], []
                for blk_idx in range(num_blocks):
                    if scores[blk_idx].any():
                        max_scores.append(max(scores[blk_idx]))
                        max_scores_indices.append(np.argmax(scores[blk_idx]))
                        if self.cfg.MODEL.PROMPT.GATE_GRADS_IDENT:
                            scores_layerwise.append(sum(scores[blk_idx]))
                        else:
                            scores_layerwise.append(sum(orig_scores[blk_idx]))  # for states in PPO
                    else:
                        max_scores.append(-10000)
                        max_scores_indices.append(-10000)
                        scores_layerwise.append(0)
                scores_lw_str = ", ".join(map(str, scores_layerwise))
                logger.info(f"Scores: [{scores_lw_str}]")

                # ==================== REWARD ====================
                if self.cfg.MODEL.PROMPT.GATE_GRADS_IDENT:
                    curr_loss = avg_train_loss
                    if epoch > warmup_epoch and flag:
                        if self.cfg.MODEL.PROMPT.PPO:
                            """ PPO (for RL) """
                            ppo_upd_iter += 1
                            actual_reward = prev_loss - curr_loss - max(max_scores)
                            logger.info(f"Actual Reward: {actual_reward}")
                            ppo_agent.buffer.rewards.append(actual_reward)
                            ppo_agent.buffer.is_terminals.append(False)
                            if ppo_upd_iter % 2 == 0:
                                avg_loss = ppo_agent.update()
                                logger.info(f'Policy Network\'s Loss: {avg_loss:.4f}')
                        else:
                            """ TS (for MAB) """
                            if curr_loss + max(max_scores) <= prev_loss:
                                num_successes[transfer_to] += 1
                                logger.info('Success (for TS)')
                            else:
                                num_failures[transfer_to] += 1
                                logger.info('Failure (for TS)')
                    prev_loss = curr_loss
                else:
                    if epoch > warmup_epoch and flag:
                        if self.cfg.MODEL.PROMPT.PPO:
                            """ PPO (for RL) """
                            ppo_upd_iter += 1
                            actual_reward = prev_loss - curr_loss
                            logger.info(f"Actual Reward: {actual_reward}")
                            ppo_agent.buffer.rewards.append(actual_reward)
                            ppo_agent.buffer.is_terminals.append(False)
                            if ppo_upd_iter % 2 == 0:
                                avg_loss = ppo_agent.update()
                                logger.info(f'Policy Network\'s Loss: {avg_loss:.4f}')
                # ==================== REWARD ====================

                flag = False
                if epoch > warmup_epoch - 1:
                    if self.cfg.MODEL.PROMPT.GATE_GRADS_IDENT and np.max(max_scores) > 0:
                        flag = True
                    elif not self.cfg.MODEL.PROMPT.GATE_GRADS_IDENT:
                        flag = True

                if flag:
                    transfer_from_layer = np.argmax(max_scores)
                    transfer_from_prompt = max_scores_indices[transfer_from_layer]

                    if self.cfg.MODEL.PROMPT.PPO:
                        """ PPO (for RL) """
                        scores_layerwise = np.array(scores_layerwise)
                        one_hot_enc = np.eye(num_blocks)
                        cur_prompt_distribution = self.model.enc.get_cur_prompt_distribution()
                        cur_prompt_distribution = np.array(cur_prompt_distribution, dtype='float64')
                        cur_prompt_distribution -= one_hot_enc[transfer_from_layer]
                        # context: scores + prompt distribution + identified layer index
                        context = np.concatenate([
                            (scores_layerwise - np.mean(scores_layerwise)) / (np.std(scores_layerwise) + 1e-8),
                            cur_prompt_distribution / sum(cur_prompt_distribution),
                            one_hot_enc[transfer_from_layer]
                        ])
                        transfer_to = ppo_agent.select_action(context)
                    else:
                        """ TS (for MAB) """
                        for k in range(action_dim):
                            reward[k] = np.random.beta(num_successes[k], num_failures[k])
                        transfer_to = np.argmax(reward)

                    self.model.enc.relocation(transfer_from_layer, transfer_from_prompt, transfer_to)
                    logger.info(f"Decision: Transferring from Block {transfer_from_layer+1} to Block {transfer_to+1}")
                else:
                    logger.info(f"No Relocation at Epoch {epoch + 1}")

                if self.cfg.MODEL.PROMPT.GATE_GRADS_IDENT:
                    self.model.enc.end_gate_ident()

            for idx, input_data in enumerate(train_loader):
                if self.cfg.DBG and idx == 20:
                    # if debugging, only need to see the first few iterations
                    break

                X, targets = self.get_input(input_data)
                # logger.info(X.shape)
                # logger.info(targets.shape)
                # measure data loading time
                data_time.update(time.time() - end)

                train_loss, _ = self.forward_one_batch(X, targets, True, max_norm=self.cfg.MODEL.PROMPT.CLP_MAX_NORM)

                if train_loss == -1:
                    # continue
                    return None

                losses.update(train_loss.item(), X.shape[0])

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # log during one batch
                if (idx + 1) % log_interval == 0:
                    seconds_per_batch = batch_time.val
                    eta = datetime.timedelta(seconds=int(
                        seconds_per_batch * (total_data - idx - 1) + seconds_per_batch*total_data*(total_epoch-epoch-1)))
                    logger.info(
                        "\tTraining {}/{}. train loss: {:.4f},".format(
                            idx + 1,
                            total_data,
                            train_loss
                        )
                        + "\t{:.4f} s / batch. (data: {:.2e}). ETA={}, ".format(
                            seconds_per_batch,
                            data_time.val,
                            str(eta),
                        )
                        + "max mem: {:.1f} GB ".format(gpu_mem_usage())
                    )
            logger.info(
                "Epoch {} / {}: ".format(epoch + 1, total_epoch)
                + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                    data_time.avg, batch_time.avg)
                + "average train loss: {:.4f}".format(losses.avg))
            # update lr, scheduler.step() must be called after optimizer.step() according to the docs: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate  # noqa
            if self.scheduler is not None:
                self.scheduler.step()

            # Enable eval mode
            self.model.eval()

            # eval at each epoch for single gpu training
            self.evaluator.update_iteration(epoch)
            self.eval_classifier(val_loader, "val", self.cfg.DATA.NAME, epoch == total_epoch - 1)
            if test_loader is not None:
                self.eval_classifier(
                    test_loader, "test", self.cfg.DATA.NAME, epoch == total_epoch - 1)

            # check the patience
            # t_name = "val_" + val_loader.dataset.name
            t_name = "val_" + self.cfg.DATA.NAME
            try:
                curr_acc = self.evaluator.results[f"epoch_{epoch}"]["classification"][t_name]["top1"]
            except KeyError:
                return

            if curr_acc > best_metric:
                best_metric = curr_acc
                best_epoch = epoch + 1
                logger.info(
                    f'Best epoch {best_epoch}: best metric: {best_metric:.3f}')
                patience = 0
            else:
                patience += 1
            if patience >= self.cfg.SOLVER.PATIENCE:
                logger.info("No improvement. Breaking out of loop.")
                break

            if self.cfg.MODEL.PROMPT.ADAPTIVE:
                cur_prompt_distribution = self.model.enc.get_cur_prompt_distribution()
                logger.info("Current Prompt Distribution: " + pprint.pformat(cur_prompt_distribution))

        # save the last checkpoints
        # if self.cfg.MODEL.SAVE_CKPT:
        #     Checkpointer(
        #         self.model,
        #         save_dir=self.cfg.OUTPUT_DIR,
        #         save_to_disk=True
        #     ).save("last_model")

    # @torch.no_grad()
    # def save_prompt(self, epoch):
    #     # only save the prompt embed if below conditions are satisfied
    #     if self.cfg.MODEL.PROMPT.SAVE_FOR_EACH_EPOCH:
    #         if self.cfg.MODEL.TYPE == "vit" and "prompt" in self.cfg.MODEL.TRANSFER_TYPE:
    #             prompt_embds = self.model.enc.transformer.prompt_embeddings.cpu().numpy()
    #             out = {"shallow_prompt": prompt_embds}
    #             if self.cfg.MODEL.PROMPT.DEEP:
    #                 deep_embds = self.model.enc.transformer.deep_prompt_embeddings.cpu().numpy()
    #                 out["deep_prompt"] = deep_embds
    #             torch.save(out, os.path.join(
    #                 self.cfg.OUTPUT_DIR, f"prompt_ep{epoch}.pth"))

    @torch.no_grad()
    def eval_classifier(self, data_loader, prefix, dataset_name, save=False):
        """evaluate classifier"""
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        test_name = prefix + "_" + dataset_name
        total = len(data_loader)

        # initialize features and target
        total_logits = []
        total_targets = []

        for idx, input_data in enumerate(data_loader):
            end = time.time()
            X, targets = self.get_input(input_data)
            # measure data loading time
            data_time.update(time.time() - end)

            if self.cfg.DBG:
                logger.info("during eval: {}".format(X.shape))
            loss, outputs = self.forward_one_batch(X, targets, False)
            if loss == -1:
                return
            losses.update(loss, X.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)

            if (idx + 1) % log_interval == 0:
                logger.info(
                    "\tTest {}/{}. loss: {:.3f}, {:.4f} s / batch. (data: {:.2e})".format(  # noqa
                        idx + 1,
                        total,
                        losses.val,
                        batch_time.val,
                        data_time.val
                    ) + "max mem: {:.5f} GB ".format(gpu_mem_usage())
                )

            # targets: List[int]
            total_targets.extend(list(targets.numpy()))
            total_logits.append(outputs)
        logger.info(
            f"Inference ({prefix}):"
            + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                data_time.avg, batch_time.avg)
            + "average loss: {:.4f}".format(losses.avg))
        if self.model.side is not None:
            logger.info(
                "--> side tuning alpha = {:.4f}".format(self.model.side_alpha))
        # total_testimages x num_classes
        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()
        self.evaluator.classify(
            joint_logits, total_targets,
            test_name, self.cfg.DATA.MULTILABEL,
        )

        # save the probs and targets
        if save and self.cfg.MODEL.SAVE_CKPT:
            out = {"targets": total_targets, "joint_logits": joint_logits}
            out_path = os.path.join(
                self.cfg.OUTPUT_DIR, f"{test_name}_logits.pth")
            torch.save(out, out_path)
            logger.info(
                f"Saved logits and targets for {test_name} at {out_path}")
