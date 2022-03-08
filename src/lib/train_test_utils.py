# -*- coding: utf-8 -*-

"""
utils for train & test
"""


import numpy as np
import torch
import tqdm


def train(model, loss_fn, optimizer, train_dataloader, device="cpu"):
    model.train()
    epoch_losses = []
    tqdm_iter = tqdm.tqdm(train_dataloader)
    for i, batch in enumerate(tqdm_iter):
        input_data = batch["image"].to(device)
        targets = batch["class_name"]
        pred_targets = model(input_data).cpu()
        loss = loss_fn(pred_targets, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_losses.append(loss.item())
        tqdm_iter.set_description(f"mean train loss: {np.mean(epoch_losses):.4f}")
    return np.mean(epoch_losses)


def validate(model, loss_fn, validate_dataloader, device="cpu"):
    model.eval()
    epoch_losses = []
    tqdm_iter = tqdm.tqdm(validate_dataloader)

    for i, batch in enumerate(tqdm_iter):
        input_data = batch["image"].to(device)
        targets = batch["class_name"]
        with torch.no_grad():
            pred_targets = model(input_data).cpu()
        loss = loss_fn(pred_targets, targets)
        epoch_losses.append(loss.item())
        tqdm_iter.set_description(f"mean validate loss: {np.mean(epoch_losses):.4f}")
    return np.mean(epoch_losses)


def predict_from_dataloader(model, dataloader,  device="cpu"):
    model.eval()

    tqdm_iter = tqdm.tqdm(dataloader, desc="predict classes")
    predict_arr = []
    ground_truth_arr = []

    for i, batch in enumerate(tqdm_iter):
        input_data = batch["image"].to(device)
        targets = batch["class_name"]
        with torch.no_grad():
            logits = model(input_data).cpu()
        pred_targets = torch.argmax(logits, dim=1)

        predict_arr.append(pred_targets.numpy())
        ground_truth_arr.append(targets.numpy())

    return np.concatenate(ground_truth_arr), np.concatenate(predict_arr)
