import os 
import torch 
from torch import nn
from torchvision import models
from pytorch_metric_learning import losses, miners
import pytorch_lightning as pl
import lightning as L
from dataloaders.usearch_db import UsearchDB

class BoQModel(L.LightningModule):
    def __init__(self, backbone, aggregator, model_config):
        super().__init__()
        # backbone
        self.backbone = backbone 
        
        # Initialize projection head (SimCLR)
        self.aggregator = aggregator
        
        learning_rate_config = model_config["learning_rate"]
        self.lr = float(learning_rate_config["lr"])
        self.lr_mul = learning_rate_config["lr_mul"]
        self.weight_decay = float(learning_rate_config["weight_decay"])
        self.milestones = learning_rate_config["milestones"]
        self.warmup_epochs = model_config["warmup_epochs"]
        self.silent = model_config["silent"] # disable console output
        
        # Initialize loss function and miner (per original BoQ code)
        loss_config = model_config["loss"]
        self.ms_loss = losses.MultiSimilarityLoss(alpha=loss_config["alpha"],
                                                  beta=loss_config["beta"], 
                                                  base=loss_config["base"])
        self.ms_miner = miners.MultiSimilarityMiner(epsilon=model_config["miner"]["epsilon"])

        #cache train loss / epoch 
        self.train_loss_epoch = list()

        # Validation infrastructure
        self.val_embeddings = []
        self.val_labels = []
        self.usearch_config = model_config["usearch"]

    def configure_optimizers(self):
        optimizer_params = [
            {"params": self.backbone.parameters(), "lr": self.lr, "weight_decay": self.weight_decay},
            {"params": self.aggregator.parameters(), "lr": self.lr, "weight_decay": self.weight_decay},
        ]
        optimizer = torch.optim.AdamW(optimizer_params)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.milestones, gamma=self.lr_mul
        )    
        return [optimizer], [scheduler]
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # warmup learning rate
        if self.trainer.current_epoch < self.warmup_epochs:
            total_warmup_steps = self.warmup_epochs * self.trainer.num_training_batches
            lr_scale = (self.trainer.global_step + 1) / total_warmup_steps
            lr_scale = min(1.0, lr_scale)
            for pg in optimizer.param_groups:
                initial_lr = float(pg.get("initial_lr", self.lr))
                # breakpoint()
                pg["lr"] = lr_scale * initial_lr

        optimizer.step(closure=optimizer_closure)
        self.log('_LR', optimizer.param_groups[-1]['lr'], prog_bar=False, logger=True)
    
    @torch.compiler.disable()
    def compute_loss(self, anchor_desc, positive_desc, negative_descs, anchor_labels, neg_labels):
        # Concatenate all embeddings
        all_embeddings = torch.cat([
            anchor_desc,
            positive_desc,
            negative_descs
        ], dim=0)
        
        # Construct labels
        all_labels = torch.cat([
            anchor_labels,
            anchor_labels,
            torch.tensor(neg_labels).view(-1).to(anchor_labels.device)
        ], dim=0)
        
        # Mine pairs and compute loss
        mined_pairs = self.ms_miner(all_embeddings, all_labels)
        loss = self.ms_loss(all_embeddings, all_labels, mined_pairs)
        
        return loss

    def handle_dimension(self, x):
        if x.dim() == 5:
            B, N, C, H, W = x.shape # (batch_size, num_negatives, channels, height, width)
            x = x.view(B * N, C, H, W) # (batch_size * num_negatives, channels, height, width)
        return x
    
    def forward(self, x): 
        x = self.backbone(x)
        x, _ = self.aggregator(x)
        return x, _
    
    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        anchor_imgs, pos_imgs, neg_imgs, anchor_labels, pos_labels, neg_labels = batch
        
        anchor_imgs = self.handle_dimension(anchor_imgs)
        pos_imgs = self.handle_dimension(pos_imgs)
        # neg_imgs = self.handle_dimension(neg_imgs)

        anchor_desc, _ = self(anchor_imgs)
        pos_desc, _ = self(pos_imgs)
        
        batch_size, num_neg, c, h, w = neg_imgs.shape
        neg_imgs_flat = neg_imgs.view(-1, c, h, w)
        neg_imgs_flat = self.handle_dimension(neg_imgs_flat)
        neg_desc_flat, _ = self(neg_imgs_flat)
        
        loss = self.compute_loss(anchor_desc, pos_desc, neg_desc_flat, anchor_labels, neg_labels)
        self.train_loss_epoch.append(loss.item())
        return loss
    
    def on_train_epoch_end(self):
        avg_loss = torch.tensor(self.train_loss_epoch).mean()
        self.log("avg_train_loss_epoch", avg_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.train_loss_epoch.clear()

    def on_validation_epoch_start(self):
    # Initialize a fresh database at the start of each validation epoch
        self.usearch_db = UsearchDB(
            embedding_dim=self.usearch_config["embedding_dim"],
            metric=self.usearch_config["metric"],
            dtype=self.usearch_config["dtype"]
        )
        self.val_embeddings.clear()  # Stores query embeddings
        self.val_labels.clear()     # Stores query labels

    def validation_step(self, batch, batch_idx):
        """Process validation batch of candidate-query pairs"""
        candidate_imgs, candidate_labels, query_imgs, query_labels = batch
        candidate_imgs = self.handle_dimension(candidate_imgs)
        query_imgs = self.handle_dimension(query_imgs)

        # Generate embeddings for candidates and queries
        candidate_embeds, _ = self(candidate_imgs)
        query_embeds, _ = self(query_imgs)
        
        # Add candidates to search database with their labels
        self.usearch_db.add(
            embeddings=candidate_embeds.cpu().detach().numpy(),
            labels=candidate_labels.cpu().detach().numpy()
        )
        
        # Store query embeddings and labels for evaluation
        self.val_embeddings.extend(query_embeds.cpu().numpy())
        self.val_labels.extend(query_labels.cpu().numpy())

    def on_validation_epoch_end(self):
        """Calculate and log retrieval metrics"""
        top_k = self.usearch_config["top_k"]
        top_k_acc = []
        top_1_acc = []
        mrr_values = []
        match_distances = []
        
        # Get all candidate labels from the database
        candidate_labels = self.usearch_db.labels  # Assumes labels are stored as a list/array
        
        # Process each query
        for query_embed, query_label in zip(self.val_embeddings, self.val_labels):
            # Retrieve top-k candidates (indices and distances)
            results = self.usearch_db.search(query_embed, k=top_k).to_list()
            
            correct_rank = None
            for rank, (result_idx, distance) in enumerate(results, start=1):
                if candidate_labels[result_idx] == query_label:
                    correct_rank = rank
                    match_distances.append(distance)
                    break
            
            # Update metrics
            if correct_rank is not None:
                top_k_acc.append(int(correct_rank <= top_k))
                top_1_acc.append(int(correct_rank == 1))
                mrr_values.append(1.0 / correct_rank)
            else:
                top_k_acc.append(0)
                top_1_acc.append(0)
                mrr_values.append(0.0)
        
        # Convert to tensors for logging
        avg_top_k = torch.tensor(top_k_acc).float().mean()
        avg_top_1 = torch.tensor(top_1_acc).float().mean()
        avg_mrr = torch.tensor(mrr_values).float().mean()
        
        # Log metrics
        self.log("val_top_k_accuracy", avg_top_k, prog_bar=True, logger=True, on_epoch=True)
        self.log("val_top_1_accuracy", avg_top_1, prog_bar=True, logger=True, on_epoch=True)
        self.log("val_mrr", avg_mrr, prog_bar=True, logger=True, on_epoch=True)
        
        # Log distance statistics if matches found
        if match_distances:
            dist_tensor = torch.tensor(match_distances).float()
            self.log("val_match_dist/mean", dist_tensor.mean())
            self.log("val_match_dist/min", dist_tensor.min())
            self.log("val_match_dist/max", dist_tensor.max())
        else:
            self.log("val_match_dist/mean", torch.tensor(0.0))
            self.log("val_match_dist/min", torch.tensor(0.0))
            self.log("val_match_dist/max", torch.tensor(0.0))