import torch
import torch.nn as nn
import time
import datetime
import pytorch_lightning as pl
import os
import numpy as np
import pandas as pd
import json
import wandb
from spared.metrics import get_metrics
from train_v1 import pearsonr, compute_correlations

class TrainerModel(pl.LightningModule):
    
    def __init__(self, config,  model):
        super().__init__()
        self.model = model
        self.config = config
        self.criterion = nn.MSELoss()
        self.automatic_optimization = False
        self.min_loss  = float("inf")
        self.max_corr  = float("-inf")
        self.min_eval_loss = float("inf")
        if self.config.opt_metric == "MSE" or self.config.opt_metric == "MAE":
            self.eval_opt_metric = float("inf")  
        else:
            self.eval_opt_metric = float("-inf")
        self.start_time  = None
        self.last_saved = None
        self.best_metrics = None
        self.validation_step_outputs = []
           
    def correlationMetric(self,x, y):
      corr = 0
      for idx in range(x.size(1)):
          corr += pearsonr(x[:,idx], y[:,idx])
      corr /= (idx + 1)
      return (1 - corr).mean()
    
    def training_step(self,data,idx):
        if self.current_epoch == 0 and idx == 0:
            self.start_time  = time.time()
        
        optimizer = self.optimizers()
        pred_count = self.model(data.x_dict,data.edge_index_dict)
        loss   = self.criterion(pred_count,data["window"]["y"])
        corrloss = self.correlationMetric(pred_count,data["window"]["y"])
        mask = data["window"]["mask"]
        metrics = get_metrics(data["window"]["y"],pred_count,mask)
        metrics_df = pd.DataFrame(metrics, index=[0])
        train_dict={f'train_{key}': val for key, val in metrics.items()}
        train_dict["epoch"]=self.current_epoch
        wandb.log(train_dict)
        optimizer.zero_grad()
        self.manual_backward(loss + corrloss * 0.5)
        optimizer.step()
       
        self.produce_log(loss.detach(),corrloss.detach(),idx)
        
        
    def produce_log(self,loss,corr,idx):
        
        train_loss = self.all_gather(loss).mean().item()
        train_corr = self.all_gather(corr).mean().item()
        
        self.min_loss   = min(self.min_loss, train_loss)
        
        if self.trainer.is_global_zero and loss.device.index == 0 and idx % self.config.verbose_step == 0:
            
            current_lr = self.optimizers().param_groups[0]['lr']
            
            len_loader = self.config.max_steps
            
            batches_done = self.current_epoch  * len_loader + idx + 1
            batches_left = self.trainer.max_epochs * len_loader - batches_done
            time_left    = datetime.timedelta(seconds = batches_left * (time.time() - self.start_time) / batches_done)
                    
            self.config.logfun(
                        "[Epoch %d/%d] [Batch %d/%d] [Loss: %f, Corr: %f, lr: %f] [Min Loss: %f] ETA: %s" % 
                        (self.current_epoch,
                         self.trainer.max_epochs,
                         idx,
                         len_loader,
                         train_loss,
                         train_corr,
                         current_lr,
                         self.min_loss,
                         time_left
                            )
                        
                        )
            
    def validation_step(self,data,idx):
        mask = data["window"]["mask"]
        pred_count = self.model(data.x_dict,data.edge_index_dict)
        outputs = (pred_count,data["window"]["y"],mask)
        self.validation_step_outputs.append(outputs)
        return outputs
    
    def test_step(self,data,idx):
        mask = data["window"]["mask"]
        pred_count = self.model(data.x_dict,data.edge_index_dict)
        metrics = get_metrics(data["window"]["y"],pred_count,mask)
        test_dict={f'test_{key}': val for key, val in metrics.items()}
        test_dict["epoch"]=self.current_epoch
        wandb.log(test_dict)
        return pred_count,data["window"]["y"]
        
    def on_validation_epoch_end(self):
        
        outputs = self.validation_step_outputs
        logfun = self.config.logfun       
        pred_count = torch.cat([i[0] for i in outputs])
        count = torch.cat([i[1] for i in outputs])
        mask = torch.cat([i[2] for i in outputs])
        pred_count = self.all_gather(pred_count).view(-1,self.config.num_genes)
        count = self.all_gather(count).view(-1,self.config.num_genes)
        mask = self.all_gather(mask).view(-1,self.config.num_genes)
        metrics = get_metrics(count,pred_count,mask)
        val_dict={f'val_{key}': val for key, val in metrics.items()}
        val_dict["epoch"]=self.current_epoch
        wandb.log(val_dict)
        
        total_loss = self.criterion(pred_count,count).item()
        self.log('val_loss', total_loss)
        gene_corr = compute_correlations(count, pred_count, True)
        corr = np.mean(gene_corr)
        
        if self.trainer.is_global_zero and self.trainer.num_devices != 0:
            if self.config.opt_metric == "MSE" or self.config.opt_metric == "MAE":
                if metrics[self.config.opt_metric] < self.eval_opt_metric:
                    self.best_metrics = metrics
                    self.save(self.current_epoch, total_loss, metrics[self.config.opt_metric])
                    bestval_dict={f'best_val_{key}': val for key, val in self.best_metrics.items()}
                    bestval_dict["epoch"]=self.current_epoch
                    wandb.log(bestval_dict)
                self.min_eval_loss = min(self.min_eval_loss, total_loss)
                self.eval_opt_metric = min(metrics[self.config.opt_metric], self.eval_opt_metric)   
            else:
                if metrics[self.config.opt_metric] > self.eval_opt_metric:
                    self.best_metrics = metrics
                    self.save(self.current_epoch, total_loss, metrics[self.config.opt_metric])
                    bestval_dict={f'best_val_{key}': val for key, val in self.best_metrics.items()}
                    bestval_dict["epoch"]=self.current_epoch
                    wandb.log(bestval_dict)  
                self.min_eval_loss = min(self.min_eval_loss, total_loss)
                self.eval_opt_metric = max(metrics[self.config.opt_metric], self.eval_opt_metric)                
                                        
            logfun("==" * 25)
            logfun(
                f"[{self.config.opt_metric} :%f, Loss: %f] [Min Loss :%f, Max {self.config.opt_metric}: %f]" %
                (metrics[self.config.opt_metric],
                 total_loss,
                 self.min_eval_loss,
                 self.eval_opt_metric,
                 )
                )            
            logfun("==" * 25)
        self.validation_step_outputs.clear()  # free memory
        
    def save(self, epoch, loss, acc):
        
        self.config.logfun(self.last_saved)
        output_path = os.path.join(self.config.store_dir, f"best_{self.config.dataset}.pt") 
        self.last_saved = output_path
        torch.save(self.model.state_dict(), output_path)
        self.config.logfun("EP:%d Model Saved on:" % epoch, output_path)
        with open(f"{self.config.store_dir}/best_metrics_{self.config.dataset}.json", 'w') as json_file:
            json.dump(self.best_metrics, json_file)
        return output_path
                      
    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(
                            self.parameters(),
                            lr = self.config.lr,
                            betas = (0.9, 0.999),
                            weight_decay = self.config.weight_decay,
            )
        
        return optimizer                      
                      
                        
                        
                        
        
     
