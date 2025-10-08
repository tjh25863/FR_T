"""
花卉识别AI挑战赛 - 训练脚本 (优化版)
AI Competition Coach - Flower Recognition Challenge Training v1.1.0

优化内容:
1. 🏷️ 添加Label Smoothing支持
2. 📈 CosineAnnealingWarmRestarts学习率调度  
3. 🔄 渐进式解冻训练策略
4. 🎲 CutMix和MixUp数据增强
5. 🌡️ 温度缩放优化
6. 📊 增强的指标监控和早停
7. 🛡️ 更强的错误处理和恢复

使用方法:
python train.py --config config.json --epochs 10
"""

import os
import argparse
import time
import math
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from model import FlowerViTModel, ModelFactory, ModelUtils
from utils import (
    Logger, ConfigLoader, set_seed, create_directories,
    prepare_data_loaders, MetricsCalculator, save_model_checkpoint,
    format_time, LabelSmoothingCrossEntropy, CutMixUp
)


class OptimizedTrainer:
    """优化的训练器类"""

    def __init__(self, config: Dict):
        """
        初始化训练器

        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = Logger(config['paths']['logs_dir'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.best_score = 0.0
        self.patience_counter = 0
        self.epoch_losses = {'train': [], 'val': []}
        self.epoch_metrics = {'train': [], 'val': []}

        # 创建必要目录
        create_directories(config)

        # 设置随机种子
        if config.get('reproducibility', {}).get('deterministic', True):
            set_seed(config['reproducibility']['seed'])

        self.logger.info("🚀 优化训练器初始化完成")

    def setup_model(self) -> nn.Module:
        """设置模型"""
        model = ModelFactory.create_vit_model(self.config)
        model = model.to(self.device)

        # 打印模型信息
        ModelUtils.print_model_info(model)

        return model

    def setup_criterion(self) -> nn.Module:
        """设置损失函数"""
        label_smoothing = self.config['training'].get('label_smoothing', 0.0)

        if label_smoothing > 0:
            criterion = LabelSmoothingCrossEntropy(
                num_classes=self.config['model']['num_classes'],
                smoothing=label_smoothing
            )
            self.logger.info(f"📊 使用Label Smoothing损失函数 (smoothing={label_smoothing})")
        else:
            criterion = nn.CrossEntropyLoss()
            self.logger.info("📊 使用标准交叉熵损失函数")

        return criterion.to(self.device)

    def setup_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """设置优化器"""
        optimizer_config = self.config['optimizer']
        training_config = self.config['training']

        # 分组参数：backbone vs 分类头
        backbone_params = []
        classifier_params = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'classifier' in name:
                    classifier_params.append(param)
                else:
                    backbone_params.append(param)

        # 为backbone和分类头设置不同学习率
        param_groups = []
        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr': training_config['learning_rate'] * 0.1,  # backbone用更小学习率
                'weight_decay': training_config['weight_decay']
            })
        if classifier_params:
            param_groups.append({
                'params': classifier_params, 
                'lr': training_config['learning_rate'],  # 分类头用标准学习率
                'weight_decay': training_config['weight_decay'] * 0.1
            })

        optimizer = optim.AdamW(
            param_groups,
            betas=optimizer_config['betas'],
            eps=optimizer_config['eps'],
            amsgrad=optimizer_config.get('amsgrad', False)
        )

        self.logger.info(f"⚙️ 优化器设置完成")
        self.logger.info(f"   Backbone学习率: {training_config['learning_rate'] * 0.1}")
        self.logger.info(f"   分类头学习率: {training_config['learning_rate']}")

        return optimizer

    def setup_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
        """设置学习率调度器"""
        scheduler_config = self.config['scheduler']

        if scheduler_config['type'] == 'cosine_annealing_warm_restarts':
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=scheduler_config['T_0'],
                T_mult=scheduler_config.get('T_mult', 1),
                eta_min=scheduler_config['eta_min']
            )
            self.logger.info("📈 使用CosineAnnealingWarmRestarts调度器")
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=self.config['training'].get('lr_scheduler_patience', 5),
                verbose=True
            )
            self.logger.info("📈 使用ReduceLROnPlateau调度器")

        return scheduler

    def train_epoch(self, model: nn.Module, train_loader, criterion: nn.Module, 
                   optimizer: torch.optim.Optimizer, scaler: torch.cuda.amp.GradScaler,
                   epoch: int, cutmix_mixup: Optional[CutMixUp] = None) -> Tuple[float, float]:
        """训练一个epoch"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # 渐进式解冻策略
        if epoch >= 3 and not self.config['model'].get('freeze_backbone', False):
            if epoch == 3:
                model.unfreeze_last_layers(2)  # 解冻最后2层
                self.logger.info("🔓 第3轮：解冻最后2层")
            elif epoch == 6:
                model.freeze_backbone(False)  # 全部解冻
                self.logger.info("🔓 第6轮：全部解冻")

        progress_bar = tqdm(train_loader, desc=f'训练 Epoch {epoch}', 
                           leave=False, ncols=100)

        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(self.device), labels.to(self.device)

            labels_a = labels
            labels_b = labels
            lam = 1.0
            # 应用CutMix/MixUp
            if cutmix_mixup is not None and torch.rand(1) < 0.5:
                images, labels_a, labels_b, lam = cutmix_mixup(images, labels)

            optimizer.zero_grad()

            # 混合精度训练
            with torch.cuda.amp.autocast(enabled=self.config['training']['use_amp']):
                outputs = model(images)

                if cutmix_mixup is not None and torch.rand(1) < 0.5:
                    # CutMix损失
                    loss = criterion(outputs, labels_a) * lam + criterion(outputs, labels_b) * (1 - lam)
                else:
                    loss = criterion(outputs, labels)

            # 反向传播
            scaler.scale(loss).backward()

            # 梯度裁剪
            if self.config['training'].get('max_grad_norm', 0) > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 
                                       self.config['training']['max_grad_norm'])

            scaler.step(optimizer)
            scaler.update()

            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 更新进度条
            current_lr = ModelUtils.get_learning_rates(optimizer)[0]
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%',
                'LR': f'{current_lr:.2e}'
            })

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total

        return epoch_loss, epoch_acc

    def validate_epoch(self, model: nn.Module, val_loader, criterion: nn.Module,
                      metrics_calculator: MetricsCalculator, epoch: int) -> Dict[str, float]:
        """验证一个epoch"""
        model.eval()
        running_loss = 0.0
        metrics_calculator.reset()

        progress_bar = tqdm(val_loader, desc=f'验证 Epoch {epoch}', 
                           leave=False, ncols=100)

        with torch.no_grad():
            for images, labels in progress_bar:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item()

                # 更新指标
                probs = F.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                metrics_calculator.update(predicted, labels, probs)

                progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        # 计算指标
        epoch_loss = running_loss / len(val_loader)
        metrics = metrics_calculator.compute_metrics()

        results = {
            'loss': epoch_loss,
            **metrics
        }

        return results

    def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                       scheduler, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """保存检查点"""
        checkpoint_info = {
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'best_score': self.best_score
        }

        save_path = Path(self.config['paths']['checkpoints_dir']) / f'checkpoint_epoch_{epoch}.pth'
        ModelUtils.save_model_weights(model, str(save_path), checkpoint_info)

        if is_best:
            best_path = Path(self.config['paths']['model_dir']) / 'best_model.pth'
            ModelUtils.save_model_weights(model, str(best_path), checkpoint_info)
            self.logger.info(f"💾 最佳模型已保存: {best_path}")

    def train(self):
        """主训练循环"""
        self.logger.info("🎯 开始优化训练流程")

        # 设置模型和训练组件
        model = self.setup_model()
        criterion = self.setup_criterion()
        optimizer = self.setup_optimizer(model)
        scheduler = self.setup_scheduler(optimizer)
        scaler = torch.cuda.amp.GradScaler(enabled=self.config['training']['use_amp'])

        # 数据加载器
        train_loader, val_loader, class_to_idx, idx_to_class = prepare_data_loaders(
            self.config, self.logger)

        # CutMix/MixUp
        cutmix_mixup = None
        if self.config['augmentation'].get('cutmix', {}).get('prob', 0) > 0:
            cutmix_mixup = CutMixUp(
                cutmix_alpha=self.config['augmentation']['cutmix']['alpha'],
                mixup_alpha=self.config['augmentation']['mixup']['alpha'],
                prob=0.5
            )
            self.logger.info("🎲 CutMix/MixUp数据增强已启用")

        # 指标计算器
        metrics_calculator = MetricsCalculator(self.config['model']['num_classes'])

        # TensorBoard
        writer = SummaryWriter(self.config['paths']['logs_dir'])

        # 训练循环
        start_time = time.time()
        epochs = self.config['training']['epochs']
        patience = self.config['training']['patience']

        for epoch in range(epochs):
            epoch_start = time.time()

            # 训练
            train_loss, train_acc = self.train_epoch(
                model, train_loader, criterion, optimizer, scaler, epoch, cutmix_mixup)

            # 验证
            val_results = self.validate_epoch(
                model, val_loader, criterion, metrics_calculator, epoch)

            # 学习率调度
            if isinstance(scheduler, CosineAnnealingWarmRestarts):
                scheduler.step()
            else:
                scheduler.step(val_results['competition_score'])

            # 记录指标
            self.epoch_losses['train'].append(train_loss)
            self.epoch_losses['val'].append(val_results['loss'])

            # 日志记录
            epoch_time = time.time() - epoch_start
            self.logger.info(f"Epoch {epoch+1:2d}/{epochs}")
            self.logger.info(f"  训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            self.logger.info(f"  验证 - Loss: {val_results['loss']:.4f}, Acc: {val_results['accuracy']:.4f}")
            self.logger.info(f"  验证 - Top5: {val_results['top5_accuracy']:.4f}, F1: {val_results['f1_macro']:.4f}")
            self.logger.info(f"  竞赛得分: {val_results['competition_score']:.4f}")
            self.logger.info(f"  用时: {format_time(epoch_time)}")

            # TensorBoard记录
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Val', val_results['loss'], epoch)
            writer.add_scalar('Accuracy/Train', train_acc, epoch)
            writer.add_scalar('Accuracy/Val', val_results['accuracy'], epoch)
            writer.add_scalar('Top5Accuracy/Val', val_results['top5_accuracy'], epoch)
            writer.add_scalar('F1Macro/Val', val_results['f1_macro'], epoch)
            writer.add_scalar('CompetitionScore/Val', val_results['competition_score'], epoch)
            writer.add_scalar('LearningRate', ModelUtils.get_learning_rates(optimizer)[0], epoch)

            # 保存最佳模型
            current_score = val_results['competition_score']
            is_best = current_score > self.best_score

            if is_best:
                self.best_score = current_score
                self.patience_counter = 0
                self.save_checkpoint(model, optimizer, scheduler, epoch, val_results, is_best=True)
            else:
                self.patience_counter += 1

            # 早停检查
            if self.patience_counter >= patience:
                self.logger.info(f"⏹️  早停触发，已等待{patience}轮无改善")
                break

        # 训练完成
        total_time = time.time() - start_time
        self.logger.info(f"🎉 训练完成！")
        self.logger.info(f"📊 最佳竞赛得分: {self.best_score:.4f}")
        self.logger.info(f"⏱️  总训练时间: {format_time(total_time)}")

        writer.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='花卉识别训练脚本 - 优化版')
    parser.add_argument('--config', type=str, default='config.json',
                       help='配置文件路径')
    parser.add_argument('--epochs', type=int, help='训练轮数 (覆盖配置)')
    parser.add_argument('--batch_size', type=int, help='批次大小 (覆盖配置)')
    parser.add_argument('--lr', type=float, help='学习率 (覆盖配置)')

    args = parser.parse_args()

    # 加载配置
    config = ConfigLoader.load_config(args.config)

    # 命令行参数覆盖
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr

    # 开始训练
    trainer = OptimizedTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
