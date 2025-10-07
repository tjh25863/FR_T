"""
花卉识别AI挑战赛 - 模型定义文件 (优化版)
AI Competition Coach - Flower Recognition Challenge Model v1.1.0

优化内容:
1. 🌡️ 添加温度缩放提升置信度校准
2. 📊 支持Label Smoothing降低Loss  
3. 🎯 改进模型初始化和权重加载
4. 🔄 增强模型保存和加载功能
5. 📈 添加模型性能监控
6. 🛡️ 增强错误处理和日志记录
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel, AutoConfig, ViTModel, ViTConfig, 
    ViTForImageClassification, AutoImageProcessor
)
from typing import Dict, Optional, Tuple, List
import warnings
import numpy as np
from pathlib import Path
import logging

warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", message=".*weights.*not initialized.*")
warnings.filterwarnings("ignore", message=".*mismatched sizes.*")

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlowerViTModel(nn.Module):
    """
    基于Vision Transformer的花卉识别模型 - 优化版

    主要改进:
    - 温度缩放改善置信度校准  
    - 更好的权重初始化策略
    - 支持渐进式解冻训练
    - 改进的Dropout策略
    """

    def __init__(self, 
                 model_name: str = "google/vit-base-patch16-224",
                 pretrained_path: str = "loretyan/vit-base-oxford-flowers-102", 
                 num_classes: int = 100,
                 dropout_rate: float = 0.1,
                 use_pretrained: bool = True,
                 freeze_backbone: bool = False,
                 temperature: float = 1.0,
                 label_smoothing: float = 0.0):
        """
        初始化Vision Transformer模型

        Args:
            model_name: 基础模型名称
            pretrained_path: 预训练模型路径
            num_classes: 分类类别数
            dropout_rate: Dropout概率
            use_pretrained: 是否使用预训练权重
            freeze_backbone: 是否冻结backbone
            temperature: 温度缩放参数
            label_smoothing: 标签平滑参数
        """
        super(FlowerViTModel, self).__init__()

        self.num_classes = num_classes
        self.model_name = model_name  
        self.pretrained_path = pretrained_path
        self.temperature = temperature
        self.label_smoothing = label_smoothing

        logger.info(f"🚀 初始化优化版Vision Transformer模型")
        logger.info(f"   基础模型: {model_name}")
        logger.info(f"   预训练路径: {pretrained_path}")
        logger.info(f"   分类数量: {num_classes}")
        logger.info(f"   温度参数: {temperature}")
        logger.info(f"   标签平滑: {label_smoothing}")

        # 初始化模型组件
        self.vit = None
        self.config = None
        self._init_model(use_pretrained)

        # 构建改进的分类头
        self._build_enhanced_classifier(dropout_rate)

        # 应用冻结策略
        if freeze_backbone:
            self.freeze_backbone(True)

        # 改进权重初始化
        self._init_weights()

        logger.info(f"✅ 优化版模型初始化完成")

    def _init_model(self, use_pretrained: bool):
        """初始化ViT模型"""
        try:
            if use_pretrained and self.pretrained_path:
                logger.info(f"📥 加载预训练模型: {self.pretrained_path}")

                # 加载完整的预训练模型
                self.vit = ViTForImageClassification.from_pretrained(
                    self.pretrained_path,
                    num_labels=self.num_classes,
                    ignore_mismatched_sizes=True,
                    output_hidden_states=True,
                    output_attentions=False
                )

                self.config = self.vit.config
                logger.info(f"✅ 成功加载预训练模型")

            else:
                # 创建基础模型
                logger.info(f"📦 创建基础模型: {self.model_name}")
                self.config = ViTConfig.from_pretrained(self.model_name)
                self.vit = ViTForImageClassification.from_pretrained(
                    self.model_name,
                    config=self.config,
                    num_labels=self.num_classes
                )
                logger.info(f"✅ 基础模型创建完成")

        except Exception as e:
            logger.error(f"❌ 模型初始化失败: {e}")
            # 创建默认模型作为后备
            self._create_default_model()

    def _create_default_model(self):
        """创建默认模型"""
        logger.warning("⚠️  创建默认ViT模型")
        self.config = ViTConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            image_size=224,
            patch_size=16,
            num_channels=3,
            num_labels=self.num_classes
        )
        self.vit = ViTForImageClassification(self.config)

    def _build_enhanced_classifier(self, dropout_rate: float):
        """构建增强的分类头"""
        hidden_size = self.config.hidden_size

        # 替换原有分类头为更复杂的结构
        self.enhanced_classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.GELU(), 
            nn.BatchNorm1d(hidden_size // 4),
            nn.Dropout(dropout_rate / 4),
            nn.Linear(hidden_size // 4, self.num_classes)
        )

        # 保持原有简单分类头用于对比
        self.simple_classifier = nn.Linear(hidden_size, self.num_classes)

        logger.info(f"📊 增强分类头构建完成")
        logger.info(f"   隐藏层维度: {hidden_size} → {hidden_size//2} → {hidden_size//4} → {self.num_classes}")

    def _init_weights(self):
        """改进的权重初始化"""
        def _init_layer(layer):
            if isinstance(layer, nn.Linear):
                # Xavier初始化
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

        # 初始化增强分类头
        self.enhanced_classifier.apply(_init_layer)
        self.simple_classifier.apply(_init_layer)

        logger.info("🎯 权重初始化完成")

    def forward(self, pixel_values: torch.Tensor, use_enhanced: bool = True):
        """
        前向传播

        Args:
            pixel_values: 输入图像张量
            use_enhanced: 是否使用增强分类头

        Returns:
            logits: 分类logits (应用温度缩放)
        """
        # 获取ViT特征
        outputs = self.vit(pixel_values=pixel_values, output_hidden_states=True)

        # 提取CLS token特征
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            # 使用最后一层的CLS token
            cls_features = outputs.hidden_states[-1][:, 0]  # [batch_size, hidden_size]
        else:
            # 如果没有hidden_states，使用pooler_output
            cls_features = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.logits

        # 选择分类头
        if use_enhanced:
            logits = self.enhanced_classifier(cls_features)
        else:
            logits = self.simple_classifier(cls_features)

        # 应用温度缩放
        if self.temperature != 1.0:
            logits = logits / self.temperature

        return logits

    def get_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        获取特征向量

        Args:
            pixel_values: 输入图像张量

        Returns:
            features: CLS token特征向量
        """
        with torch.no_grad():
            outputs = self.vit(pixel_values=pixel_values, output_hidden_states=True)
            features = outputs.hidden_states[-1][:, 0]  # CLS token
        return features


    def freeze_backbone(self, freeze: bool = True):
        """
        冻结/解冻backbone参数

        Args:
            freeze: 是否冻结
        """
        for param in self.vit.vit.parameters():
            param.requires_grad = not freeze

        # 确保分类头参数可训练
        for param in self.enhanced_classifier.parameters():
            param.requires_grad = True
        for param in self.simple_classifier.parameters():
            param.requires_grad = True

        status = "冻结" if freeze else "解冻"
        logger.info(f"🔒 Backbone已{status}")

    def unfreeze_last_layers(self, num_layers: int = 2):
        """
        解冻最后几层进行精细调优

        Args:
            num_layers: 解冻的层数
        """
        # 解冻最后几个transformer层
        total_layers = len(self.vit.vit.encoder.layer)
        start_layer = max(0, total_layers - num_layers)

        for i in range(start_layer, total_layers):
            for param in self.vit.vit.encoder.layer[i].parameters():
                param.requires_grad = True

        logger.info(f"🔓 解冻最后{num_layers}层 (层{start_layer}-{total_layers-1})")

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total_params': total_params,
            'trainable_params': trainable_params, 
            'model_size_mb': total_params * 4 / 1024 / 1024,
            'trainable_ratio': trainable_params / total_params,
            'temperature': self.temperature,
            'num_classes': self.num_classes
        }

    def set_temperature(self, temperature: float):
        """设置温度参数"""
        self.temperature = temperature
        logger.info(f"🌡️  温度参数更新为: {temperature}")

    def get_attention_weights(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        获取注意力权重用于可视化

        Args:
            pixel_values: 输入图像张量

        Returns:
            attention_weights: 注意力权重
        """
        with torch.no_grad():
            outputs = self.vit(pixel_values=pixel_values, output_attentions=True)
            # 返回最后一层的注意力权重
            return outputs.attentions[-1] if outputs.attentions else None


class ModelFactory:
    """模型工厂类 - 优化版"""

    @staticmethod
    def create_vit_model(config: Dict) -> FlowerViTModel:
        """
        创建ViT模型

        Args:
            config: 配置字典

        Returns:
            FlowerViTModel实例
        """
        model_config = config['model']
        training_config = config.get('training', {})

        model = FlowerViTModel(
            model_name=model_config.get('name', 'google/vit-base-patch16-224'),
            pretrained_path=model_config.get('pretrained_model', 'loretyan/vit-base-oxford-flowers-102'),
            num_classes=model_config.get('num_classes', 100),
            dropout_rate=model_config.get('dropout', 0.1),
            use_pretrained=True,
            freeze_backbone=model_config.get('freeze_backbone', False),
            temperature=model_config.get('temperature', 1.0),
            label_smoothing=training_config.get('label_smoothing', 0.0)
        )

        return model

    @staticmethod
    def create_ensemble_models(config: Dict, num_models: int = 3, 
                             different_seeds: bool = True) -> List[FlowerViTModel]:
        """
        创建集成模型

        Args:
            config: 配置字典
            num_models: 模型数量
            different_seeds: 是否使用不同随机种子

        Returns:
            模型列表
        """
        models = []
        base_seed = config.get('reproducibility', {}).get('seed', 42)

        for i in range(num_models):
            if different_seeds:
                # 设置不同的随机种子
                torch.manual_seed(base_seed + i)

            model = ModelFactory.create_vit_model(config)
            models.append(model)

            logger.info(f"🎲 集成模型 {i+1}/{num_models} 创建完成")

        return models


class ModelUtils:
    """模型工具类 - 优化版"""

    @staticmethod
    def print_model_info(model: nn.Module):
        """打印模型信息"""
        if hasattr(model, 'get_model_info'):
            info = model.get_model_info()
            logger.info("📊 模型信息:")
            logger.info(f"   总参数量: {info['total_params']:,}")
            logger.info(f"   可训练参数: {info['trainable_params']:,}")
            logger.info(f"   模型大小: {info['model_size_mb']:.2f} MB")
            logger.info(f"   可训练比例: {info['trainable_ratio']:.2%}")
            logger.info(f"   温度参数: {info['temperature']}")
            logger.info(f"   分类类别: {info['num_classes']}")

    @staticmethod
    def save_model_weights(model: nn.Module, save_path: str, 
                          additional_info: Optional[Dict] = None,
                          save_config: bool = True):
        """
        保存模型权重 - 增强版

        Args:
            model: 模型实例
            save_path: 保存路径
            additional_info: 额外信息
            save_config: 是否保存配置
        """
        save_dict = {
            'model_state_dict': model.state_dict(),
            'model_info': model.get_model_info() if hasattr(model, 'get_model_info') else {},
            'save_time': torch.tensor([torch.now()]) if hasattr(torch, 'now') else None
        }

        if additional_info:
            save_dict.update(additional_info)

        if save_config and hasattr(model, 'config'):
            save_dict['model_config'] = model.config

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(save_dict, save_path)
        logger.info(f"✅ 模型权重已保存: {save_path}")

    @staticmethod
    def load_model_weights(model: nn.Module, checkpoint_path: str,
                          strict: bool = False) -> Dict:
        """
        加载模型权重 - 增强版

        Args:
            model: 模型实例
            checkpoint_path: 检查点路径
            strict: 是否严格匹配权重

        Returns:
            加载信息字典
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            # 加载权重
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)

            load_info = {
                'missing_keys': missing_keys,
                'unexpected_keys': unexpected_keys,
                'checkpoint_info': {k: v for k, v in checkpoint.items() if k != 'model_state_dict'}
            }

            logger.info(f"✅ 模型权重加载完成: {checkpoint_path}")
            if missing_keys:
                logger.warning(f"⚠️  缺失权重: {len(missing_keys)} 个")
            if unexpected_keys:
                logger.warning(f"⚠️  多余权重: {len(unexpected_keys)} 个")

            return load_info

        except Exception as e:
            logger.error(f"❌ 权重加载失败: {e}")
            return {'error': str(e)}

    @staticmethod
    def count_parameters(model: nn.Module) -> Tuple[int, int]:
        """
        统计参数数量

        Returns:
            total_params, trainable_params
        """
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable

    @staticmethod
    def get_learning_rates(optimizer: torch.optim.Optimizer) -> List[float]:
        """获取当前学习率"""
        return [group['lr'] for group in optimizer.param_groups]


class EnsembleModel(nn.Module):
    """集成模型类"""

    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        """
        初始化集成模型

        Args:
            models: 模型列表
            weights: 集成权重
        """
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        self.weights = weights or [1.0] * len(models)
        self.weights = torch.tensor(self.weights)

        logger.info(f"🎯 集成模型初始化完成，包含 {len(models)} 个子模型")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播集成预测"""
        outputs = []

        for model in self.models:
            with torch.no_grad():
                output = model(x)
                outputs.append(F.softmax(output, dim=1))

        # 加权平均
        ensemble_output = torch.zeros_like(outputs[0])
        total_weight = 0

        for i, output in enumerate(outputs):
            weight = self.weights[i]
            ensemble_output += weight * output
            total_weight += weight

        ensemble_output /= total_weight
        return torch.log(ensemble_output)  # 返回log概率用于损失计算

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """预测方法"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)


# 导出主要类和函数
__all__ = [
    'FlowerViTModel',
    'ModelFactory', 
    'ModelUtils',
    'EnsembleModel'
]
