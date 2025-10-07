"""
花卉识别AI挑战赛 - 工具函数库 (修复版)
AI Competition Coach - Flower Recognition Challenge Utils v1.2.0

🔧 重要修复:
1. RandomErasing移动到ToTensor之后
2. 修复导入错误
3. 修复数据变换顺序问题
4. 增强错误处理
5. 优化内存使用
"""

import os
import json
import random
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter

import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing交叉熵损失函数 - 修复版

    有效降低模型过拟合，提升泛化能力，同时显著降低Loss值
    """

    def __init__(self, num_classes: int, smoothing: float = 0.1):
        """
        Args:
            num_classes: 类别数量
            smoothing: 平滑参数 (0.0-1.0)
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测logits [batch_size, num_classes]
            target: 真实标签 [batch_size]

        Returns:
            smoothed_loss: 平滑后的损失
        """
        pred = F.log_softmax(pred, dim=-1)

        # 创建平滑标签
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist += self.smoothing / self.num_classes

        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


class CutMixUp:
    """
    CutMix和MixUp数据增强实现 - 修复版

    结合CutMix和MixUp两种技术，提升模型泛化能力
    """

    def __init__(self, cutmix_alpha: float = 1.0, mixup_alpha: float = 0.2, prob: float = 0.5):
        """
        Args:
            cutmix_alpha: CutMix的alpha参数
            mixup_alpha: MixUp的alpha参数  
            prob: 使用增强的概率
        """
        self.cutmix_alpha = cutmix_alpha
        self.mixup_alpha = mixup_alpha
        self.prob = prob

    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Args:
            images: 输入图像 [batch_size, channels, height, width]
            labels: 标签 [batch_size]

        Returns:
            mixed_images, labels_a, labels_b, lam
        """
        if random.random() > self.prob:
            return images, labels, labels, 1.0

        batch_size = images.size(0)

        # 随机选择CutMix或MixUp
        if random.random() < 0.5:
            return self._cutmix(images, labels, batch_size)
        else:
            return self._mixup(images, labels, batch_size)

    def _cutmix(self, images: torch.Tensor, labels: torch.Tensor, batch_size: int):
        """CutMix增强"""
        indices = torch.randperm(batch_size)
        shuffled_labels = labels[indices]

        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)

        # 计算裁剪区域
        _, _, H, W = images.shape
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # 随机选择裁剪位置
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        # 应用裁剪
        mixed_images = images.clone()
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[indices, :, bby1:bby2, bbx1:bbx2]

        # 调整lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

        return mixed_images, labels, shuffled_labels, lam

    def _mixup(self, images: torch.Tensor, labels: torch.Tensor, batch_size: int):
        """MixUp增强"""
        indices = torch.randperm(batch_size)
        shuffled_labels = labels[indices]

        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)

        mixed_images = lam * images + (1 - lam) * images[indices]

        return mixed_images, labels, shuffled_labels, lam


def set_seed_strict(seed: int = 42):
    """
    更严格的随机种子设置 - 修复版

    Args:
        seed: 随机种子
    """
    import os

    # 设置所有随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 更严格的确定性设置
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

    # 设置环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    # PyTorch 1.8+ 的确定性设置
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except:
        pass

    logger.info(f"🎯 严格随机种子已设置: {seed}")


def set_seed(seed: int = 42):
    """标准随机种子设置（保持向后兼容）"""
    set_seed_strict(seed)


class Logger:
    """增强的日志记录器 - 修复版"""
    def __init__(self, log_dir: str = './logs', log_level: str = 'INFO'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 设置日志格式
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'

        # 配置日志
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            datefmt=date_format,
            handlers=[
                logging.FileHandler(self.log_dir / 'training.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger('FlowerRecognition')
        self.metrics_history = []

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)

    def debug(self, message: str):
        self.logger.debug(message)

    def log_metrics(self, epoch: int, metrics: Dict[str, float]):
        """记录训练指标"""
        metrics_with_epoch = {'epoch': epoch, **metrics}
        self.metrics_history.append(metrics_with_epoch)

        # 保存指标历史
        metrics_file = self.log_dir / 'metrics_history.json'
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_history, f, indent=2, ensure_ascii=False)


class ConfigLoader:
    """配置文件加载器 - 修复版"""

    @staticmethod
    def load_config(config_path: str) -> Dict:
        """加载并验证配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # 基础验证
            ConfigLoader._validate_config(config)

            logger.info(f"✅ 配置文件加载成功: {config_path}")
            return config
        except Exception as e:
            logger.error(f"❌ 配置文件加载失败: {e}")
            raise ValueError(f"配置文件加载失败: {e}")

    @staticmethod
    def _validate_config(config: Dict):
        """验证配置文件"""
        required_sections = ['model', 'training', 'data', 'paths']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"配置文件缺少必需部分: {section}")

    @staticmethod
    def save_config(config: Dict, save_path: str):
        """保存配置文件"""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ 配置文件已保存: {save_path}")


def create_directories(config: Dict):
    """创建必要的目录"""
    dirs_to_create = [
        config['paths']['model_dir'],
        config['paths']['results_dir'],
        config['paths']['logs_dir'],
        config['paths']['checkpoints_dir']
    ]

    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    logger.info("📁 必要目录已创建")


def worker_init_fn(worker_id):
    """DataLoader worker初始化函数"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



class FlowerDataset(Dataset):
    """花卉数据集 - 修复版"""

    def __init__(self, 
                 image_paths: List[str], 
                 labels: Optional[List[int]] = None, 
                 transform: Optional[transforms.Compose] = None, 
                 class_to_idx: Optional[Dict] = None,
                 cache_images: bool = False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.cache_images = cache_images
        self._image_cache = {} if cache_images else None

        if cache_images:
            logger.info("🗂️ 启用图像缓存模式")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 获取图像
        image_path = self.image_paths[idx]

        # 尝试从缓存获取
        if self.cache_images and idx in self._image_cache:
            image = self._image_cache[idx].copy()
        else:
            try:
                image = Image.open(image_path).convert('RGB')
                if self.cache_images:
                    self._image_cache[idx] = image.copy()
            except Exception as e:
                logger.warning(f"图像加载失败: {image_path}, 使用黑色图像替代")
                image = Image.new('RGB', (224, 224), (0, 0, 0))

        # 应用数据变换
        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                logger.error(f"数据变换失败: {image_path}, 错误: {e}")
                # 使用基础变换作为后备
                basic_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                image = basic_transform(image)

        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        else:
            # 推理时返回图像ID
            image_id = Path(image_path).stem
            return image, image_id

    def get_class_distribution(self) -> Dict[int, int]:
        """获取类别分布"""
        if self.labels is None:
            return {}

        return dict(Counter(self.labels))


def load_class_mappings(mapping_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    加载类别映射 - 修复版

    Args:
        mapping_path: 映射文件路径

    Returns:
        class_to_idx, idx_to_class
    """
    try:
        with open(mapping_path, 'r', encoding='utf-8') as f:
            class_names = json.load(f)

        # 处理不同格式的映射文件
        if isinstance(class_names, dict):
            # 字典格式 {"1": "类别名"}
            idx_to_class = {int(k): v for k, v in class_names.items()}
            class_to_idx = {v: int(k) for k, v in class_names.items()}
        else:
            # 列表格式
            idx_to_class = {i: name for i, name in enumerate(class_names)}
            class_to_idx = {name: i for i, name in enumerate(class_names)}

        logger.info(f"✅ 类别映射加载成功: {len(class_to_idx)} 个类别")
        return class_to_idx, idx_to_class
    except Exception as e:
        logger.error(f"❌ 类别映射加载失败: {e}")
        # 创建默认映射
        idx_to_class = {i: f'class_{i+1}' for i in range(100)}
        class_to_idx = {f'class_{i+1}': i for i in range(100)}
        return class_to_idx, idx_to_class


def get_train_transforms_fixed(config: Dict) -> transforms.Compose:
    """
    🔧 修复版：获取训练数据变换（修复RandomErasing位置）

    主要修复：
    1. 将RandomErasing移动到ToTensor()之后
    2. 确保所有变换的正确顺序
    3. 更保守的增强参数

    Args:
        config: 配置字典
    Returns:
        数据变换序列
    """
    aug_config = config['augmentation']

    # Step 1: PIL图像变换（在ToTensor之前）
    pil_transforms = [
        transforms.Resize(tuple(aug_config['resize'])),

        transforms.RandomResizedCrop(
            aug_config['resize'][0], 
            scale=tuple(aug_config['random_crop']['scale']),    # [0.90, 1.0]
            ratio=tuple(aug_config['random_crop']['ratio'])     # [0.85, 1.15]
        ),

        transforms.RandomHorizontalFlip(aug_config['horizontal_flip']),  # 0.3
        transforms.RandomRotation(aug_config['rotation']),  # 8度

        transforms.ColorJitter(
            brightness=aug_config['color_jitter']['brightness'],  # 0.15
            contrast=aug_config['color_jitter']['contrast'],      # 0.15
            saturation=aug_config['color_jitter']['saturation'],  # 0.15
            hue=aug_config['color_jitter']['hue']                 # 0.05
        ),
    ]

    # Step 2: 转换为张量
    tensor_conversion = [
        transforms.ToTensor(),
    ]

    # Step 3: 张量变换（在ToTensor之后）
    tensor_transforms = [
        # ✅ RandomErasing现在在正确位置（ToTensor之后）
        transforms.RandomErasing(
            p=0.1,               # 10%概率应用
            scale=(0.02, 0.08),  # 更小的擦除区域
            ratio=(0.3, 3.3),    # 擦除区域宽高比
            value=0,             # 擦除值
            inplace=False        # 不原地操作
        ),

        transforms.Normalize(
            mean=aug_config['normalize']['mean'],
            std=aug_config['normalize']['std']
        )
    ]

    # 合并所有变换
    all_transforms = pil_transforms + tensor_conversion + tensor_transforms

    return transforms.Compose(all_transforms)


def get_train_transforms_safe(config: Dict) -> transforms.Compose:
    """
    🛡️ 安全版：获取训练数据变换（移除RandomErasing）

    Args:
        config: 配置字典
    Returns:
        数据变换序列
    """
    aug_config = config['augmentation']

    transform_list = [
        # PIL变换
        transforms.Resize(tuple(aug_config['resize'])),
        transforms.RandomResizedCrop(
            aug_config['resize'][0], 
            scale=tuple(aug_config['random_crop']['scale']),
            ratio=tuple(aug_config['random_crop']['ratio'])
        ),
        transforms.RandomHorizontalFlip(aug_config['horizontal_flip']),
        transforms.RandomRotation(aug_config['rotation']),
        transforms.ColorJitter(
            brightness=aug_config['color_jitter']['brightness'],
            contrast=aug_config['color_jitter']['contrast'],
            saturation=aug_config['color_jitter']['saturation'],
            hue=aug_config['color_jitter']['hue']
        ),

        # 转换和归一化
        transforms.ToTensor(),
        transforms.Normalize(
            mean=aug_config['normalize']['mean'],
            std=aug_config['normalize']['std']
        )
    ]

    return transforms.Compose(transform_list)


def get_val_transforms(config: Dict) -> transforms.Compose:
    """获取验证数据变换"""
    aug_config = config['augmentation']

    transform_list = [
        transforms.Resize(tuple(aug_config['resize'])),
        transforms.CenterCrop(aug_config['resize'][0]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=aug_config['normalize']['mean'],
            std=aug_config['normalize']['std']
        )
    ]

    return transforms.Compose(transform_list)


def get_test_transforms(config: Dict) -> transforms.Compose:
    """获取测试数据变换"""
    return get_val_transforms(config)


def get_tta_transforms(config: Dict, num_augmentations: int = 5) -> List[transforms.Compose]:
    """
    获取TTA (Test Time Augmentation) 变换列表

    Args:
        config: 配置字典
        num_augmentations: 增强数量

    Returns:
        变换列表
    """
    aug_config = config['augmentation']
    base_size = aug_config['resize'][0]

    tta_transforms = []

    # 1. 中心裁剪（原图）
    tta_transforms.append(transforms.Compose([
        transforms.Resize((base_size, base_size)),
        transforms.CenterCrop(base_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=aug_config['normalize']['mean'],
            std=aug_config['normalize']['std']
        )
    ]))

    # 2. 稍大尺寸 + 中心裁剪
    if len(tta_transforms) < num_augmentations:
        tta_transforms.append(transforms.Compose([
            transforms.Resize((int(base_size * 1.1), int(base_size * 1.1))),
            transforms.CenterCrop(base_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=aug_config['normalize']['mean'],
                std=aug_config['normalize']['std']
            )
        ]))

    # 3. 水平翻转
    if len(tta_transforms) < num_augmentations:
        tta_transforms.append(transforms.Compose([
            transforms.Resize((base_size, base_size)),
            transforms.CenterCrop(base_size),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=aug_config['normalize']['mean'],
                std=aug_config['normalize']['std']
            )
        ]))

    # 4. 多尺度测试1
    if len(tta_transforms) < num_augmentations:
        tta_transforms.append(transforms.Compose([
            transforms.Resize((int(base_size * 0.9), int(base_size * 0.9))),
            transforms.CenterCrop(base_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=aug_config['normalize']['mean'],
                std=aug_config['normalize']['std']
            )
        ]))

    # 5. 多尺度测试2
    if len(tta_transforms) < num_augmentations:
        tta_transforms.append(transforms.Compose([
            transforms.Resize((int(base_size * 1.15), int(base_size * 1.15))),
            transforms.CenterCrop(base_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=aug_config['normalize']['mean'],
                std=aug_config['normalize']['std']
            )
        ]))

    return tta_transforms[:num_augmentations]


def prepare_data_loaders(config: Dict, logger: Logger) -> Tuple[DataLoader, DataLoader, Dict, Dict]:
    """
    准备数据加载器 - 修复版

    Args:
        config: 配置字典
        logger: 日志记录器

    Returns:
        train_loader, val_loader, class_to_idx, idx_to_class
    """
    logger.info("🚀 开始准备修复版数据加载器...")

    # 加载类别映射
    class_to_idx, idx_to_class = load_class_mappings(config['data']['class_mapping'])

    # 收集训练数据
    train_dir = Path(config['data']['train_dir'])
    image_paths = []
    labels = []

    for class_dir in sorted(train_dir.iterdir()):
        if class_dir.is_dir():
            class_name = class_dir.name
            try:
                class_idx = int(class_name) - 1  # 1-100 -> 0-99
                if 0 <= class_idx < 100:
                    # 收集该类别的所有图像
                    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG'}
                    for ext in valid_extensions:
                        for img_path in class_dir.glob(f'*{ext}'):
                            if img_path.is_file():  # 确保是文件
                                image_paths.append(str(img_path))
                                labels.append(class_idx)
                else:
                    logger.warning(f"跳过无效类别: {class_name}")
            except ValueError:
                logger.warning(f"跳过非数字类别: {class_name}")

    logger.info(f"📊 总共找到 {len(image_paths)} 张训练图像")

    if len(image_paths) == 0:
        raise ValueError("没有找到有效的训练图像，请检查数据目录路径")

    # 分析数据分布
    label_counts = Counter(labels)
    logger.info(f"📈 类别分布统计:")
    logger.info(f"   最多类别: {max(label_counts.values())} 张图像")
    logger.info(f"   最少类别: {min(label_counts.values())} 张图像")
    logger.info(f"   平均每类: {len(image_paths) / len(label_counts):.1f} 张图像")

    # 分层划分训练验证集
    try:
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, 
            test_size=config['data']['val_split'],
            random_state=config.get('reproducibility', {}).get('seed', 42),
            stratify=labels
        )
    except Exception as e:
        logger.warning(f"分层划分失败，使用随机划分: {e}")
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, 
            test_size=config['data']['val_split'],
            random_state=config.get('reproducibility', {}).get('seed', 42)
        )

    logger.info(f"📊 数据划分完成:")
    logger.info(f"   训练集: {len(train_paths)} 张图像")
    logger.info(f"   验证集: {len(val_paths)} 张图像")

    # 创建数据变换 - 使用修复版本
    try:
        train_transform = get_train_transforms_fixed(config)
        logger.info("✅ 使用修复版训练变换（包含RandomErasing）")
    except Exception as e:
        logger.warning(f"修复版变换创建失败，使用安全版本: {e}")
        train_transform = get_train_transforms_safe(config)
        logger.info("🛡️ 使用安全版训练变换（移除RandomErasing）")

    val_transform = get_val_transforms(config)

    # 创建数据集
    cache_images = config['data'].get('cache_images', False)
    train_dataset = FlowerDataset(train_paths, train_labels, train_transform, cache_images=cache_images)
    val_dataset = FlowerDataset(val_paths, val_labels, val_transform)

    # 创建数据加载器
    num_workers = min(config['data']['num_workers'], os.cpu_count() or 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=config['data']['pin_memory'],
        drop_last=True,
        persistent_workers=config['data'].get('persistent_workers', False),
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(config.get('reproducibility', {}).get('seed', 42))
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config['data']['pin_memory'],
        persistent_workers=config['data'].get('persistent_workers', False)
    )

    logger.info("✅ 修复版数据加载器准备完成")
    return train_loader, val_loader, class_to_idx, idx_to_class



class MetricsCalculator:
    """优化的指标计算器 - 修复版"""

    def __init__(self, num_classes: int = 100, device: str = 'cpu'):
        self.num_classes = num_classes
        self.device = device
        self.reset()

    def reset(self):
        """重置所有累积的指标"""
        self.all_preds = []
        self.all_labels = []  
        self.all_probs = []
        self.all_confidences = []

    def update(self, preds: torch.Tensor, labels: torch.Tensor, probs: torch.Tensor):
        """更新指标"""
        # 移到CPU进行计算
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()
        probs = probs.cpu().numpy()

        self.all_preds.extend(preds)
        self.all_labels.extend(labels)
        self.all_probs.extend(probs)

        # 计算预测置信度
        max_probs = np.max(probs, axis=1)
        self.all_confidences.extend(max_probs)

    def compute_metrics(self) -> Dict[str, float]:
        """计算所有指标"""
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)
        probs = np.array(self.all_probs)
        confidences = np.array(self.all_confidences)

        # 基础指标
        top1_acc = accuracy_score(labels, preds)
        top5_acc = self._calculate_topk_accuracy(probs, labels, k=5)
        f1_macro = f1_score(labels, preds, average='macro', zero_division=0)

        # 置信度统计
        mean_confidence = np.mean(confidences)
        confidence_std = np.std(confidences)

        # 校准误差 (Expected Calibration Error)
        ece = self._calculate_ece(probs, labels, preds)

        # 竞赛综合得分 (70% Top-1 + 20% Top-5 + 10% F1)
        competition_score = 0.7 * top1_acc + 0.2 * top5_acc + 0.1 * f1_macro

        metrics = {
            'accuracy': top1_acc,
            'top5_accuracy': top5_acc,
            'f1_macro': f1_macro,
            'competition_score': competition_score,
            'mean_confidence': mean_confidence,
            'confidence_std': confidence_std,
            'expected_calibration_error': ece
        }

        return metrics

    def _calculate_topk_accuracy(self, probs: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
        """计算Top-K准确率"""
        topk_preds = np.argsort(probs, axis=1)[:, -k:]

        correct = 0
        for i, label in enumerate(labels):
            if label in topk_preds[i]:
                correct += 1

        return correct / len(labels)

    def _calculate_ece(self, probs: np.ndarray, labels: np.ndarray, preds: np.ndarray, n_bins: int = 15) -> float:
        """计算期望校准误差 (Expected Calibration Error)"""
        confidences = np.max(probs, axis=1)
        accuracies = (preds == labels).astype(float)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

    def get_confusion_matrix(self) -> np.ndarray:
        """获取混淆矩阵"""
        return confusion_matrix(self.all_labels, self.all_preds)

    def get_classification_report(self) -> str:
        """获取分类报告"""
        return classification_report(self.all_labels, self.all_preds, zero_division=0)

    def get_per_class_metrics(self) -> Dict[int, Dict[str, float]]:
        """获取每个类别的详细指标"""
        from sklearn.metrics import precision_recall_fscore_support

        precision, recall, f1, support = precision_recall_fscore_support(
            self.all_labels, self.all_preds, average=None, zero_division=0
        )

        per_class_metrics = {}
        for i in range(len(precision)):
            per_class_metrics[i] = {
                'precision': precision[i],
                'recall': recall[i], 
                'f1': f1[i],
                'support': support[i]
            }

        return per_class_metrics


def save_model_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                         scheduler, epoch: int, metrics: Dict[str, float], 
                         config: Dict, save_path: str, is_best: bool = False):
    """
    保存模型检查点 - 修复版
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'config': config,
        'model_info': {
            'total_params': sum(p.numel() for p in model.parameters()),
            'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
    }

    # 创建保存目录
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # 保存检查点
    torch.save(checkpoint, save_path)

    if is_best:
        best_path = Path(save_path).parent / 'best_model.pth'
        torch.save(checkpoint, best_path)
        logger.info(f"💾 最佳模型已保存: {best_path}")

    logger.info(f"✅ 检查点已保存: {save_path}")


def visualize_training_history_enhanced(history: Dict[str, List[float]], save_path: str = None):
    """
    增强的训练历史可视化 - 修复版

    Args:
        history: 训练历史字典
        save_path: 保存路径
    """
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 损失曲线
        if 'train_loss' in history and 'val_loss' in history:
            axes[0, 0].plot(history['train_loss'], label='训练损失', linewidth=2)
            axes[0, 0].plot(history['val_loss'], label='验证损失', linewidth=2)
            axes[0, 0].set_title('损失曲线', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # 准确率曲线
        if 'train_acc' in history and 'val_acc' in history:
            axes[0, 1].plot(history['train_acc'], label='训练准确率', linewidth=2)
            axes[0, 1].plot(history['val_acc'], label='验证准确率', linewidth=2)
            axes[0, 1].set_title('准确率曲线', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # Top-5准确率
        if 'val_top5_acc' in history:
            axes[0, 2].plot(history['val_top5_acc'], label='Top-5准确率', color='green', linewidth=2)
            axes[0, 2].set_title('Top-5准确率', fontsize=14, fontweight='bold')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Top-5 Accuracy')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)

        # 竞赛得分
        if 'val_competition_score' in history:
            axes[1, 0].plot(history['val_competition_score'], label='竞赛得分', color='red', linewidth=2)
            axes[1, 0].set_title('竞赛综合得分', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Competition Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # F1分数
        if 'val_f1_macro' in history:
            axes[1, 1].plot(history['val_f1_macro'], label='F1宏平均', color='purple', linewidth=2)
            axes[1, 1].set_title('F1宏平均分数', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('F1 Macro')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        # 置信度统计
        if 'val_mean_confidence' in history:
            axes[1, 2].plot(history['val_mean_confidence'], label='平均置信度', color='orange', linewidth=2)
            if 'val_expected_calibration_error' in history:
                ax2 = axes[1, 2].twinx()
                ax2.plot(history['val_expected_calibration_error'], 
                        label='校准误差', color='brown', linewidth=2, linestyle='--')
                ax2.set_ylabel('ECE')
                ax2.legend(loc='upper right')
            axes[1, 2].set_title('模型校准度', fontsize=14, fontweight='bold')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Mean Confidence')
            axes[1, 2].legend(loc='upper left')
            axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 训练历史图表已保存: {save_path}")

        plt.show()
    except Exception as e:
        logger.error(f"可视化失败: {e}")


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str] = None, 
                         save_path: str = None, normalize: bool = True):
    """
    绘制混淆矩阵 - 修复版

    Args:
        cm: 混淆矩阵
        class_names: 类别名称列表
        save_path: 保存路径
        normalize: 是否归一化
    """
    try:
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = '归一化混淆矩阵'
            fmt = '.2f'
        else:
            title = '混淆矩阵'
            fmt = 'd'

        plt.figure(figsize=(12, 10))

        # 只显示前20个类别，避免图表过大
        display_size = min(20, cm.shape[0])
        cm_display = cm[:display_size, :display_size]

        sns.heatmap(cm_display, 
                    annot=True, 
                    fmt=fmt, 
                    cmap='Blues',
                    xticklabels=class_names[:display_size] if class_names else range(display_size),
                    yticklabels=class_names[:display_size] if class_names else range(display_size))

        plt.title(f'{title} (前{display_size}个类别)', fontsize=16, fontweight='bold')
        plt.ylabel('真实标签', fontsize=12)
        plt.xlabel('预测标签', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 混淆矩阵已保存: {save_path}")

        plt.show()
    except Exception as e:
        logger.error(f"混淆矩阵绘制失败: {e}")


def create_submission_file(predictions: List[Tuple[str, int, float]], 
                          output_path: str, 
                          class_mapping: Dict[int, str] = None):
    """
    创建竞赛提交文件 - 修复版

    Args:
        predictions: 预测结果列表 [(image_id, pred_class, confidence), ...]
        output_path: 输出路径
        class_mapping: 类别映射字典
    """
    df_data = []

    for result in predictions:
        if len(result) == 3:
            image_id, pred_class, confidence = result
        else:
            image_id, pred_class = result
            confidence = 1.0

        # 转换标签格式
        if class_mapping and pred_class in class_mapping:
            label = class_mapping[pred_class]
        else:
            label = str(pred_class + 1)  # 0-99 -> 1-100

        df_data.append({
            'img_name': image_id,
            'predicted_class': label,
            'confidence': confidence
        })

    # 创建DataFrame并保存
    df = pd.DataFrame(df_data)

    # 确保目录存在
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # 保存为CSV（不包含索引）
    df.to_csv(output_path, index=False, encoding='utf-8')

    logger.info(f"✅ 提交文件已保存: {output_path}")
    logger.info(f"📊 提交统计:")
    logger.info(f"   总预测数: {len(df)} 个")
    logger.info(f"   平均置信度: {df['confidence'].mean():.4f}")
    logger.info(f"   预测类别数: {df['predicted_class'].nunique()} 个")

    return df


def format_time(seconds: float) -> str:
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.2f} 秒"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)} 分 {seconds:.1f} 秒"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)} 时 {int(minutes)} 分 {seconds:.1f} 秒"


def print_model_summary(model: nn.Module, input_size: Tuple[int, int, int] = (3, 224, 224)):
    """
    打印模型摘要信息 - 修复版

    Args:
        model: 模型
        input_size: 输入尺寸
    """
    try:
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info("📊 模型摘要信息:")
        logger.info("="*50)
        logger.info(f"📊 总参数量: {total_params:,}")
        logger.info(f"📊 可训练参数: {trainable_params:,}")
        logger.info(f"📊 冻结参数: {total_params - trainable_params:,}")
        logger.info(f"📊 模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")
        logger.info(f"📊 可训练比例: {trainable_params / total_params:.2%}")

        # 测试推理时间
        device = next(model.parameters()).device
        model.eval()

        dummy_input = torch.randn(1, *input_size).to(device)

        # 预热
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)

        # 测试推理速度
        import time
        if device.type == 'cuda':
            torch.cuda.synchronize()

        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = model(dummy_input)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()

        avg_inference_time = (end_time - start_time) / 100 * 1000  # 转换为毫秒
        logger.info(f"📊 平均推理时间: {avg_inference_time:.2f} ms")
        logger.info(f"📊 推理FPS: {1000/avg_inference_time:.1f}")
        logger.info("="*50)
    except Exception as e:
        logger.error(f"模型摘要生成失败: {e}")


# 导出主要函数
__all__ = [
    'LabelSmoothingCrossEntropy',
    'CutMixUp', 
    'set_seed_strict',
    'set_seed',
    'Logger',
    'ConfigLoader',
    'create_directories',
    'worker_init_fn',
    'FlowerDataset',
    'load_class_mappings',
    'prepare_data_loaders',
    'get_train_transforms_fixed',
    'get_train_transforms_safe',
    'get_val_transforms',
    'get_test_transforms', 
    'get_tta_transforms',
    'MetricsCalculator',
    'save_model_checkpoint',
    'visualize_training_history_enhanced',
    'plot_confusion_matrix',
    'create_submission_file',
    'format_time',
    'print_model_summary'
]
