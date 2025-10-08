"""
花卉识别AI挑战赛 - 预测脚本 (优化版)
AI Competition Coach - Flower Recognition Challenge Prediction v1.1.0

优化内容:
1. 🌡️ 支持温度缩放的推理
2. 🎯 TTA (Test Time Augmentation) 集成
3. 📊 置信度校准和统计
4. 🔄 批量预测优化
5. 📋 增强的提交文件生成
6. 🛡️ 错误处理和恢复机制

使用方法:
python predict.py --config config.json --model_path models/best_model.pth --test_dir data/test_images --output results/submission.csv --use_tta
"""

import os
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from model import FlowerViTModel, ModelFactory, ModelUtils
from utils import (
    Logger, ConfigLoader, set_seed_strict, FlowerDataset,
    get_test_transforms, get_tta_transforms, create_submission_file,
    format_time, load_class_mappings
)


class OptimizedPredictor:
    """优化的预测器类"""

    def __init__(self, config: Dict, model_path: str):
        """
        初始化预测器

        Args:
            config: 配置字典
            model_path: 模型权重路径
        """
        self.config = config
        self.model_path = model_path
        self.logger = Logger(config['paths']['logs_dir'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 设置随机种子
        if config.get('reproducibility', {}).get('deterministic', True):
            set_seed_strict(config['reproducibility']['seed'])

        self.logger.info("🚀 优化预测器初始化完成")
        self.logger.info(f"🔧 使用设备: {self.device}")

        # 加载模型
        self.model = self._load_model()

        # 加载类别映射
        self.class_to_idx, self.idx_to_class = load_class_mappings(
            config['data']['class_mapping'])

    def _load_model(self) -> nn.Module:
        """加载训练好的模型"""
        self.logger.info(f"📥 加载模型: {self.model_path}")

        # 创建模型
        model = ModelFactory.create_vit_model(self.config)
        model = model.to(self.device)

        # 加载权重
        try:
            load_info = ModelUtils.load_model_weights(model, self.model_path, strict=False)

            if 'error' not in load_info:
                self.logger.info("✅ 模型权重加载成功")

                # 记录模型信息
                if 'checkpoint_info' in load_info:
                    info = load_info['checkpoint_info']
                    if 'metrics' in info:
                        metrics = info['metrics']
                        self.logger.info(f"📊 模型性能:")
                        self.logger.info(f"   验证准确率: {metrics.get('accuracy', 0):.4f}")
                        self.logger.info(f"   Top-5准确率: {metrics.get('top5_accuracy', 0):.4f}")
                        self.logger.info(f"   竞赛得分: {metrics.get('competition_score', 0):.4f}")
            else:
                self.logger.error(f"❌ 模型加载失败: {load_info['error']}")
                raise Exception(load_info['error'])

        except Exception as e:
            self.logger.error(f"❌ 模型权重加载失败: {e}")
            raise e

        model.eval()
        return model

    def _prepare_test_dataloader(self, test_dir: str, use_tta: bool = False) -> DataLoader:
        """准备测试数据加载器"""
        self.logger.info(f"📂 准备测试数据: {test_dir}")

        # 收集测试图像
        test_dir = Path(test_dir)
        image_paths = []

        if test_dir.is_dir():
            # 支持多种图像格式
            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            for ext in valid_extensions:
                image_paths.extend(test_dir.glob(f'*{ext}'))
                image_paths.extend(test_dir.glob(f'*{ext.upper()}'))
        else:
            raise ValueError(f"测试目录不存在: {test_dir}")

        image_paths = [str(p) for p in sorted(image_paths)]
        self.logger.info(f"📊 找到测试图像: {len(image_paths)} 张")

        if len(image_paths) == 0:
            raise ValueError("测试目录中没有找到有效图像")

        # 选择变换
        if use_tta:
            # TTA模式：创建多个变换
            transforms_list = get_tta_transforms(
                self.config, 
                self.config['inference'].get('tta_steps', 5)
            )
            self.logger.info(f"🎯 TTA模式: {len(transforms_list)} 种变换")
        else:
            # 标准模式
            transforms_list = [get_test_transforms(self.config)]
            self.logger.info("🎯 标准推理模式")

        # 创建数据集和加载器
        datasets = []
        for transform in transforms_list:
            dataset = FlowerDataset(
                image_paths=image_paths,
                labels=None,  # 测试模式不需要标签
                transform=transform
            )
            datasets.append(dataset)

        return datasets, image_paths

    def _predict_single_batch(self, images: torch.Tensor, 
                            temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测单个批次

        Args:
            images: 输入图像批次
            temperature: 温度缩放参数

        Returns:
            predictions, confidences
        """
        with torch.no_grad():
            # 前向传播
            logits = self.model(images.to(self.device))

            # 应用温度缩放
            if temperature != 1.0:
                logits = logits / temperature

            # 计算概率和预测
            probs = F.softmax(logits, dim=1)
            confidences, predictions = torch.max(probs, dim=1)

        return predictions.cpu(), confidences.cpu()

    def _predict_with_tta(self, datasets: List[FlowerDataset], 
                         image_paths: List[str], batch_size: int = 64) -> List[Tuple[str, int, float]]:
        """
        使用TTA进行预测

        Args:
            datasets: 数据集列表（不同变换）
            image_paths: 图像路径列表
            batch_size: 批次大小

        Returns:
            预测结果列表
        """
        self.logger.info("🎯 开始TTA预测...")

        num_images = len(image_paths)
        num_augmentations = len(datasets)

        # 存储所有变换的预测结果
        all_probs = np.zeros((num_images, num_augmentations, self.config['model']['num_classes']))

        # 对每种变换进行预测
        for aug_idx, dataset in enumerate(datasets):
            self.logger.info(f"🔄 TTA变换 {aug_idx+1}/{num_augmentations}")

            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=min(4, os.cpu_count()),
                pin_memory=True
            )

            batch_start_idx = 0
            for images, _ in tqdm(dataloader, desc=f'TTA {aug_idx+1}', leave=False):
                batch_size_actual = images.size(0)

                # 预测
                with torch.no_grad():
                    logits = self.model(images.to(self.device))
                    probs = F.softmax(logits, dim=1).cpu().numpy()

                # 存储概率
                end_idx = batch_start_idx + batch_size_actual
                all_probs[batch_start_idx:end_idx, aug_idx] = probs
                batch_start_idx = end_idx

        # 集成预测结果（平均概率）
        self.logger.info("🔀 集成TTA预测结果...")
        ensemble_probs = np.mean(all_probs, axis=1)
        predictions = np.argmax(ensemble_probs, axis=1)
        confidences = np.max(ensemble_probs, axis=1)

        # 生成结果
        results = []
        for i, image_path in enumerate(image_paths):
            image_id = Path(image_path).stem
            pred_class = int(predictions[i])
            confidence = float(confidences[i])
            results.append((image_id, pred_class, confidence))

        return results

    def _predict_standard(self, datasets: List[FlowerDataset], 
                         image_paths: List[str], batch_size: int = 64) -> List[Tuple[str, int, float]]:
        """
        标准预测模式

        Args:
            datasets: 数据集列表（单个变换）
            image_paths: 图像路径列表
            batch_size: 批次大小

        Returns:
            预测结果列表
        """
        self.logger.info("🎯 开始标准预测...")

        dataset = datasets[0]  # 只使用第一个变换
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=min(4, os.cpu_count()),
            pin_memory=True
        )

        results = []
        temperature = self.config['model'].get('temperature', 1.0)

        batch_start_idx = 0
        for images, image_ids in tqdm(dataloader, desc='预测中'):
            # 预测
            predictions, confidences = self._predict_single_batch(images, temperature)

            # 处理结果
            for i in range(len(image_ids)):
                image_id = image_ids[i]
                pred_class = int(predictions[i])
                confidence = float(confidences[i])
                results.append((image_id, pred_class, confidence))

        return results

    def predict(self, test_dir: str, output_path: str, 
               use_tta: bool = False, batch_size: Optional[int] = None) -> pd.DataFrame:
        """
        主预测函数

        Args:
            test_dir: 测试图像目录
            output_path: 输出文件路径
            use_tta: 是否使用TTA
            batch_size: 批次大小

        Returns:
            预测结果DataFrame
        """
        start_time = time.time()

        # 使用配置中的批次大小
        if batch_size is None:
            batch_size = self.config['inference']['batch_size']

        self.logger.info(f"🚀 开始预测任务")
        self.logger.info(f"📂 测试目录: {test_dir}")
        self.logger.info(f"📄 输出文件: {output_path}")
        self.logger.info(f"🎯 TTA模式: {'启用' if use_tta else '禁用'}")
        self.logger.info(f"📦 批次大小: {batch_size}")

        # 准备数据
        datasets, image_paths = self._prepare_test_dataloader(test_dir, use_tta)

        # 执行预测
        if use_tta:
            results = self._predict_with_tta(datasets, image_paths, batch_size)
        else:
            results = self._predict_standard(datasets, image_paths, batch_size)

        # 创建提交文件
        submission_df = create_submission_file(
            results, output_path, self.idx_to_class
        )

        # 预测统计
        total_time = time.time() - start_time
        self.logger.info(f"🎉 预测完成!")
        self.logger.info(f"📊 预测统计:")
        self.logger.info(f"   预测图像数: {len(results)}")
        self.logger.info(f"   平均置信度: {np.mean([r[2] for r in results]):.4f}")
        self.logger.info(f"   置信度标准差: {np.std([r[2] for r in results]):.4f}")
        self.logger.info(f"   预测类别数: {len(set(r[1] for r in results))}")
        self.logger.info(f"⏱️  总用时: {format_time(total_time)}")
        self.logger.info(f"🚀 平均速度: {len(results)/total_time:.1f} 张/秒")

        return submission_df

    def predict_single_image(self, image_path: str, 
                           use_tta: bool = False) -> Tuple[int, float, Dict[int, float]]:
        """
        预测单张图像

        Args:
            image_path: 图像路径
            use_tta: 是否使用TTA

        Returns:
            predicted_class, confidence, top5_probs
        """
        self.logger.info(f"🖼️ 预测单张图像: {image_path}")

        # 加载和预处理图像
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            self.logger.error(f"❌ 图像加载失败: {e}")
            raise e

        if use_tta:
            # TTA预测
            transforms_list = get_tta_transforms(
                self.config, self.config['inference'].get('tta_steps', 5)
            )

            all_probs = []
            for transform in transforms_list:
                img_tensor = transform(image).unsqueeze(0)

                with torch.no_grad():
                    logits = self.model(img_tensor.to(self.device))
                    probs = F.softmax(logits, dim=1)
                    all_probs.append(probs.cpu().numpy())

            # 平均概率
            ensemble_probs = np.mean(all_probs, axis=0).flatten()

        else:
            # 标准预测
            transform = get_test_transforms(self.config)
            img_tensor = transform(image).unsqueeze(0)

            temperature = self.config['model'].get('temperature', 1.0)

            with torch.no_grad():
                logits = self.model(img_tensor.to(self.device))
                if temperature != 1.0:
                    logits = logits / temperature
                probs = F.softmax(logits, dim=1)
                ensemble_probs = probs.cpu().numpy().flatten()

        # 获取预测结果
        predicted_class = int(np.argmax(ensemble_probs))
        confidence = float(np.max(ensemble_probs))

        # 获取Top-5概率
        top5_indices = np.argsort(ensemble_probs)[-5:][::-1]
        top5_probs = {int(idx): float(ensemble_probs[idx]) for idx in top5_indices}

        # 日志记录
        class_name = self.idx_to_class.get(predicted_class, f"类别{predicted_class+1}")
        self.logger.info(f"🎯 预测结果: {class_name} (置信度: {confidence:.4f})")

        return predicted_class, confidence, top5_probs


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='花卉识别预测脚本 - 优化版')
    parser.add_argument('--config', type=str, default='config.json',
                       help='配置文件路径')
    parser.add_argument('--model_path', type=str, required=True,
                       help='训练好的模型权重路径')
    parser.add_argument('--test_dir', type=str, required=True,
                       help='测试图像目录')
    parser.add_argument('--output', type=str, default='results/submission.csv',
                       help='输出文件路径')
    parser.add_argument('--use_tta', action='store_true',
                       help='使用TTA (Test Time Augmentation)')
    parser.add_argument('--batch_size', type=int,
                       help='批次大小 (覆盖配置)')
    parser.add_argument('--single_image', type=str,
                       help='预测单张图像的路径')

    args = parser.parse_args()

    # 加载配置
    config = ConfigLoader.load_config(args.config)

    # 命令行参数覆盖
    if args.batch_size:
        config['inference']['batch_size'] = args.batch_size
    if args.use_tta:
        config['inference']['use_tta'] = True

    # 创建预测器
    predictor = OptimizedPredictor(config, args.model_path)

    if args.single_image:
        # 单图像预测
        pred_class, confidence, top5_probs = predictor.predict_single_image(
            args.single_image, args.use_tta
        )

        print(f"\n🎯 预测结果:")
        print(f"预测类别: {pred_class + 1}")
        print(f"置信度: {confidence:.4f}")
        print(f"\nTop-5预测:")
        for i, (class_idx, prob) in enumerate(top5_probs.items(), 1):
            print(f"{i}. 类别{class_idx + 1}: {prob:.4f}")

    else:
        # 批量预测
        submission_df = predictor.predict(
            test_dir=args.test_dir,
            output_path=args.output,
            use_tta=args.use_tta,
            batch_size=args.batch_size
        )

        print(f"\n✅ 预测完成，提交文件已保存至: {args.output}")


if __name__ == "__main__":
    main()
