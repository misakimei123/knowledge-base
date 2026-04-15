#!/usr/bin/env python3
"""
置信度校准训练脚本
使用黄金数据集训练校准参数（温度缩放/Platt Scaling）
"""

import json
import numpy as np
from typing import List, Tuple
from scipy.optimize import minimize
from pathlib import Path


class ConfidenceCalibratorTrainer:
    """置信度校准器训练工具"""
    
    def __init__(self, method: str = "temperature_scaling"):
        self.method = method
        self.params = {}
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid 函数"""
        return 1 / (1 + np.exp(-x))
    
    def _logit(self, p: np.ndarray) -> np.ndarray:
        """Logit 变换"""
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p))
    
    def _temperature_loss(self, T: float, logits: np.ndarray, labels: np.ndarray) -> float:
        """温度缩放损失函数（负对数似然）"""
        T = max(T, 0.1)  # 防止除零
        calibrated_probs = self._sigmoid(logits / T)
        calibrated_probs = np.clip(calibrated_probs, 1e-7, 1 - 1e-7)
        
        # 二元交叉熵损失
        loss = -np.mean(
            labels * np.log(calibrated_probs) + 
            (1 - labels) * np.log(1 - calibrated_probs)
        )
        return loss
    
    def _platt_loss(self, params: Tuple[float, float], logits: np.ndarray, labels: np.ndarray) -> float:
        """Platt Scaling 损失函数"""
        A, B = params
        calibrated_probs = self._sigmoid(A * logits + B)
        calibrated_probs = np.clip(calibrated_probs, 1e-7, 1 - 1e-7)
        
        loss = -np.mean(
            labels * np.log(calibrated_probs) + 
            (1 - labels) * np.log(1 - calibrated_probs)
        )
        return loss
    
    def fit_temperature_scaling(self, raw_scores: List[float], labels: List[int]) -> dict:
        """训练温度缩放参数"""
        raw_scores = np.array(raw_scores)
        labels = np.array(labels)
        
        # 转换为 logit
        logits = self._logit(raw_scores)
        
        # 优化温度参数
        result = minimize(
            fun=lambda T: self._temperature_loss(T[0], logits, labels),
            x0=[1.0],
            method='L-BFGS-B',
            bounds=[(0.1, 10.0)]
        )
        
        temperature = result.x[0]
        print(f"✅ 温度缩放训练完成：T = {temperature:.4f}")
        
        self.params = {"temperature": float(temperature)}
        return self.params
    
    def fit_platt_scaling(self, raw_scores: List[float], labels: List[int]) -> dict:
        """训练 Platt Scaling 参数"""
        raw_scores = np.array(raw_scores)
        labels = np.array(labels)
        
        # 转换为 logit
        logits = self._logit(raw_scores)
        
        # 优化 A, B 参数
        result = minimize(
            fun=lambda params: self._platt_loss(params, logits, labels),
            x0=[1.0, 0.0],
            method='L-BFGS-B',
            bounds=[(-5.0, 5.0), (-5.0, 5.0)]
        )
        
        A, B = result.x
        print(f"✅ Platt Scaling 训练完成：A = {A:.4f}, B = {B:.4f}")
        
        self.params = {"A": float(A), "B": float(B)}
        return self.params
    
    def fit(self, raw_scores: List[float], labels: List[int]) -> dict:
        """根据配置选择方法训练"""
        if self.method == "temperature_scaling":
            return self.fit_temperature_scaling(raw_scores, labels)
        elif self.method == "platt_scaling":
            return self.fit_platt_scaling(raw_scores, labels)
        else:
            raise ValueError(f"未知校准方法：{self.method}")
    
    def save_params(self, path: str):
        """保存校准参数"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                "method": self.method,
                "params": self.params
            }, f, indent=2)
        print(f"✅ 校准参数已保存到：{path}")
    
    def load_params(self, path: str) -> dict:
        """加载校准参数"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.method = data["method"]
        self.params = data["params"]
        return self.params


def load_gold_dataset(dataset_path: str) -> Tuple[List[float], List[int]]:
    """
    加载黄金数据集
    :return: (raw_confidence_scores, binary_labels)
             labels: 1=正确，0=错误
    """
    import yaml
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = yaml.safe_load(f)
    
    raw_scores = []
    labels = []
    
    for sample in dataset.get("samples", []):
        # 假设有 simulated_confidence 字段（未校准的原始分数）
        if "simulated_confidence" in sample:
            raw_scores.append(sample["simulated_confidence"])
        elif "confidence" in sample:
            raw_scores.append(sample["confidence"])
        else:
            continue
        
        # 标签：answer_correct=True → 1, else 0
        label = 1 if sample.get("answer_correct", False) else 0
        labels.append(label)
    
    print(f"📊 加载黄金数据集：{len(raw_scores)} 条样本")
    print(f"   - 正样本（正确）：{sum(labels)}")
    print(f"   - 负样本（错误）：{len(labels) - sum(labels)}")
    
    return raw_scores, labels


def evaluate_calibration(raw_scores: List[float], labels: List[int], params: dict, method: str) -> dict:
    """评估校准效果"""
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    
    raw_scores = np.array(raw_scores)
    labels = np.array(labels)
    
    # 计算校准后的分数
    if method == "temperature_scaling":
        T = params["temperature"]
        logits = np.log(raw_scores / (1 - raw_scores + 1e-7))
        calibrated_scores = 1 / (1 + np.exp(-logit / T))
    elif method == "platt_scaling":
        A, B = params["A"], params["B"]
        logits = np.log(raw_scores / (1 - raw_scores + 1e-7))
        calibrated_scores = 1 / (1 + np.exp(-(A * logits + B)))
    else:
        calibrated_scores = raw_scores
    
    # 评估指标
    metrics = {
        "brier_score_raw": float(brier_score_loss(labels, raw_scores)),
        "brier_score_calibrated": float(brier_score_loss(labels, calibrated_scores)),
        "log_loss_raw": float(log_loss(labels, raw_scores)),
        "log_loss_calibrated": float(log_loss(labels, calibrated_scores)),
        "roc_auc": float(roc_auc_score(labels, calibrated_scores)),
    }
    
    # 计算 ECE (Expected Calibration Error)
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (calibrated_scores >= bin_boundaries[i]) & (calibrated_scores < bin_boundaries[i + 1])
        if np.sum(mask) > 0:
            bin_acc = np.mean(labels[mask])
            bin_conf = np.mean(calibrated_scores[mask])
            ece += np.abs(bin_acc - bin_conf) * np.sum(mask) / len(labels)
    
    metrics["ece"] = float(ece)
    
    return metrics


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="置信度校准训练")
    parser.add_argument("--dataset", type=str, default="tests/gold_dataset/samples.yaml",
                        help="黄金数据集路径")
    parser.add_argument("--output", type=str, default="models/calibration_params.json",
                        help="校准参数输出路径")
    parser.add_argument("--method", type=str, default="temperature_scaling",
                        choices=["temperature_scaling", "platt_scaling"],
                        help="校准方法")
    
    args = parser.parse_args()
    
    # 创建输出目录
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    print("🚀 开始置信度校准训练...")
    print(f"   - 数据集：{args.dataset}")
    print(f"   - 方法：{args.method}")
    
    # 加载数据
    raw_scores, labels = load_gold_dataset(args.dataset)
    
    if len(raw_scores) < 10:
        print("❌ 数据量不足，需要至少 10 条样本")
        return
    
    # 训练校准器
    trainer = ConfidenceCalibratorTrainer(method=args.method)
    params = trainer.fit(raw_scores, labels)
    
    # 评估校准效果
    print("\n📊 校准效果评估:")
    metrics = evaluate_calibration(raw_scores, labels, params, args.method)
    
    print(f"   - Brier Score (原始): {metrics['brier_score_raw']:.4f}")
    print(f"   - Brier Score (校准): {metrics['brier_score_calibrated']:.4f}")
    print(f"   - 改善幅度：{(metrics['brier_score_raw'] - metrics['brier_score_calibrated']) / metrics['brier_score_raw'] * 100:.1f}%")
    print(f"   - Log Loss (原始): {metrics['log_loss_raw']:.4f}")
    print(f"   - Log Loss (校准): {metrics['log_loss_calibrated']:.4f}")
    print(f"   - ECE (校准误差): {metrics['ece']:.4f}")
    print(f"   - ROC AUC: {metrics['roc_auc']:.4f}")
    
    # 保存参数
    trainer.save_params(args.output)
    
    print("\n✨ 校准训练完成！")


if __name__ == "__main__":
    main()
