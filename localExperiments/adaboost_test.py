#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AdaBoost Baseline（参数由 YAML 读取，训练进度使用 tqdm 可视化）
"""

import os, sys, yaml, time, datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from utils.utils import get_logger, plot_confusion_matrix

# ------------------------------------------------------------
def load_cfg(path: str, model_name: str) -> dict:
    with open(path, "r") as f:
        cfg_all = yaml.safe_load(f)
    for item in cfg_all["model"]:
        if item["model_name"] == model_name:
            return item
    raise ValueError(f"模型 {model_name} 未在 {path} 中定义")

# ------------------------------------------------------------
def load_npz_dataset(window_size: int = 300, overlap: float = 0.3):
    tr = np.load(f"data/SHL_2018/all_data_train_0.8_window_{window_size}_overlap_{overlap}.npz")
    te = np.load(f"data/SHL_2018/all_data_test_0.8_window_{window_size}_overlap_{overlap}.npz")
    x_tr, y_tr = tr["x"].reshape(len(tr["x"]), -1), tr["y"]
    x_te, y_te = te["x"].reshape(len(te["x"]), -1), te["y"]
    y_tr = np.argmax(y_tr, 1) if y_tr.ndim > 1 else y_tr
    y_te = np.argmax(y_te, 1) if y_te.ndim > 1 else y_te
    return (x_tr, y_tr.astype(int)), (x_te, y_te.astype(int))

# ------------------------------------------------------------
def train_and_eval(cfg: dict):
    os.makedirs(cfg["log_dir"],  exist_ok=True)
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)
    logger = get_logger(
        filename=os.path.join(cfg["log_dir"], f'{cfg["model_name"]}.log'),
        name=f'{cfg["model_name"]}Logger',
        overwrite=True, to_stdout=True)

    # ---------- 数据 ----------
    (x_train, y_train), (x_test, y_test) = load_npz_dataset(cfg["window_size"], cfg["overlap"])
    logger.info(f"Train {x_train.shape}  Test {x_test.shape}")

    # ---------- 模型 ----------
    adb_params = cfg["adaboost_params"].copy()   # 避免 pop 影响 cfg
    total_estimators = adb_params.pop("n_estimators")
    model = AdaBoostClassifier(**adb_params, n_estimators=1, warm_start=True)
    logger.info(f"AdaBoost 参数:  { {'n_estimators': total_estimators, **adb_params} }")

    # ---------- 逐棵增量训练 ----------
    t0 = time.time()
    for i in tqdm(range(1, total_estimators + 1), desc="AdaBoost Estimators", ncols=80):
        model.n_estimators = i
        model.fit(x_train, y_train)              # warm_start=True ⇒ 累积新基学习器

        # 每 20 棵树或最后一棵时，实时验证
        if i % 20 == 0 or i == total_estimators:
            val_pred = model.predict(x_test)
            val_acc  = accuracy_score(y_test, val_pred) * 100
            logger.info(f"迭代 {i:3d}/{total_estimators}: Val-ACC={val_acc:6.2f}%")

    train_time = time.time() - t0
    logger.info(f"训练完成，总耗时 {train_time:.2f}s")

    # ---------- 保存模型 ----------
    ckpt_path = os.path.join(cfg["ckpt_dir"], f'{cfg["model_name"]}_best.adb')
    model.save(ckpt_path) if hasattr(model, "save") else np.save(ckpt_path, model)  # 兼容性存储
    logger.info(f"模型已保存至 {ckpt_path}")

    # ---------- 推理 ----------
    infer_start = time.time()
    y_pred = model.predict(x_test)
    infer_ms = (time.time() - infer_start) / len(x_test) * 1e3

    acc  = accuracy_score(y_test, y_pred) * 100
    f1_m = f1_score(y_test, y_pred, average='macro')
    logger.info(f"最终指标: ACC={acc:.2f}%  F1-macro={f1_m:.4f}  "
                f"推理时延={infer_ms:.3f} ms/样本")

    # ---------- 混淆矩阵 ----------
    cm = confusion_matrix(y_test, y_pred, labels=range(cfg["num_classes"]))
    plot_confusion_matrix(y_test, y_pred,
                          class_names=[str(i) for i in range(cfg["num_classes"])],
                          normalize=True, fontsize=18)
    import matplotlib.pyplot as plt
    plt.title(f'Confusion Matrix – {cfg["model_name"]}\n'
              f'ACC {acc:.2f}% | F1 {f1_m:.4f}')
    fig_path = os.path.join(cfg["ckpt_dir"], f'{cfg["model_name"]}_cm.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"混淆矩阵已保存至 {fig_path}")

    # ---------- 总结 ----------
    with open(os.path.join(cfg["log_dir"], f'{cfg["model_name"]}_summary.txt'), 'w') as f:
        f.write(f'Time {datetime.datetime.now()}\n')
        f.write(f'ACC {acc:.2f}%   F1 {f1_m:.4f}\n')
        f.write(f'Inference {infer_ms:.3f} ms/sample\n')
        f.write(f'Model {ckpt_path}\n')

# ------------------------------------------------------------
if __name__ == "__main__":
    # 直接修改下方路径/名称即可运行；避免依赖命令行
    CONFIG_PATH = "localExperiments/model_param/adaboost_params.yaml"
    MODEL_NAME  = "ADB_test_v1"
    cfg = load_cfg(CONFIG_PATH, MODEL_NAME)
    train_and_eval(cfg)
