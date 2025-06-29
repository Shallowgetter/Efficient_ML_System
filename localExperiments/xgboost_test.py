#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost Baseline (parameters read from YAML)

运行示例：
    python xgboost_test.py \
        --config localExperiments/model_param/xgb_params.yaml \
        --model_name XGB_test_v1
"""
import os, sys, time, yaml, argparse, datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
from utils.utils import get_logger, plot_confusion_matrix

# ------------------------------------------------------------------
def parse_arguments():
    p = argparse.ArgumentParser(description='XGBoost baseline')
    p.add_argument('--config', type=str,
                   default='localExperiments/model_param/xgb_params.yaml')
    p.add_argument('--model_name', type=str, default='XGB_test_v1')
    return p.parse_args()

def load_cfg(path, model_name):
    with open(path, 'r') as f:
        cfg_all = yaml.safe_load(f)
    for item in cfg_all['model']:
        if item['model_name'] == model_name:
            return item
    raise ValueError(f'Model {model_name} not found in {path}')

# ------------------------------------------------------------------
def load_npz_dataset(window_size=300, overlap=0.3):
    tr = np.load(f'data/SHL_2018/all_data_train_0.5_window_{window_size}_overlap_{overlap}.npz')
    te = np.load(f'data/SHL_2018/all_data_test_0.5_window_{window_size}_overlap_{overlap}.npz')
    x_tr, y_tr = tr["x"].reshape(len(tr["x"]), -1), tr["y"]
    x_te, y_te = te["x"].reshape(len(te["x"]), -1), te["y"]
    y_tr = np.argmax(y_tr, 1) if y_tr.ndim > 1 else y_tr
    y_te = np.argmax(y_te, 1) if y_te.ndim > 1 else y_te
    return (x_tr, y_tr.astype(int)), (x_te, y_te.astype(int))

# ------------------------------------------------------------------
def train_eval(cfg):
    os.makedirs(cfg["log_dir"],  exist_ok=True)
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)
    logger = get_logger(os.path.join(cfg["log_dir"],
                        f'{cfg["model_name"]}.log'),
                        name=f'{cfg["model_name"]}Logger',
                        overwrite=True, to_stdout=True)

    # ------------ 数据 ------------
    (x_train, y_train), (x_test, y_test) = \
        load_npz_dataset(cfg["window_size"], cfg["overlap"])
    logger.info(f'Train {x_train.shape}  Test {x_test.shape}')

    # ------------ 模型 ------------
    xgb_params = cfg["xgb_params"]
    model = XGBClassifier(**xgb_params)
    logger.info(f'XGB params: {xgb_params}')

    # ------------ 训练 ------------
    t0 = time.time()
    model.fit(x_train, y_train,
              eval_set=[(x_test, y_test)],
              verbose=False)
    logger.info(f'Training time: {time.time()-t0:.2f}s')

    # ------------ 保存模型 ------------
    ckpt_path = os.path.join(cfg["ckpt_dir"],
                             f'{cfg["model_name"]}_best.xgb')
    model.save_model(ckpt_path)
    logger.info(f'Saved to {ckpt_path}')

    # ------------ 推理与指标 ------------
    infer_start = time.time()
    y_pred = model.predict(x_test)
    infer_ms = (time.time() - infer_start) / len(x_test) * 1e3
    acc = accuracy_score(y_test, y_pred) * 100
    f1m = f1_score(y_test, y_pred, average='macro')
    logger.info(f'ACC={acc:.2f}%  F1-macro={f1m:.4f}  '
                f'Inference={infer_ms:.3f} ms/样本')

    # ------------ 混淆矩阵 ------------
    cm = confusion_matrix(y_test, y_pred,
                          labels=range(cfg["num_classes"]))
    plot_confusion_matrix(y_test, y_pred,
                          class_names=[str(i) for i in range(cfg["num_classes"])],
                          normalize=True, fontsize=18)
    import matplotlib.pyplot as plt
    plt.title(f'Confusion Matrix – {cfg["model_name"]}\n'
              f'ACC {acc:.2f}% | F1 {f1m:.4f}')
    fig_path = os.path.join(cfg["ckpt_dir"],
                            f'{cfg["model_name"]}_cm.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f'CM saved: {fig_path}')

    # ------------ 摘要文件 ------------
    with open(os.path.join(cfg["log_dir"],
              f'{cfg["model_name"]}_summary.txt'), 'w') as f:
        f.write(f'Time {datetime.datetime.now()}\n')
        f.write(f'ACC {acc:.2f}%  F1 {f1m:.4f}\n')
        f.write(f'Inference {infer_ms:.3f} ms/sample\n')
        f.write(f'Model {ckpt_path}\n')

# ------------------------------------------------------------------
if __name__ == '__main__':
    args = parse_arguments()
    cfg = load_cfg(args.config, args.model_name)
    train_eval(cfg)
