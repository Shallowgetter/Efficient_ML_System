# ================================================================
# Random-Forest Baseline 训练与评估脚本
# ================================================================
import os, time, yaml, datetime
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ------------------ 配置读取 ------------------
def load_cfg(cfg_path="localExperiments/model_param/rf_params.yaml", model_name="RF_baseline_v1"):
    with open(cfg_path, "r") as f:
        root = yaml.safe_load(f)
    for m in root["model"]:
        if m["model_name"] == model_name:
            return m
    raise KeyError(f"{model_name} not found in {cfg_path}")

# ------------------ 数据加载 ------------------
def load_dataset(window_size, overlap):
    """沿用 cnn_test.py 的 npz 文件，返回展平后的特征矩阵"""
    train = np.load("data/SHL_2018/all_data_train_0.8_window_300_overlap_0.3.npz")
    test  = np.load("data/SHL_2018/all_data_test_0.8_window_300_overlap_0.3.npz")

    x_tr, y_tr_raw = train["x"], train["y"]
    x_te, y_te_raw = test ["x"], test ["y"]

    # one-hot → label
    y_tr = np.argmax(y_tr_raw, 1) if y_tr_raw.ndim == 2 else y_tr_raw
    y_te = np.argmax(y_te_raw, 1) if y_te_raw.ndim == 2 else y_te_raw

    # (N, C, W) → (N, C·W)
    x_tr = x_tr.reshape(x_tr.shape[0], -1)
    x_te = x_te.reshape(x_te.shape[0], -1)
    return x_tr, y_tr, x_te, y_te

# ------------------ 主流程 ------------------
def main():
    cfg = load_cfg()
    rf_kw = cfg["rf_params"]

    os.makedirs(cfg["log_dir"], exist_ok=True)
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)

    print(f"[{datetime.datetime.now()}] 载入数据…")
    x_tr, y_tr, x_te, y_te = load_dataset(cfg["window_size"], cfg["overlap"])
    print(f"Train: {x_tr.shape} | Test: {x_te.shape}")

    print(f"[{datetime.datetime.now()}] 训练 Random-Forest…")
    clf = RandomForestClassifier(**rf_kw)
    t0 = time.time()
    clf.fit(x_tr, y_tr)
    train_time = time.time() - t0

    print(f"[{datetime.datetime.now()}] 推理与评估…")
    t0 = time.time()
    y_pred = clf.predict(x_te)
    infer_time = (time.time() - t0) / len(y_te)

    acc = accuracy_score(y_te, y_pred)
    f1_macro = f1_score(y_te, y_pred, average="macro")
    f1_weight = f1_score(y_te, y_pred, average="weighted")
    cm = confusion_matrix(y_te, y_pred)

    print(f"Accuracy: {acc*100:.2f}%")
    print(f"F1-macro: {f1_macro:.4f} | F1-weighted: {f1_weight:.4f}")
    print(f"Inference per sample: {infer_time*1e3:.3f} ms")

    # 保存模型
    model_path = os.path.join(cfg["ckpt_dir"], f"{cfg['model_name']}_best.joblib")
    joblib.dump(clf, model_path)
    print(f"模型已保存至 {model_path}")

    # 记录结果
    summary = os.path.join(cfg["log_dir"], f"{cfg['model_name']}_summary.txt")
    with open(summary, "w") as f:
        f.write(f"Finished @ {datetime.datetime.now()}\n")
        f.write(f"Train time: {train_time:.2f}s | "
                f"Infer/sample: {infer_time*1e3:.3f} ms\n")
        f.write(f"Acc: {acc:.4f}  F1-macro: {f1_macro:.4f}  "
                f"F1-weighted: {f1_weight:.4f}\n")
        f.write(f"RF params: {rf_kw}\n")

if __name__ == "__main__":
    main()
