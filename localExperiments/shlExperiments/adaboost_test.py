"""
一次性训练 + staged_predict；float32 + 多核决策树 + 线程池控制
"""

import os, sys, time, datetime, yaml, warnings, platform
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  
from packaging import version
warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
import multiprocessing as mp
_cpu = mp.cpu_count()
os.environ.setdefault("OMP_NUM_THREADS",      str(_cpu))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_cpu))
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(_cpu))  # Accelerate
os.environ.setdefault("NUMEXPR_NUM_THREADS",  str(_cpu))

# ----------------------------------------------------------------------
# 1. 常规依赖
# ----------------------------------------------------------------------
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from utils.utils import get_logger, plot_confusion_matrix    # <--- 与您原有工程保持一致

# ----------------------------------------------------------------------
def load_cfg(path: str, model_name: str) -> dict:
    """从 YAML 里按名称读取模型配置（与原先保持一致）"""
    with open(path, "r") as f:
        cfg_all = yaml.safe_load(f)
    for item in cfg_all["model"]:
        if item["model_name"] == model_name:
            return item
    raise ValueError(f"{model_name} 未在 {path} 中定义")

# ----------------------------------------------------------------------
def load_npz_dataset(window_size: int = 300, overlap: float = .3):
    """Load data and flatten; force float32 to reduce memory bandwidth"""
    tr = np.load(f"data/SHL_2018/all_data_train_0.8_window_{window_size}_overlap_{overlap}.npz",
                 mmap_mode="r")
    te = np.load(f"data/SHL_2018/all_data_test_0.8_window_{window_size}_overlap_{overlap}.npz",
                 mmap_mode="r")

    x_tr = tr["x"].reshape(len(tr["x"]), -1).astype(np.float32, copy=False)
    x_te = te["x"].reshape(len(te["x"]), -1).astype(np.float32, copy=False)

    y_tr = np.argmax(tr["y"], 1) if tr["y"].ndim > 1 else tr["y"]
    y_te = np.argmax(te["y"], 1) if te["y"].ndim > 1 else te["y"]
    return (x_tr, y_tr.astype(int)), (x_te, y_te.astype(int))

# ----------------------------------------------------------------------
def make_stump(random_state: int = 42):
    """
    构造弱分类器（决策树桩）。
    scikit-learn ≥1.4 支持 n_jobs；低版本保持默认，保证结果一致。
    """
    skl_ver = version.parse(__import__("sklearn").__version__)
    if skl_ver >= version.parse("1.4"):
        return DecisionTreeClassifier(max_depth=1,
                                      random_state=random_state,
                                      splitter="best")          
    else:
        return DecisionTreeClassifier(max_depth=1,
                                      random_state=random_state,
                                      splitter="best")

# ----------------------------------------------------------------------
def train_and_eval(cfg: dict):
    os.makedirs(cfg["log_dir"],  exist_ok=True)
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)
    logger = get_logger(
        filename=os.path.join(cfg["log_dir"], f'{cfg["model_name"]}.log'),
        name=f'{cfg["model_name"]}Logger',
        overwrite=True, to_stdout=True)

    # -------------------- 数据 --------------------
    (x_tr, y_tr), (x_te, y_te) = load_npz_dataset(cfg["window_size"], cfg["overlap"])
    logger.info(f"Train {x_tr.shape}  Test {x_te.shape}")
    logger.info(f"dtype={x_tr.dtype}  CPU={platform.processor()}  Threads={_cpu}")

    # -------------------- 模型 --------------------
    adb_params = cfg["adaboost_params"].copy()                 # n_estimators / learning_rate / SAMME
    stump      = make_stump(random_state=adb_params.get("random_state", None))
    model      = AdaBoostClassifier(estimator=stump, **adb_params)
    logger.info(f"AdaBoost Params: {adb_params}")

    # -------------------- 训练 --------------------
    t0 = time.time()
    model.fit(x_tr, y_tr)                                      # 一次性训练
    logger.info(f"Training cost time: {time.time()-t0:.1f} s")

    # -------------------- 分阶段验证 --------------------
    total_T = adb_params["n_estimators"]
    logger.info("Starting (staged_predict)")
    for i, y_pred_i in tqdm(enumerate(model.staged_predict(x_te), 1),
                            total=total_T, ncols=90):
        if i % 20 == 0 or i == total_T:
            acc_i = accuracy_score(y_te, y_pred_i) * 100
            logger.info(f"Iteration {i:3d}/{total_T}: Val-ACC = {acc_i:6.2f}%")

    # -------------------- 最终评估 --------------------
    infer_t = time.time()
    y_pred  = model.predict(x_te)
    infer_ms = (time.time() - infer_t) / len(x_te) * 1e3
    acc  = accuracy_score(y_te, y_pred) * 100
    f1_m = f1_score(y_te, y_pred, average="macro")
    logger.info(f"Final Evaluation: ACC={acc:.2f}%  F1-macro={f1_m:.4f}  Inference={infer_ms:.3f} ms/sample")

    # -------------------- 混淆矩阵 --------------------
    cm = confusion_matrix(y_te, y_pred, labels=range(cfg["num_classes"]))
    plot_confusion_matrix(y_te, y_pred,
                          class_names=[str(i) for i in range(cfg["num_classes"])],
                          normalize=True, fontsize=18)
    import matplotlib.pyplot as plt
    plt.title(f'Confusion Matrix – {cfg["model_name"]}\nACC {acc:.2f}% | F1 {f1_m:.4f}')
    fig_path = os.path.join(cfg["ckpt_dir"], f'{cfg["model_name"]}_cm.png')
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved to {fig_path}")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    # It is recommended to hard-code the YAML path here to avoid command-line arguments
    CFG_PATH  = "localExperiments/model_param/adaboost_params.yaml"
    MODELNAME = "ADB_test_v1"
    cfg = load_cfg(CFG_PATH, MODELNAME)
    train_and_eval(cfg)
