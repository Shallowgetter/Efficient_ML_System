# svm_test.py
# Traditional machine-learning baseline (SVM) for 3-D magnetic vehicle classification
# ------------------------------------------------------------------------------

import os, random, sys, time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)
from sklearn.preprocessing import StandardScaler

# ---------- utils (same path as before) ------------------------------------
current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from utils.utils import get_logger, plot_confusion_matrix

# ---------- 0. directories & logger ---------------------------------------
RESULT_DIR     = "localExperiments/geoMagExperiments/model_result/svmModelResult"
os.makedirs(RESULT_DIR, exist_ok=True)

logger = get_logger(
    filename=os.path.join(RESULT_DIR, "svm_experiment.log"),
    name="SVM_Experiment",
    overwrite=True,
    to_stdout=True,
    level="INFO"
)

# ---------- 1. reproducibility --------------------------------------------
SEED = 10
random.seed(SEED)
np.random.seed(SEED)

logger.info(f"Random seed set to {SEED}")

# ---------- 2. load dataset ------------------------------------------------
DATA_PATH = "data/geoMag/geoMagDataset.npz"
logger.info(f"Loading dataset from: {DATA_PATH}")

data   = np.load(DATA_PATH)
X_train, y_train = data["X_train"], data["y_train"]   # SMOTE-balanced train set
X_test,  y_test  = data["X_test"],  data["y_test"]

logger.info(f"Shapes  Train:{X_train.shape}  Test:{X_test.shape}")  # (621,621) / (113,621)

# ---------- 3. scale features (StandardScaler recommended for SVM) --------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ---------- 4. fit SVM -----------------------------------------------------
logger.info("Training SVM with optimum hyper-parameters "
            "(C=100, gamma=0.001, kernel='sigmoid')")

start = time.time()
svm_clf = SVC(C=100, gamma=0.001, kernel="sigmoid", probability=False, random_state=SEED)
svm_clf.fit(X_train_scaled, y_train)
train_time = time.time() - start
logger.info(f"SVM training finished in {train_time:.2f} s")

# ---------- 5. evaluation --------------------------------------------------
logger.info("Evaluating SVM model on test set")

# 预热推理（避免第一次推理的初始化开销）
for _ in range(5):
    _ = svm_clf.predict(X_test_scaled[:1])

# 计算推理时间
inference_times = []
total_samples = len(X_test_scaled)

# 对每个样本单独计时以获得准确的推理时间
for i in range(total_samples):
    start_time = time.time()
    _ = svm_clf.predict(X_test_scaled[i:i+1])
    end_time = time.time()
    inference_times.append(end_time - start_time)

# 获得预测结果
y_pred = svm_clf.predict(X_test_scaled)

# 计算平均推理时间
total_inference_time = sum(inference_times)
avg_inference_time_per_sample = total_inference_time / total_samples

# 记录推理时间信息
logger.info(f"SVM Inference Timing:")
logger.info(f"  Total inference time: {total_inference_time:.4f} seconds")
logger.info(f"  Average time per sample: {avg_inference_time_per_sample*1000:.4f} ms")
logger.info(f"  Total samples: {total_samples}")
logger.info(f"  Throughput: {total_samples/total_inference_time:.2f} samples/second")

acc = accuracy_score(y_test, y_pred)
pr  = precision_score(y_test, y_pred, average="weighted", zero_division=0)
rc  = recall_score(   y_test, y_pred, average="macro",    zero_division=0)
f1  = f1_score(       y_test, y_pred, average="weighted", zero_division=0)

logger.info(f"SVM Final Results | Acc:{acc:.4f}  Prec:{pr:.4f}  Recall:{rc:.4f}  F1:{f1:.4f}")
logger.info(f"SVM Average Inference Time: {avg_inference_time_per_sample*1000:.4f} ms per sample")

# classification report
rep = classification_report(y_test, y_pred, digits=4)
with open(os.path.join(RESULT_DIR, "svm_classification_report.txt"), "w") as f:
    f.write("SVM Classification Report\n")
    f.write(rep)

# ---------- 6. confusion matrix plot --------------------------------------
class_names = ["Light", "Medium", "Heavy"]
cm_fig = plot_confusion_matrix(
    test_y=y_test,
    pred_y=y_pred,
    class_names=class_names,
    normalize=True,
    fontsize=16,
    vmin=0,
    vmax=1,
    axis=1
)
plt.title("SVM Confusion Matrix", fontsize=18, pad=20)
plt.tight_layout()
cm_path = os.path.join(RESULT_DIR, "svm_confusion_matrix.png")
cm_fig.get_figure().savefig(cm_path, dpi=300)
plt.close()
logger.info(f"SVM confusion matrix saved to: {cm_path}")

# ---------- 7. metrics summary --------------------------------------------
with open(os.path.join(RESULT_DIR, "final_metrics.txt"), "w") as f:
    f.write("SVM Model Final Metrics\n")
    f.write("="*50 + "\n")
    f.write(f"SVM | Acc:{acc:.4f}  Prec:{pr:.4f}  Recall:{rc:.4f}  F1:{f1:.4f}\n")
    f.write(f"SVM | Average Inference Time: {avg_inference_time_per_sample*1000:.4f} ms per sample\n")
    f.write(f"SVM | Throughput: {1/avg_inference_time_per_sample:.2f} samples/second\n")
    f.write(f"SVM | Training Time: {train_time:.2f} seconds\n")

logger.info("SVM experiment completed successfully!")
logger.info(f"All results saved in: {RESULT_DIR}")

