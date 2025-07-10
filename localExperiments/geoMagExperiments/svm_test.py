# svm_test.py
# Traditional machine-learning baseline (SVM) for 3-D magnetic vehicle classification
# ------------------------------------------------------------------------------

import os, random, sys, time
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)
from sklearn.preprocessing import StandardScaler

# ---------- utils (same path as before) ------------------------------------
current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from utils.utils import get_logger, plot_confusion_matrix

# ---------- 0. Setup directories & logger ----------------------------------
RESULT_DIR = "localExperiments/geoMagExperiments/model_result/svmModelResult"
MODEL_DIR = "localExperiments/geoMagExperiments/model_result/svm_models"
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

logger = get_logger(
    filename=os.path.join(RESULT_DIR, "svm_experiment.log"),
    name="SVM_Experiment",
    level="INFO",
    overwrite=True,
    to_stdout=True
)

# ---------- 1. Reproducibility --------------------------------------------
SEED = 10
random.seed(SEED)
np.random.seed(SEED)

logger.info(f"Running SVM experiment with random seed: {SEED}")
logger.info(f"Results will be saved to: {RESULT_DIR}")
logger.info(f"Models will be saved to: {MODEL_DIR}")

# ---------- 2. Load dataset ------------------------------------------------
DATA_PATH = "data/geoMag/geoMagDataset.npz"
logger.info(f"Loading dataset from: {DATA_PATH}")

data = np.load(DATA_PATH)
X_train, y_train = data["X_train"], data["y_train"]   # SMOTE-balanced train set
X_test,  y_test  = data["X_test"],  data["y_test"]

logger.info(f"Original data shapes - Train: {X_train.shape}, Test: {X_test.shape}")
logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
logger.info(f"Feature dimension: {X_train.shape[1]}")
logger.info(f"Number of classes: {len(np.unique(y_train))}")

# ---------- 3. Feature scaling (StandardScaler recommended for SVM) --------
logger.info("Applying StandardScaler for feature normalization")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Save scaler for future use
scaler_path = os.path.join(MODEL_DIR, "feature_scaler.pkl")
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
logger.info(f"Feature scaler saved to: {scaler_path}")

# ---------- 4. SVM Training ------------------------------------------------
logger.info("=" * 50)
logger.info("TRAINING PHASE")
logger.info("=" * 50)

logger.info("Training SVM with optimized hyperparameters:")
logger.info("  - C=100 (regularization parameter)")
logger.info("  - gamma=0.001 (kernel coefficient)")
logger.info("  - kernel='sigmoid'")
logger.info("  - probability=False (for faster inference)")

start_time = time.time()
svm_clf = SVC(C=100, gamma=0.001, kernel="sigmoid", probability=False, random_state=SEED)
svm_clf.fit(X_train_scaled, y_train)
train_time = time.time() - start_time

logger.info(f"SVM training completed in {train_time:.2f} seconds")
logger.info(f"Number of support vectors: {svm_clf.n_support_}")
logger.info(f"Total support vectors: {sum(svm_clf.n_support_)}")

# Save trained model
model_path = os.path.join(MODEL_DIR, "svm_model_final.pkl")
with open(model_path, 'wb') as f:
    pickle.dump(svm_clf, f)
logger.info(f"Final SVM model saved to: {model_path}")

# ---------- 5. Evaluation --------------------------------------------------
logger.info("=" * 50)
logger.info("EVALUATION PHASE")
logger.info("=" * 50)

logger.info("Evaluating SVM model on test set")

# 预热推理（避免第一次推理的初始化开销）
logger.info("Warming up inference...")
for _ in range(5):
    _ = svm_clf.predict(X_test_scaled[:1])

# 计算推理时间
logger.info("Measuring inference time...")
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

# 计算推理时间统计
total_inference_time = sum(inference_times)
avg_inference_time_per_sample = total_inference_time / total_samples
throughput = total_samples / total_inference_time

# 记录推理时间信息
logger.info(f"SVM Inference Timing:")
logger.info(f"  Total inference time: {total_inference_time:.4f} seconds")
logger.info(f"  Average time per sample: {avg_inference_time_per_sample*1000:.4f} ms")
logger.info(f"  Total samples: {total_samples}")
logger.info(f"  Throughput: {throughput:.2f} samples/second")

# 计算性能指标
acc = accuracy_score(y_test, y_pred)
pr  = precision_score(y_test, y_pred, average="weighted", zero_division=0)
rc  = recall_score(y_test, y_pred, average="macro", zero_division=0)
f1  = f1_score(y_test, y_pred, average="weighted", zero_division=0)

logger.info(f"SVM Final Results:")
logger.info(f"  Accuracy: {acc:.4f}")
logger.info(f"  Precision (weighted): {pr:.4f}")
logger.info(f"  Recall (macro): {rc:.4f}")
logger.info(f"  F1-score (weighted): {f1:.4f}")
logger.info(f"  Latency: {avg_inference_time_per_sample*1000:.2f} ms/sample")
logger.info(f"  Throughput: {throughput:.2f} samples/second")

# 计算模型大小
model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
logger.info(f"  Model size: {model_size_mb:.2f} MB")

# Classification report
class_report = classification_report(y_test, y_pred, digits=4)
logger.info(f"SVM Classification Report:\n{class_report}")

# Save classification report
report_path = os.path.join(RESULT_DIR, "svm_classification_report.txt")
with open(report_path, "w") as f:
    f.write("SVM Classification Report\n")
    f.write("=" * 50 + "\n")
    f.write(class_report)
logger.info(f"Classification report saved to: {report_path}")

# ---------- 6. Confusion Matrix -------------------------------------------
logger.info("Generating confusion matrix")

class_names = ["Light", "Medium", "Heavy"]

plt.figure(figsize=(8, 6))
ax = plot_confusion_matrix(
    test_y=y_test,
    pred_y=y_pred,
    class_names=class_names,
    normalize=True,
    fontsize=16,
    vmin=0,
    vmax=1,
    axis=1
)
plt.title("SVM Confusion Matrix", fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()

cm_path = os.path.join(RESULT_DIR, "svm_confusion_matrix.png")
plt.savefig(cm_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
logger.info(f"SVM confusion matrix saved to: {cm_path}")

# ---------- 7. Training Progress Visualization ----------------------------
logger.info("Generating training summary visualization")

# 创建训练摘要图
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# 支持向量分布
support_vector_counts = svm_clf.n_support_
classes = ['Light', 'Medium', 'Heavy']
ax1.bar(classes, support_vector_counts, color=['#3498db', '#e74c3c', '#f39c12'])
ax1.set_title('Support Vectors by Class', fontweight='bold')
ax1.set_ylabel('Number of Support Vectors')
for i, v in enumerate(support_vector_counts):
    ax1.text(i, v + 0.5, str(v), ha='center', va='bottom')

# 性能指标
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
metrics_values = [acc, pr, rc, f1]
bars = ax2.bar(metrics_names, metrics_values, color=['#2ecc71', '#9b59b6', '#e67e22', '#34495e'])
ax2.set_title('Performance Metrics', fontweight='bold')
ax2.set_ylabel('Score')
ax2.set_ylim(0, 1)
for bar, value in zip(bars, metrics_values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{value:.3f}', ha='center', va='bottom')

# 时间性能
time_metrics = ['Training Time (s)', 'Avg Inference (ms)']
time_values = [train_time, avg_inference_time_per_sample * 1000]
ax3.bar(time_metrics, time_values, color=['#16a085', '#c0392b'])
ax3.set_title('Time Performance', fontweight='bold')
ax3.set_ylabel('Time')
for i, v in enumerate(time_values):
    ax3.text(i, v + max(time_values) * 0.01, f'{v:.2f}', ha='center', va='bottom')

# 模型统计
stats_names = ['Model Size (MB)', 'Total Support Vectors', 'Throughput (samples/s)']
stats_values = [model_size_mb, sum(support_vector_counts), throughput]
# 归一化显示
normalized_stats = [v / max(stats_values) for v in stats_values]
bars = ax4.bar(stats_names, normalized_stats, color=['#8e44ad', '#d35400', '#27ae60'])
ax4.set_title('Model Statistics (Normalized)', fontweight='bold')
ax4.set_ylabel('Normalized Value')
ax4.set_xticklabels(stats_names, rotation=45, ha='right')
for i, (bar, actual_val) in enumerate(zip(bars, stats_values)):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{actual_val:.1f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
summary_path = os.path.join(RESULT_DIR, "svm_training_summary.png")
plt.savefig(summary_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
logger.info(f"Training summary visualization saved to: {summary_path}")

# ---------- 8. Comprehensive Metrics Summary ------------------------------
logger.info("Saving comprehensive metrics summary")

metrics_path = os.path.join(RESULT_DIR, "final_metrics.txt")
with open(metrics_path, "w") as f:
    f.write("SVM Model Final Metrics\n")
    f.write("=" * 50 + "\n")
    
    f.write("TRAINING SUMMARY:\n")
    f.write(f"Training time (s):             {train_time:.2f}\n")
    f.write(f"Support vectors by class:      {list(support_vector_counts)}\n")
    f.write(f"Total support vectors:         {sum(support_vector_counts)}\n")
    f.write(f"Support vector ratio:          {sum(support_vector_counts)/len(X_train):.3f}\n")
    f.write("-" * 30 + "\n")
    
    f.write("PERFORMANCE METRICS:\n")
    f.write(f"Accuracy:                      {acc:.4f}\n")
    f.write(f"Precision (weighted):          {pr:.4f}\n")
    f.write(f"Recall (macro):                {rc:.4f}\n")
    f.write(f"F1-score (weighted):           {f1:.4f}\n")
    f.write("-" * 30 + "\n")
    
    f.write("EFFICIENCY METRICS:\n")
    f.write(f"Latency per sample (ms):       {avg_inference_time_per_sample*1000:.4f}\n")
    f.write(f"Throughput (samples/s):        {throughput:.2f}\n")
    f.write(f"Total inference time (s):      {total_inference_time:.4f}\n")
    f.write(f"Model size (MB):               {model_size_mb:.2f}\n")
    f.write("-" * 30 + "\n")
    
    f.write("HYPERPARAMETERS:\n")
    f.write(f"C (regularization):            {svm_clf.C}\n")
    f.write(f"Gamma:                         {svm_clf.gamma}\n")
    f.write(f"Kernel:                        {svm_clf.kernel}\n")
    f.write(f"Random seed:                   {SEED}\n")

logger.info(f"Final metrics saved to: {metrics_path}")

# ---------- 9. Save Experiment Data ---------------------------------------
logger.info("Saving experiment data")

# 保存预测结果
predictions_path = os.path.join(RESULT_DIR, "predictions.npz")
np.savez(predictions_path, 
         y_true=y_test, 
         y_pred=y_pred,
         inference_times=inference_times)
logger.info(f"Predictions and timing data saved to: {predictions_path}")

# ---------- 10. Final Summary ------------------------------------------
logger.info("=" * 50)
logger.info("SVM experiment completed successfully!")
logger.info(f"All results saved in: {RESULT_DIR}")
logger.info(f"Models saved in: {MODEL_DIR}")
logger.info("Generated files:")
logger.info("- svm_experiment.log (detailed log file)")
logger.info("- svm_model_final.pkl (trained SVM model)")
logger.info("- feature_scaler.pkl (fitted StandardScaler)")
logger.info("- final_metrics.txt (comprehensive summary metrics)")
logger.info("- svm_confusion_matrix.png (confusion matrix)")
logger.info("- svm_training_summary.png (training summary visualization)")
logger.info("- svm_classification_report.txt (detailed classification report)")
logger.info("- predictions.npz (predictions and timing data)")
logger.info("=" * 50)

logger.info("Key Results Summary:")
logger.info(f"- Accuracy: {acc:.4f}")
logger.info(f"- F1-Score: {f1:.4f}")
logger.info(f"- Training Time: {train_time:.2f}s")
logger.info(f"- Inference Latency: {avg_inference_time_per_sample*1000:.2f}ms")
logger.info(f"- Model Size: {model_size_mb:.2f}MB")
logger.info(f"- Support Vector Ratio: {sum(support_vector_counts)/len(X_train):.3f}")
logger.info("=" * 50)

