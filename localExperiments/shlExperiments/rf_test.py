"""
RandomForest experiment under sk-learn.
Using CPU to train and test the model.
"""

import sys
import os
from typing import Dict, Any
import datetime
import time
import yaml
import argparse
import pickle  # Add this import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import get_logger, save_checkpoint, AverageMeter, plot_confusion_matrix, select_certain_classes
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="RandomForest Training Script")
    parser.add_argument('--config', type=str, default='localExperiments/model_param/rf_params.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model_name', type=str, default='RF_test_v2',
                        help='Model name as specified in the configuration file')
    return parser.parse_args()

def load_config(config_path, model_name):
    """
    Load model configuration from yaml file
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Find the specific model configuration
    for model_config in config['model']:
        if model_config['model_name'] == model_name:
            return model_config
    
    raise ValueError(f"Model configuration for {model_name} not found in {config_path}")

def load_dataset(train_path, val_path):
    train = select_certain_classes(train_path, 
                                   selected_classes= [
        "acc_x", "acc_y", "acc_z",
        "gra_x", "gra_y", "gra_z", 
        "gyr_x", "gyr_y", "gyr_z",
        "lacc_x", "lacc_y", "lacc_z",
        "mag_x", "mag_y", "mag_z",
        "ori_w", "ori_x", "ori_y", "ori_z",
        "pressure"
    ])
    val = select_certain_classes(val_path, 
                                  selected_classes= [
        "acc_x", "acc_y", "acc_z",
        "gra_x", "gra_y", "gra_z", 
        "gyr_x", "gyr_y", "gyr_z",
        "lacc_x", "lacc_y", "lacc_z",
        "mag_x", "mag_y", "mag_z",
        "ori_w", "ori_x", "ori_y", "ori_z",
        "pressure"
    ])

    train_x, train_y = train['x'].astype(np.float32), train['y']
    val_x, val_y = val['x'].astype(np.float32), val['y']

    print(f"Train data shape: {train_x.shape}, Train labels shape: {train_y.shape}")
    print(f"Validation data shape: {val_x.shape}, Validation labels shape: {val_y.shape}")

    return train_x, train_y, val_x, val_y

def _flatten(x: np.ndarray) -> np.ndarray:
    """
    Flatten a 3-D time-series tensor (N, T, C) into shape (N, T × C) so that
    it can be consumed by scikit-learn estimators.

    Parameters
    ----------
    x : np.ndarray
        Input array with shape (n_samples, window_size, n_channels).

    Returns
    -------
    np.ndarray
        Flattened 2-D array.
    """
    return x.reshape(x.shape[0], -1)

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    rf_params: Dict[str, Any],
    logger,
) -> RandomForestClassifier:
    """
    Fit a RandomForest model and log training latency.

    Returns
    -------
    RandomForestClassifier
        Trained model instance.
    """
    logger.info("Fitting the RandomForest model …")
    t0 = time.time()
    model = RandomForestClassifier(**rf_params)
    model.fit(_flatten(X_train), y_train)
    logger.info(f"Model fitted in {time.time() - t0:.2f} seconds.")
    return model


def evaluate(
    model: RandomForestClassifier,
    X_val: np.ndarray,
    y_val: np.ndarray,
    logger=None,
) -> Dict[str, Any]:
    """
    Evaluate the trained model on a validation set.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing accuracy, weighted-F1, confusion matrix and
        average inference latency (ms / sample).
    """
    X_val_f = _flatten(X_val)

    # ----- inference & latency ------------------------------------------------
    t0 = time.time()
    y_pred = model.predict(X_val_f)
    total_inf_time = time.time() - t0
    avg_latency_ms = total_inf_time / len(y_val) * 1000.0

    # ----- metrics ------------------------------------------------------------
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="weighted")
    cm = confusion_matrix(y_val, y_pred)

    if logger is not None:
        logger.info(f"Validation Accuracy       : {acc:.4f}")
        logger.info(f"Validation F1-Score       : {f1:.4f}")
        logger.info(f"Avg inference time / samp : {avg_latency_ms:.3f} ms")

    return {
        "accuracy": acc,
        "f1_weighted": f1,
        "avg_latency_ms": avg_latency_ms,
        "confusion_matrix": cm,
        "predictions": y_pred,
    }



if __name__ == "__main__":
    args = parse_arguments()
    config = load_config(args.config, args.model_name)

    # Set up logging
    log_dir = config['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{args.model_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = get_logger(log_file)

    # Load dataset
    train_path = 'data/SHL_2018/all_data_train_0.8_window_450_overlap_0.0.npz'  
    val_path = 'data/SHL_2018/all_data_test_0.8_window_450_overlap_0.0.npz'      
    train_x, train_y, val_x, val_y = load_dataset(train_path, val_path)

    # Train the RandomForest model using the existing function
    rf_params = config['rf_params']
    logger.info("Training RandomForest model...")
    rf_model = train_random_forest(train_x, train_y, rf_params, logger)

    # Evaluate the model using the existing function
    logger.info("Evaluating RandomForest model...")
    results = evaluate(rf_model, val_x, val_y, logger)

    # Extract results
    accuracy = results['accuracy']
    f1 = results['f1_weighted']
    avg_inference_time_ms = results['avg_latency_ms']
    cm = results['confusion_matrix']
    pred_y = results['predictions']

    # Convert inference time to seconds for consistency with original logging
    avg_inference_time_seconds = avg_inference_time_ms / 1000.0
    logger.info(f"Average Inference Time: {avg_inference_time_seconds:.4f} seconds")

    # Model size
    model_size = rf_model.__sizeof__()
    logger.info(f"Model Size: {model_size / (1024 * 1024):.2f} MB")
    logger.info(f"Model Parameters: {rf_params}")
    logger.info(f"Model Name: {args.model_name}")
    logger.info(f"Configuration: {config}")

    # Create checkpoint directory if it doesn't exist
    ckpt_dir = config['ckpt_dir']
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Save the model using pickle instead of save_checkpoint
    model_path = os.path.join(ckpt_dir, f"{args.model_name}_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(rf_model, f)
    logger.info(f"Model saved to: {model_path}")
    
    # Plot confusion matrix
    class_names = [str(i) for i in range(len(np.unique(val_y)))]
    
    ax = plot_confusion_matrix(
        test_y=val_y, 
        pred_y=pred_y,
        class_names=class_names,
        normalize=True, 
        fontsize=18
    )
    
    import matplotlib.pyplot as plt
    plt.title(f'Confusion Matrix – {args.model_name}\nAccuracy {accuracy:.4f} | F1 {f1:.4f}')
    
    # Create results directory for saving confusion matrix plots
    results_dir = "localExperiments/model_result/rf_confusion_matrix_plots"
    os.makedirs(results_dir, exist_ok=True)
    fig_path = os.path.join(results_dir, f"{args.model_name}_cm.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.show()
    logger.info(f"Confusion matrix saved to {fig_path}")
    
    # Final summary
    logger.info("=" * 50)
    logger.info("TRAINING AND VALIDATION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Validation Accuracy: {accuracy:.4f}")
    logger.info(f"F1 Score (Weighted): {f1:.4f}")
    logger.info(f"Average Inference Time: {avg_inference_time_ms:.3f} ms per sample")
    logger.info(f"Model Size: {model_size / (1024 * 1024):.2f} MB")
    logger.info("=" * 50)
