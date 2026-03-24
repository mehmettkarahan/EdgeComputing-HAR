import os
import time
import threading
import psutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras.applications.nasnet import preprocess_input as nasnet_preprocess
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer

# --- CONFIGURATION ---
BASE_PATH = "../dataset"
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train")
TRAIN_CSV = os.path.join(BASE_PATH, "Training_set.csv")
OUTPUT_DIR = "../results/nasnet_har"

BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 10
IMAGE_SIZE = (224, 224)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- SYSTEM MONITOR ---
class SystemMonitor(threading.Thread):
    def __init__(self, interval=1):
        threading.Thread.__init__(self)
        self.interval = interval
        self.running = True
        self.stats = {
            "cpu_usage": [],
            "ram_usage": [],
            "temperature": []
        }

    def get_cpu_temp(self):
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                temp = float(f.read()) / 1000.0
            return temp
        except:
            return 0.0

    def run(self):
        while self.running:
            self.stats["cpu_usage"].append(psutil.cpu_percent())
            self.stats["ram_usage"].append(psutil.virtual_memory().percent)
            self.stats["temperature"].append(self.get_cpu_temp())
            time.sleep(self.interval)

    def stop(self):
        self.running = False

# --- DATA PREPARATION ---
print("[INFO] Loading and splitting the dataset...")
try:
    df = pd.read_csv(TRAIN_CSV)
    df['label'] = df['label'].astype(str)
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

    print(f"[INFO] Total Samples: {len(df)}")
    print(f"[INFO] Training Set: {len(train_df)}")
    print(f"[INFO] Testing Set: {len(test_df)}")
except Exception as e:
    print(f"[ERROR] Could not read CSV: {e}")
    exit()

NUM_CLASSES = len(df['label'].unique())
print(f"[INFO] Number of Classes: {NUM_CLASSES}")

# --- MODEL BUILDER ---
def build_model(model_name, num_classes):
    if model_name == "NASNetMobile":
        base_model = NASNetMobile(weights='imagenet', include_top=False, input_shape=IMAGE_SIZE + (3,))
        preprocess_func = nasnet_preprocess
    else:
        raise ValueError("Unknown model name!")

    # Freeze base model
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model, preprocess_func

# --- MAIN TRAINING LOOP ---
models_to_train = ["NASNetMobile"]
final_results = []

for model_name in models_to_train:
    print("\n" + "=" * 40)
    print(f"[INFO] Starting Training for {model_name}...")
    print("=" * 40)

    model_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)

    model, preprocess_func = build_model(model_name, num_classes=NUM_CLASSES)

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_func,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_func
    )

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=TRAIN_IMG_DIR,
        x_col="filename",
        y_col="label",
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=TRAIN_IMG_DIR,
        x_col="filename",
        y_col="label",
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    # ---- TRAINING ----
    monitor = SystemMonitor()
    monitor.start()

    start_training = time.time()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
        ModelCheckpoint(os.path.join(model_dir, f"{model_name}_best.keras"), save_best_only=True)
    ]

    history = model.fit(
        train_generator,
        validation_data=test_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    total_train_time = time.time() - start_training
    monitor.stop()
    monitor.join()

    # --- PLOTTING ---
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'{model_name} Accuracy')
    plt.legend()
    plt.savefig(os.path.join(model_dir, "accuracy.jpg"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} Loss')
    plt.legend()
    plt.savefig(os.path.join(model_dir, "loss.jpg"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(monitor.stats['temperature'], label="Temperature")
    plt.plot(monitor.stats['cpu_usage'], label="CPU %", alpha=0.4)
    plt.legend()
    plt.title("Raspberry Pi 5 System Stats")
    plt.savefig(os.path.join(model_dir, "system_stats.jpg"))
    plt.close()

    # --- TESTING & METRICS ---
    test_generator.reset()
    t1 = time.time()
    predictions = model.predict(test_generator, verbose=1)
    t2 = time.time()

    inf_total = t2 - t1
    fps = len(test_df) / inf_total
    inf_ms = (inf_total / len(test_df)) * 1000

    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    labels = list(test_generator.class_indices.keys())

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"{model_name} Confusion Matrix")
    plt.savefig(os.path.join(model_dir, "confusion_matrix.jpg"))
    plt.close()

    lb = LabelBinarizer()
    lb.fit(y_true)
    y_true_bin = lb.transform(y_true)

    plt.figure(figsize=(12, 10))
    for i, label in enumerate(labels):
        try:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], predictions[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{label} AUC={roc_auc:.2f}')
        except:
            pass
    plt.legend()
    plt.title(f"{model_name} ROC-AUC")
    plt.savefig(os.path.join(model_dir, "roc_auc.jpg"))
    plt.close()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    max_temp = max(monitor.stats['temperature']) if monitor.stats['temperature'] else 0

    report = f"""
Model: {model_name}
------------------------------------
Accuracy: {acc:.4f}
Precision: {prec:.4f}
Recall: {rec:.4f}
F1 Score: {f1:.4f}

FPS: {fps:.2f}
Avg Inference: {inf_ms:.2f} ms
Training Time: {total_train_time:.2f} s
Max Temp: {max_temp:.2f} C
"""

    with open(os.path.join(model_dir, "report.txt"), "w") as f:
        f.write(report)
        f.write("\n\n")
        f.write(classification_report(y_true, y_pred, target_names=labels, zero_division=0))

    final_results.append({
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "FPS": fps,
        "Avg_ms": inf_ms,
        "Train_s": total_train_time,
        "Max_Temp": max_temp
    })

results_df = pd.DataFrame(final_results)
results_df.to_csv(os.path.join(OUTPUT_DIR, "nasnet_results.csv"), index=False)

print("\n[FINISHED] All processes completed successfully!")