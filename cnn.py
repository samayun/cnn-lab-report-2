import argparse
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


# -----------------------------
# Configuration dataclass
# -----------------------------


@dataclass
class TrainConfig:
	data_dir: str
	output_dir: str
	models: List[str]
	batch_size: int
	epochs: int
	lr: float
	seed: int
	feature_extract: bool
	val_split: float
	test_split: float


# -----------------------------
# Utils
# -----------------------------


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	tf.random.set_seed(seed)


def ensure_dir(path: str) -> None:
	os.makedirs(path, exist_ok=True)


# -----------------------------
# Data handling
# -----------------------------


ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images_and_labels(data_dir: str) -> Tuple[List[str], List[int], Dict[int, str]]:
	root = Path(data_dir)
	class_names = sorted([p.name for p in root.iterdir() if p.is_dir()])
	idx_to_class = {i: name for i, name in enumerate(class_names)}
	class_to_idx = {v: k for k, v in idx_to_class.items()}

	paths: List[str] = []
	labels: List[int] = []
	for class_name in class_names:
		class_dir = root / class_name
		for p in class_dir.iterdir():
			if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
				paths.append(str(p))
				labels.append(class_to_idx[class_name])
	return paths, labels, idx_to_class


def split_indices(n: int, val_split: float, test_split: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	assert 0 < val_split < 1 and 0 < test_split < 1 and val_split + test_split < 1
	idx = np.arange(n)
	rng = np.random.default_rng(seed)
	rng.shuffle(idx)
	val_len = int(n * val_split)
	test_len = int(n * test_split)
	train_idx = idx[: n - val_len - test_len]
	val_idx = idx[n - val_len - test_len : n - test_len]
	test_idx = idx[n - test_len :]
	return train_idx, val_idx, test_idx


def build_dataset(paths: List[str], labels: List[int], image_size: int, batch_size: int, shuffle: bool, seed: int) -> tf.data.Dataset:
	def _load(path, label):
		img = tf.io.read_file(path)
		img = tf.io.decode_image(img, channels=3, expand_animations=False)
		img = tf.image.resize(img, [image_size, image_size])
		img = tf.cast(img, tf.float32)
		return img, tf.cast(label, tf.int32)

	ds = tf.data.Dataset.from_tensor_slices((paths, labels))
	if shuffle:
		ds = ds.shuffle(buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True)
	ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
	ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
	return ds


# -----------------------------
# Models
# -----------------------------


def get_model_spec(name: str) -> Tuple[Callable, Callable, int]:
	lname = name.lower()
	if lname in {"resnet50", "resnet-50"}:
		from tensorflow.keras.applications import resnet50

		return resnet50.ResNet50, resnet50.preprocess_input, 224
	if lname in {"densenet121", "densenet-121"}:
		from tensorflow.keras.applications import densenet

		return densenet.DenseNet121, densenet.preprocess_input, 224
	if lname in {"efficientnet_b0", "efficientnet-b0", "efficientnet"}:
		from tensorflow.keras.applications import efficientnet

		return efficientnet.EfficientNetB0, efficientnet.preprocess_input, 224
	if lname in {"mobilenet_v3_large", "mobilenet-v3-large", "mobilenet"}:
		from tensorflow.keras.applications import mobilenet_v3

		return mobilenet_v3.MobileNetV3Large, mobilenet_v3.preprocess_input, 224
	if lname in {"inception_v3", "inception-v3", "inception"}:
		from tensorflow.keras.applications import inception_v3

		return inception_v3.InceptionV3, inception_v3.preprocess_input, 299
	if lname in {"vgg16", "vgg-16"}:
		from tensorflow.keras.applications import vgg16

		return vgg16.VGG16, vgg16.preprocess_input, 224
	raise ValueError(f"Unsupported model: {name}")


def build_model(model_name: str, num_classes: int, feature_extract: bool, image_size: int, lr: float) -> tf.keras.Model:
	constructor, preprocess_input, _ = get_model_spec(model_name)

	data_augmentation = tf.keras.Sequential(
		[
			tf.keras.layers.RandomFlip("horizontal"),
			tf.keras.layers.RandomRotation(0.1),
			tf.keras.layers.RandomZoom(0.1),
		],
		name="augmentation",
	)

	inputs = tf.keras.Input(shape=(image_size, image_size, 3))
	x = data_augmentation(inputs)
	x = tf.keras.layers.Lambda(preprocess_input, name="preprocess")(x)

	base_model = constructor(include_top=False, weights="imagenet", input_shape=(image_size, image_size, 3))
	base_model.trainable = not feature_extract

	# Bonus fine-tune: unfreeze last 50 layers for EfficientNet when not feature_extract
	if not feature_extract and model_name.lower() in {"efficientnet_b0", "efficientnet-b0", "efficientnet"}:
		for layer in base_model.layers[:-50]:
			layer.trainable = False
		for layer in base_model.layers[-50:]:
			layer.trainable = True

	x = base_model(x)
	x = tf.keras.layers.GlobalAveragePooling2D()(x)
	x = tf.keras.layers.Dropout(0.2)(x)
	outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
	model = tf.keras.Model(inputs, outputs)

	optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
	model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
	return model


# -----------------------------
# Training and evaluation
# -----------------------------


def train_and_evaluate(model_name: str, cfg: TrainConfig) -> Dict:
	constructor, preprocess_input, image_size = get_model_spec(model_name)

	paths, labels, idx_to_class = list_images_and_labels(cfg.data_dir)
	num_classes = len(idx_to_class)
	train_idx, val_idx, test_idx = split_indices(len(paths), cfg.val_split, cfg.test_split, cfg.seed)

	def select(idx: np.ndarray) -> Tuple[List[str], List[int]]:
		return [paths[i] for i in idx], [labels[i] for i in idx]

	train_paths, train_labels = select(train_idx)
	val_paths, val_labels = select(val_idx)
	test_paths, test_labels = select(test_idx)

	train_ds = build_dataset(train_paths, train_labels, image_size, cfg.batch_size, shuffle=True, seed=cfg.seed)
	val_ds = build_dataset(val_paths, val_labels, image_size, cfg.batch_size, shuffle=False, seed=cfg.seed)
	test_ds = build_dataset(test_paths, test_labels, image_size, cfg.batch_size, shuffle=False, seed=cfg.seed)

	model = build_model(model_name, num_classes=num_classes, feature_extract=cfg.feature_extract, image_size=image_size, lr=cfg.lr)

	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	model_dir = os.path.join(cfg.output_dir, "models")
	report_dir = os.path.join(cfg.output_dir, "reports")
	ensure_dir(model_dir)
	ensure_dir(report_dir)

	ckpt_path = os.path.join(model_dir, f"{model_name}_best_{timestamp}.weights.h5")
	callbacks = [
		tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", mode="max", save_best_only=True, save_weights_only=True, verbose=1),
		tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
		tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1),
	]

	history = model.fit(train_ds, validation_data=val_ds, epochs=cfg.epochs, callbacks=callbacks, verbose=1).history

	test_loss, test_acc = model.evaluate(test_ds, verbose=0)
	y_prob = model.predict(test_ds, verbose=0)
	y_pred = np.argmax(y_prob, axis=1)
	y_true = np.array(test_labels)

	average = "binary" if num_classes == 2 else "weighted"
	precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
	cm = confusion_matrix(y_true, y_pred).tolist()

	report = {
		"model": model_name,
		"input_size": image_size,
		"feature_extract": cfg.feature_extract,
		"epochs": cfg.epochs,
		"batch_size": cfg.batch_size,
		"best_val_acc": max(history.get("val_accuracy", [0.0])),
		"test": {
			"loss": float(test_loss),
			"accuracy": float(test_acc),
			"precision": float(precision),
			"recall": float(recall),
			"f1": float(f1),
			"confusion_matrix": cm,
		},
		"classes": {int(k): v for k, v in idx_to_class.items()},
		"history": {k: [float(x) for x in v] for k, v in history.items()},
		"checkpoint": ckpt_path,
	}

	report_path = os.path.join(report_dir, f"{model_name}_{timestamp}.json")
	with open(report_path, "w") as f:
		json.dump(report, f, indent=2)

	return report


# -----------------------------
# CLI
# -----------------------------


def parse_args() -> TrainConfig:
	parser = argparse.ArgumentParser(description="Train multiple pretrained CNNs on X-ray dataset (TensorFlow)")
	default_root = os.path.dirname(os.path.abspath(__file__))
	default_data = os.path.join(default_root, "Augmented Dataset")
	default_output = os.path.join(default_root, "outputs")

	parser.add_argument("--data-dir", type=str, default=default_data, help="Path to dataset root containing class subfolders")
	parser.add_argument("--output-dir", type=str, default=default_output, help="Where to save models and reports")
	parser.add_argument(
		"--models",
		type=str,
		default="resnet50,densenet121,efficientnet_b0,mobilenet_v3_large,inception_v3,vgg16",
		help="Comma-separated list of models to train",
	)
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--epochs", type=int, default=8)
	parser.add_argument("--lr", type=float, default=3e-4)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--feature-extract", action="store_true", help="Freeze backbone and train classifier only")
	parser.add_argument("--val-split", type=float, default=0.15, help="Fraction for validation set")
	parser.add_argument("--test-split", type=float, default=0.15, help="Fraction for test set")

	args = parser.parse_args()

	cfg = TrainConfig(
		data_dir=args.data_dir,
		output_dir=args.output_dir,
		models=[m.strip() for m in args.models.split(",") if m.strip()],
		batch_size=args.batch_size,
		epochs=args.epochs,
		lr=args.lr,
		seed=args.seed,
		feature_extract=args.feature_extract,
		val_split=args.val_split,
		test_split=args.test_split,
	)
	return cfg


def main() -> None:
	cfg = parse_args()
	set_seed(cfg.seed)
	ensure_dir(cfg.output_dir)

	print(f"Training models: {', '.join(cfg.models)}")
	print(f"Dataset: {cfg.data_dir}")

	all_reports = []
	for model_name in cfg.models:
		try:
			report = train_and_evaluate(model_name, cfg)
			all_reports.append(report)
		except Exception as e:
			print(f"Error training {model_name}: {e}")

	summary = {
		"generated_at": datetime.now().isoformat(),
		"num_models": len(all_reports),
		"reports": [
			{
				"model": r["model"],
				"best_val_acc": r["best_val_acc"],
				"test_accuracy": r["test"]["accuracy"],
				"checkpoint": r["checkpoint"],
				"report_path": None,
			}
			for r in all_reports
		],
	}

	for entry in summary["reports"]:
		ckpt = entry["checkpoint"]
		base_name = os.path.basename(ckpt)
		stem = base_name.replace("_best_", "_").replace(".weights.h5", ".json")
		entry["report_path"] = os.path.join(cfg.output_dir, "reports", stem)

	with open(os.path.join(cfg.output_dir, "summary.json"), "w") as f:
		json.dump(summary, f, indent=2)


if __name__ == "__main__":
	main()
