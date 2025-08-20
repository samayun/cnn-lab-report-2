import os
import sys
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.applications import InceptionV3, ResNet50, VGG16, DenseNet121, MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

tf.config.run_functions_eagerly(True)

class XRayClassifier:
    def __init__(self, config=None):
        # Default hyperparameters
        default_config = {
            'img_size': (224, 224),
            'batch_size': 16,
            'epochs': 3,
            'learning_rate': 0.001,
            'dropout_rate': 0.5,
            'use_subset': 0.02,  # 2% of data for reasonable training time
            'models': ['InceptionV3', 'ResNet50', 'VGG16'],
            'patience': 2,
            'min_lr': 1e-7,
            'lr_factor': 0.5
        }
        
        self.config = {**default_config, **(config or {})}
        
        # Setup logging
        self.setup_logging()
        
        self.log_hyperparameters()
        
        self.results = {}
        self.training_histories = {}
        
    def setup_logging(self):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"log-{timestamp}.log"
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging initialized - Log file: {log_filename}")
        
    def log_hyperparameters(self):
        self.logger.info("="*60)
        self.logger.info("HYPERPARAMETER CONFIGURATION")
        self.logger.info("="*60)
        for key, value in self.config.items():
            self.logger.info(f"{key.upper().replace('_', ' ')}: {value}")
        self.logger.info("="*60)
        
    def load_data(self):
        self.logger.info("Loading dataset...")
        
        data_dir = 'Augmented Dataset'
        fractured_dir = os.path.join(data_dir, 'Fractured')
        non_fractured_dir = os.path.join(data_dir, 'Non-Fractured')
        
        if not os.path.exists(data_dir):
            self.logger.error(f"Dataset directory not found: {data_dir}")
            return None, None
            
        fractured_files = [f for f in os.listdir(fractured_dir) if f.endswith('.jpg')]
        non_fractured_files = [f for f in os.listdir(non_fractured_dir) if f.endswith('.jpg')]
        
        # Apply subset
        n_fractured = int(len(fractured_files) * self.config['use_subset'])
        n_non_fractured = int(len(non_fractured_files) * self.config['use_subset'])
        
        fractured_files = fractured_files[:n_fractured]
        non_fractured_files = non_fractured_files[:n_non_fractured]
        
        file_paths, labels = [], []
        
        for file in fractured_files:
            file_paths.append(os.path.join(fractured_dir, file))
            labels.append(1)
            
        for file in non_fractured_files:
            file_paths.append(os.path.join(non_fractured_dir, file))
            labels.append(0)
            
        self.logger.info(f"Dataset loaded: {len(file_paths)} images")
        self.logger.info(f"Fractured: {sum(labels)} | Non-fractured: {len(labels) - sum(labels)}")
        
        return file_paths, labels
        
    def create_data_generators(self, file_paths, labels):
        self.logger.info("Creating data generators...")
        
        X_train_paths, X_test_paths, y_train, y_test = train_test_split(
            file_paths, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Only rescaling for test data
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        train_df = pd.DataFrame({
            'filename': X_train_paths,
            'class': ['fractured' if label == 1 else 'non_fractured' for label in y_train]
        })
        
        test_df = pd.DataFrame({
            'filename': X_test_paths,
            'class': ['fractured' if label == 1 else 'non_fractured' for label in y_test]
        })
        
        train_gen = train_datagen.flow_from_dataframe(
            train_df, x_col='filename', y_col='class',
            target_size=self.config['img_size'], 
            batch_size=self.config['batch_size'], 
            class_mode='binary'
        )
        
        test_gen = test_datagen.flow_from_dataframe(
            test_df, x_col='filename', y_col='class',
            target_size=self.config['img_size'], 
            batch_size=self.config['batch_size'], 
            class_mode='binary', 
            shuffle=False
        )
        
        self.logger.info(f"Training samples: {len(train_df)}")
        self.logger.info(f"Test samples: {len(test_df)}")
        
        return train_gen, test_gen, test_df
        
    def create_model(self, model_name):
        self.logger.info(f"Creating {model_name} model...")
        
        base_models = {
            'InceptionV3': InceptionV3(weights='imagenet', include_top=False, input_shape=(*self.config['img_size'], 3)),
            'ResNet50': ResNet50(weights='imagenet', include_top=False, input_shape=(*self.config['img_size'], 3)),
            'VGG16': VGG16(weights='imagenet', include_top=False, input_shape=(*self.config['img_size'], 3)),
            'DenseNet121': DenseNet121(weights='imagenet', include_top=False, input_shape=(*self.config['img_size'], 3)),
            'MobileNetV2': MobileNetV2(weights='imagenet', include_top=False, input_shape=(*self.config['img_size'], 3))
        }
        
        base_model = base_models[model_name]
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(self.config['dropout_rate']),
            layers.Dense(128, activation='relu'),
            layers.Dropout(self.config['dropout_rate'] * 0.6),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.config['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    def train_model(self, model_name, train_gen, val_gen):
        self.logger.info(f"Training {model_name}...")
        
        model = self.create_model(model_name)
        
        callbacks = [
            EarlyStopping(
                patience=self.config['patience'], 
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                factor=self.config['lr_factor'], 
                patience=1, 
                min_lr=self.config['min_lr'],
                verbose=1
            )
        ]
        
        history = model.fit(
            train_gen,
            epochs=self.config['epochs'],
            callbacks=callbacks,
            verbose=1
        )
        
        self.training_histories[model_name] = history
        self.logger.info(f"{model_name} training completed")
        
        return model
        
    def evaluate_model(self, model, model_name, test_gen):
        self.logger.info(f"Evaluating {model_name}...")
        
        test_gen.reset()
        predictions = model.predict(test_gen, verbose=0)
        y_pred = (predictions > 0.5).astype(int)
        y_true = test_gen.classes
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary')
        recall = recall_score(y_true, y_pred, average='binary')
        f1 = f1_score(y_true, y_pred, average='binary')
        cm = confusion_matrix(y_true, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'predictions': predictions,
            'y_pred': y_pred,
            'y_true': y_true
        }
        
        self.logger.info(f"{model_name} Results:")
        self.logger.info(f"  Accuracy:  {accuracy:.4f}")
        self.logger.info(f"  Precision: {precision:.4f}")
        self.logger.info(f"  Recall:    {recall:.4f}")
        self.logger.info(f"  F1-Score:  {f1:.4f}")
        
        return metrics
        
    def plot_sample_predictions(self, test_df, model_name, metrics, n_samples=8):
        self.logger.info(f"Generating sample predictions plot for {model_name}...")
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f'{model_name} - Sample vs Predicted Results', fontsize=16)
        
        sample_indices = np.random.choice(len(test_df), n_samples, replace=False)
        
        for i, idx in enumerate(sample_indices):
            row = i // 4
            col = i % 4
            
            img_path = test_df.iloc[idx]['filename']
            actual = metrics['y_true'][idx]
            predicted = metrics['y_pred'][idx]
            confidence = metrics['predictions'][idx][0]
            
            try:
                img = load_img(img_path, target_size=self.config['img_size'], color_mode='grayscale')
                axes[row, col].imshow(img, cmap='gray')
            except Exception as e:
                self.logger.warning(f"Could not load image {img_path}: {e}")
                axes[row, col].text(0.5, 0.5, 'Image\nNot\nAvailable', 
                                  ha='center', va='center', transform=axes[row, col].transAxes)
            
            title = f'Actual: {"Fractured" if actual == 1 else "Normal"}\n'
            title += f'Pred: {"Fractured" if predicted == 1 else "Normal"}\n'
            title += f'Conf: {confidence:.3f}'
            axes[row, col].set_title(title, fontsize=10)
            axes[row, col].axis('off')
            
            # Color border based on correctness
            color = 'green' if actual == predicted else 'red'
            for spine in axes[row, col].spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
                spine.set_visible(True)
        
        plt.tight_layout()
        filename = f'{model_name}_sample_predictions.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Sample predictions saved: {filename}")
        
    def plot_confusion_matrix(self, model_name, metrics):
        self.logger.info(f"Generating confusion matrix for {model_name}...")
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Non-Fractured', 'Fractured'],
                   yticklabels=['Non-Fractured', 'Fractured'])
        plt.title(f'{model_name} Confusion Matrix\n'
                 f'Accuracy: {metrics["accuracy"]:.4f} | F1-Score: {metrics["f1_score"]:.4f}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        filename = f'{model_name}_confusion_matrix.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Confusion matrix saved: {filename}")
        
    def plot_model_comparison(self):
        self.logger.info("Generating model comparison plots...")
        
        models = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        colors = ['skyblue', 'lightgreen', 'salmon', 'gold', 'lightcoral']
        
        for i, metric in enumerate(metrics):
            row = i // 2
            col = i % 2
            
            values = [self.results[model][metric] for model in models]
            bars = axes[row, col].bar(models, values, color=colors[:len(models)])
            
            axes[row, col].set_title(f'{metric.capitalize().replace("_", "-")}')
            axes[row, col].set_ylim(0, 1)
            axes[row, col].set_ylabel('Score')
            axes[row, col].tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars, values):
                axes[row, col].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                   f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        filename = 'model_comparison.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Model comparison saved: {filename}")
        
    def log_evaluation_matrices(self):
        self.logger.info("")
        self.logger.info("="*60)
        self.logger.info("EVALUATION MATRICES")
        self.logger.info("="*60)
        
        for model_name, metrics in self.results.items():
            self.logger.info(f"\n{model_name}:")
            self.logger.info("-" * 30)
            self.logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
            self.logger.info(f"Precision: {metrics['precision']:.4f}")
            self.logger.info(f"Recall:    {metrics['recall']:.4f}")
            self.logger.info(f"F1-Score:  {metrics['f1_score']:.4f}")
            
        # Classification result comparison
        self.logger.info("")
        self.logger.info("="*60)
        self.logger.info("CLASSIFICATION RESULT COMPARISON")
        self.logger.info("="*60)
        
        sorted_models = sorted(self.results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        for rank, (model_name, metrics) in enumerate(sorted_models, 1):
            self.logger.info(f"{rank}. {model_name:15} - Accuracy: {metrics['accuracy']:.4f}")
            
    def run_experiment(self):
        self.logger.info("Starting X-Ray Fracture Classification Experiment")
        self.logger.info("="*60)
        
        # Load data
        file_paths, labels = self.load_data()
        if file_paths is None:
            self.logger.error("Failed to load data. Exiting.")
            return None
            
        # Create data generators
        train_gen, test_gen, test_df = self.create_data_generators(file_paths, labels)
        
        # Train and evaluate each model
        for model_name in self.config['models']:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Processing {model_name}")
            self.logger.info(f"{'='*60}")
            
            try:
                # Train model
                model = self.train_model(model_name, train_gen, test_gen)
                
                # Evaluate model
                metrics = self.evaluate_model(model, model_name, test_gen)
                self.results[model_name] = metrics
                
                # Generate visualizations
                self.plot_sample_predictions(test_df, model_name, metrics)
                self.plot_confusion_matrix(model_name, metrics)
                
            except Exception as e:
                self.logger.error(f"Error processing {model_name}: {e}")
                continue
        
        if self.results:
            # Generate comparison plots
            self.plot_model_comparison()
            
            # Log final results
            self.log_evaluation_matrices()
            
            self.logger.info("\n🎯 Experiment completed successfully!")
            self.logger.info("Generated files:")
            self.logger.info("- Sample prediction plots for each model")
            self.logger.info("- Confusion matrices for each model")  
            self.logger.info("- Model comparison chart")
            self.logger.info("- Detailed log file with all results")
        else:
            self.logger.error("No models completed successfully!")
            
        return self.results

if __name__ == "__main__":
    # Custom configuration (optional)
    custom_config = {
        'epochs': 2,
        'use_subset': 0.015,  # 1.5% of data
        'models': ['InceptionV3', 'ResNet50', 'VGG16']
    }
    
    classifier = XRayClassifier(config=custom_config)
    results = classifier.run_experiment()