import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.applications import InceptionV3, ResNet50, VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

tf.config.run_functions_eagerly(True)

class CNN:
    def __init__(self):
        self.img_size = (224, 224)
        self.batch_size = 32
        self.epochs = 1
        self.use_subset = 0.01
        self.learning_rate = 0.001
        self.dropout_rate = 0.5
        
        print("=== HYPERPARAMETER SETUP ===")
        print(f"Image Size: {self.img_size}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Epochs: {self.epochs}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Dropout Rate: {self.dropout_rate}")
        print(f"Data Subset: {self.use_subset*100}%")
        print("="*40)
        
    def load_data(self):
        data_dir = 'Augmented Dataset'
        fractured_dir = os.path.join(data_dir, 'Fractured')
        non_fractured_dir = os.path.join(data_dir, 'Non-Fractured')
        
        fractured_files = [f for f in os.listdir(fractured_dir) if f.endswith('.jpg')][:int(4650*self.use_subset)]
        non_fractured_files = [f for f in os.listdir(non_fractured_dir) if f.endswith('.jpg')][:int(4650*self.use_subset)]
        
        file_paths = []
        labels = []
        
        for file in fractured_files:
            file_paths.append(os.path.join(fractured_dir, file))
            labels.append(1)
            
        for file in non_fractured_files:
            file_paths.append(os.path.join(non_fractured_dir, file))
            labels.append(0)
            
        return file_paths, labels
    
    def create_generators(self, file_paths, labels):
        X_train_paths, X_test_paths, y_train, y_test = train_test_split(
            file_paths, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        datagen = ImageDataGenerator(rescale=1./255)
        
        train_df = pd.DataFrame({
            'filename': X_train_paths,
            'class': ['fractured' if label == 1 else 'non_fractured' for label in y_train]
        })
        
        test_df = pd.DataFrame({
            'filename': X_test_paths,
            'class': ['fractured' if label == 1 else 'non_fractured' for label in y_test]
        })
        
        train_gen = datagen.flow_from_dataframe(
            train_df, x_col='filename', y_col='class',
            target_size=self.img_size, batch_size=self.batch_size, class_mode='binary'
        )
        
        test_gen = datagen.flow_from_dataframe(
            test_df, x_col='filename', y_col='class',
            target_size=self.img_size, batch_size=self.batch_size, class_mode='binary', shuffle=False
        )
        
        return train_gen, test_gen, test_df
    
    def create_model(self, name):
        base_models = {
            'InceptionV3': InceptionV3(weights='imagenet', include_top=False, input_shape=(*self.img_size, 3)),
            'ResNet50': ResNet50(weights='imagenet', include_top=False, input_shape=(*self.img_size, 3)),
            'VGG16': VGG16(weights='imagenet', include_top=False, input_shape=(*self.img_size, 3))
        }
        
        base = base_models[name]
        base.trainable = False
        
        model = models.Sequential([
            base,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer=Adam(self.learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def calculate_metrics(self, y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary')
        recall = recall_score(y_true, y_pred, average='binary')
        f1 = f1_score(y_true, y_pred, average='binary')
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        }
    
    def plot_sample_predictions(self, test_df, y_true, y_pred, y_pred_prob, model_name):
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f'{model_name} - Sample vs Predicted Results', fontsize=16)
        
        sample_indices = np.random.choice(len(test_df), 8, replace=False)
        
        for i, idx in enumerate(sample_indices):
            row = i // 4
            col = i % 4
            
            img_path = test_df.iloc[idx]['filename']
            actual = y_true[idx]
            predicted = y_pred[idx]
            confidence = y_pred_prob[idx][0]
            
            img = load_img(img_path, target_size=self.img_size)
            
            axes[row, col].imshow(img, cmap='gray')
            axes[row, col].set_title(f'Actual: {"Fractured" if actual == 1 else "Normal"}\n'
                                   f'Pred: {"Fractured" if predicted == 1 else "Normal"}\n'
                                   f'Conf: {confidence:.3f}')
            axes[row, col].axis('off')
            
            if actual == predicted:
                for spine in axes[row, col].spines.values():
                    spine.set_edgecolor('green')
                    spine.set_linewidth(3)
            else:
                for spine in axes[row, col].spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(3)
        
        plt.tight_layout()
        plt.savefig(f'{model_name}_sample_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, cm, model_name, metrics):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Non-Fractured', 'Fractured'],
                   yticklabels=['Non-Fractured', 'Fractured'])
        plt.title(f'{model_name} Confusion Matrix\n'
                 f'Accuracy: {metrics["accuracy"]:.4f} | F1: {metrics["f1_score"]:.4f}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(f'{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(self, results):
        models = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        for i, metric in enumerate(metrics):
            row = i // 2
            col = i % 2
            
            values = [results[model][metric] for model in models]
            bars = axes[row, col].bar(models, values, color=['skyblue', 'lightgreen', 'salmon'][:len(models)])
            
            axes[row, col].set_title(f'{metric.capitalize()}')
            axes[row, col].set_ylim(0, 1)
            axes[row, col].set_ylabel('Score')
            
            for bar, value in zip(bars, values):
                axes[row, col].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                                   f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_evaluation_matrices(self, results):
        print("\n" + "="*60)
        print("EVALUATION MATRICES")
        print("="*60)
        
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            print("-" * 30)
            print(f"Accuracy:  {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall:    {metrics['recall']:.4f}")
            print(f"F1-Score:  {metrics['f1_score']:.4f}")
        
        print(f"\n{'='*60}")
        print("CLASSIFICATION RESULT COMPARISON:")
        print(f"{'='*60}")
        
        sorted_models = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        for rank, (model_name, metrics) in enumerate(sorted_models, 1):
            print(f"{rank}. {model_name:12} - Accuracy: {metrics['accuracy']:.4f}")
    
    def run(self):
        print("\nCNN Enhanced - Sample vs Predicted + Evaluation Metrics")
        print("="*60)
        
        file_paths, labels = self.load_data()
        print(f"\nDataset: {len(file_paths)} images")
        print(f"Fractured: {sum(labels)} | Non-fractured: {len(labels) - sum(labels)}")
        
        train_gen, test_gen, test_df = self.create_generators(file_paths, labels)
        
        results = {}
        model_names = ['InceptionV3', 'ResNet50', 'VGG16']
        
        for i, model_name in enumerate(model_names):
            print(f"\n[{i+1}/3] Training {model_name}...")
            print("-" * 40)
            
            model = self.create_model(model_name)
            model.fit(train_gen, epochs=self.epochs, verbose=1)
            
            test_gen.reset()
            predictions = model.predict(test_gen, verbose=0)
            y_pred = (predictions > 0.5).astype(int)
            y_true = test_gen.classes
            
            metrics = self.calculate_metrics(y_true, y_pred)
            results[model_name] = metrics
            
            print(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}")
            
            self.plot_sample_predictions(test_df, y_true, y_pred, predictions, model_name)
            self.plot_confusion_matrix(metrics['confusion_matrix'], model_name, metrics)
        
        self.plot_model_comparison(results)
        self.print_evaluation_matrices(results)
        
        return results

if __name__ == "__main__":
    CNN().run()