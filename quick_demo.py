import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

class QuickDemo:
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
        
        file_paths, labels = [], []
        
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
    
    def create_model(self):
        base = VGG16(weights='imagenet', include_top=False, input_shape=(*self.img_size, 3))
        base.trainable = False
        
        model = models.Sequential([
            base,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer=Adam(self.learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def plot_sample_predictions(self, test_df, y_true, y_pred, y_pred_prob):
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('VGG16 - Sample vs Predicted Results', fontsize=16)
        
        sample_indices = np.random.choice(len(test_df), 8, replace=False)
        
        for i, idx in enumerate(sample_indices):
            row = i // 4
            col = i % 4
            
            img_path = test_df.iloc[idx]['filename']
            actual = y_true[idx]
            predicted = y_pred[idx]
            confidence = y_pred_prob[idx][0]
            
            img = load_img(img_path, target_size=self.img_size, color_mode='grayscale')
            
            axes[row, col].imshow(img, cmap='gray')
            
            title = f'Actual: {"Fractured" if actual == 1 else "Normal"}\n'
            title += f'Pred: {"Fractured" if predicted == 1 else "Normal"}\n'
            title += f'Confidence: {confidence:.3f}'
            axes[row, col].set_title(title, fontsize=10)
            axes[row, col].axis('off')
            
            if actual == predicted:
                rect = plt.Rectangle((10, 10), 200, 200, fill=False, edgecolor='green', linewidth=3)
            else:
                rect = plt.Rectangle((10, 10), 200, 200, fill=False, edgecolor='red', linewidth=3)
            axes[row, col].add_patch(rect)
        
        plt.tight_layout()
        plt.savefig('sample_vs_predicted.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_evaluation_matrices(self, y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary')
        recall = recall_score(y_true, y_pred, average='binary')
        f1 = f1_score(y_true, y_pred, average='binary')
        cm = confusion_matrix(y_true, y_pred)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Non-Fractured', 'Fractured'],
                   yticklabels=['Non-Fractured', 'Fractured'],
                   ax=axes[0])
        axes[0].set_title(f'Confusion Matrix\nAccuracy: {accuracy:.4f}')
        axes[0].set_ylabel('Actual')
        axes[0].set_xlabel('Predicted')
        
        # Metrics Bar Chart
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [accuracy, precision, recall, f1]
        colors = ['skyblue', 'lightgreen', 'salmon', 'gold']
        
        bars = axes[1].bar(metrics, values, color=colors)
        axes[1].set_title('Evaluation Metrics')
        axes[1].set_ylim(0, 1)
        axes[1].set_ylabel('Score')
        
        for bar, value in zip(bars, values):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('evaluation_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return accuracy, precision, recall, f1, cm
    
    def run(self):
        print("\nQuick CNN Demo - Sample vs Predicted + Evaluation")
        print("="*50)
        
        file_paths, labels = self.load_data()
        print(f"Dataset: {len(file_paths)} images")
        print(f"Fractured: {sum(labels)} | Non-fractured: {len(labels) - sum(labels)}")
        
        train_gen, test_gen, test_df = self.create_generators(file_paths, labels)
        
        print("\nTraining VGG16...")
        print("-" * 30)
        
        model = self.create_model()
        model.fit(train_gen, epochs=self.epochs, verbose=1)
        
        test_gen.reset()
        predictions = model.predict(test_gen, verbose=0)
        y_pred = (predictions > 0.5).astype(int)
        y_true = test_gen.classes
        
        print(f"\nVGG16 Training Complete!")
        
        # Generate visualizations
        self.plot_sample_predictions(test_df, y_true, y_pred, predictions)
        accuracy, precision, recall, f1, cm = self.plot_evaluation_matrices(y_true, y_pred)
        
        # Print results
        print("\n" + "="*50)
        print("CLASSIFICATION RESULTS")
        print("="*50)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print("="*50)
        
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1}

if __name__ == "__main__":
    demo = QuickDemo()
    results = demo.run()