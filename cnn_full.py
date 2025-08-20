import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.applications import InceptionV3, ResNet50, VGG16, DenseNet121, MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

class CNN:
    def __init__(self):
        self.img_size = (224, 224)
        self.batch_size = 32
        self.epochs = 3
        self.use_subset = 1.0  # Use all data
        
    def load_data(self):
        data_dir = 'Augmented Dataset'
        fractured_dir = os.path.join(data_dir, 'Fractured')
        non_fractured_dir = os.path.join(data_dir, 'Non-Fractured')
        
        fractured_files = [f for f in os.listdir(fractured_dir) if f.endswith('.jpg')]
        non_fractured_files = [f for f in os.listdir(non_fractured_dir) if f.endswith('.jpg')]
        
        print(f"Found {len(fractured_files)} fractured and {len(non_fractured_files)} non-fractured images")
        
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
        
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True
        )
        
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
            target_size=self.img_size, batch_size=self.batch_size, class_mode='binary'
        )
        
        test_gen = test_datagen.flow_from_dataframe(
            test_df, x_col='filename', y_col='class',
            target_size=self.img_size, batch_size=self.batch_size, class_mode='binary', shuffle=False
        )
        
        return train_gen, test_gen
    
    def create_model(self, name):
        base_models = {
            'InceptionV3': InceptionV3(weights='imagenet', include_top=False, input_shape=(*self.img_size, 3)),
            'ResNet50': ResNet50(weights='imagenet', include_top=False, input_shape=(*self.img_size, 3)),
            'VGG16': VGG16(weights='imagenet', include_top=False, input_shape=(*self.img_size, 3)),
            'DenseNet121': DenseNet121(weights='imagenet', include_top=False, input_shape=(*self.img_size, 3)),
            'MobileNetV2': MobileNetV2(weights='imagenet', include_top=False, input_shape=(*self.img_size, 3))
        }
        
        base = base_models[name]
        base.trainable = False
        
        model = models.Sequential([
            base,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def run(self):
        print("CNN Full Dataset Training - All 5 Models")
        print("="*50)
        
        file_paths, labels = self.load_data()
        print(f"Total images: {len(file_paths)}")
        
        train_gen, test_gen = self.create_generators(file_paths, labels)
        
        results = {}
        model_names = ['InceptionV3', 'ResNet50', 'VGG16', 'DenseNet121', 'MobileNetV2']
        
        for i, model_name in enumerate(model_names):
            print(f"\n[{i+1}/5] Training {model_name}...")
            print("-"*30)
            
            model = self.create_model(model_name)
            
            history = model.fit(
                train_gen, 
                epochs=self.epochs, 
                verbose=1,
                steps_per_epoch=len(train_gen)
            )
            
            test_gen.reset()
            predictions = model.predict(test_gen, verbose=0)
            y_pred = (predictions > 0.5).astype(int)
            accuracy = accuracy_score(test_gen.classes, y_pred)
            
            results[model_name] = {
                'accuracy': accuracy,
                'final_train_acc': history.history['accuracy'][-1]
            }
            
            print(f"{model_name} - Test Accuracy: {accuracy:.4f}")
        
        print(f"\n{'='*60}")
        print("FINAL RESULTS - ALL MODELS")
        print(f"{'='*60}")
        
        sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        for rank, (name, metrics) in enumerate(sorted_results, 1):
            print(f"{rank}. {name:12} - Test: {metrics['accuracy']:.4f} | Train: {metrics['final_train_acc']:.4f}")
        
        return results

if __name__ == "__main__":
    CNN().run()