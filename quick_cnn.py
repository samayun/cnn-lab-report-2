import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.applications import InceptionV3, ResNet50, VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

class CNN:
    def __init__(self):
        self.img_size = (224, 224)
        self.batch_size = 8
        self.epochs = 1
        self.use_subset = 0.01
        
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
            file_paths, labels, test_size=0.2, random_state=42
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
        
        return train_gen, test_gen
    
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
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def run(self):
        print("CNN Test - 1% data, 1 epoch")
        
        file_paths, labels = self.load_data()
        print(f"Using {len(file_paths)} images")
        
        train_gen, test_gen = self.create_generators(file_paths, labels)
        
        results = {}
        for model_name in ['InceptionV3', 'ResNet50', 'VGG16']:
            print(f"\nTraining {model_name}...")
            model = self.create_model(model_name)
            
            model.fit(train_gen, epochs=self.epochs, verbose=0)
            
            test_gen.reset()
            predictions = model.predict(test_gen, verbose=0)
            y_pred = (predictions > 0.5).astype(int)
            accuracy = accuracy_score(test_gen.classes, y_pred)
            
            results[model_name] = accuracy
            print(f"{model_name}: {accuracy:.4f}")
        
        print(f"\nResults:")
        for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
            print(f"{name}: {acc:.4f}")

if __name__ == "__main__":
    CNN().run()