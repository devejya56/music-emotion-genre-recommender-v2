import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import librosa
import numpy as np
import os
from sklearn.model_selection import train_test_split
import json

class AudioGenreCNN:
    def __init__(self, n_genres=10, input_shape=(128, 128, 1)):
        self.n_genres = n_genres
        self.input_shape = input_shape
        self.model = self.build_model()
        
    def build_model(self):
        """Build CNN architecture for genre classification"""
        model = keras.Sequential([
            # First Conv Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Conv Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Conv Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense Layers
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.n_genres, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def extract_features(self, audio_path, sr=22050, n_mels=128, max_duration=30):
        """Extract mel-spectrogram features from audio file using librosa"""
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=sr, duration=max_duration)
            
            # Extract mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize
            mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / mel_spec_db.std()
            
            # Resize to fixed shape
            if mel_spec_db.shape[1] < 128:
                mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, 128 - mel_spec_db.shape[1])), mode='constant')
            else:
                mel_spec_db = mel_spec_db[:, :128]
            
            return mel_spec_db.reshape(128, 128, 1)
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None
    
    def prepare_dataset(self, data_dir, genres):
        """Prepare dataset from audio files organized by genre folders"""
        X = []
        y = []
        
        for genre_idx, genre in enumerate(genres):
            genre_path = os.path.join(data_dir, genre)
            if not os.path.exists(genre_path):
                continue
                
            for audio_file in os.listdir(genre_path):
                if audio_file.endswith(('.wav', '.mp3')):
                    audio_path = os.path.join(genre_path, audio_file)
                    features = self.extract_features(audio_path)
                    
                    if features is not None:
                        X.append(features)
                        y.append(genre_idx)
        
        X = np.array(X)
        y = keras.utils.to_categorical(y, num_classes=self.n_genres)
        
        return X, y
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the model"""
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
            ]
        )
        return history
    
    def save_model(self, path='models/genre_cnn_model.h5'):
        """Save trained model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path='models/genre_cnn_model.h5'):
        """Load trained model"""
        self.model = keras.models.load_model(path)
        print(f"Model loaded from {path}")
    
    def predict(self, audio_path, genres):
        """Predict genre for a single audio file"""
        features = self.extract_features(audio_path)
        if features is None:
            return None
        
        features = features.reshape(1, 128, 128, 1)
        predictions = self.model.predict(features)
        predicted_genre_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_genre_idx]
        
        return genres[predicted_genre_idx], confidence

if __name__ == "__main__":
    # Define genres
    GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 
              'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    # Initialize model
    genre_cnn = AudioGenreCNN(n_genres=len(GENRES))
    
    # Prepare dataset (assuming data organized in folders)
    DATA_DIR = 'data/genres'  # Update with your data path
    
    print("Preparing dataset...")
    X, y = genre_cnn.prepare_dataset(DATA_DIR, GENRES)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")
    
    # Train model
    print("Training model...")
    history = genre_cnn.train(X_train, y_train, X_val, y_val, epochs=50)
    
    # Save model
    genre_cnn.save_model('models/genre_cnn_model.h5')
    
    # Save genre labels
    with open('models/genres.json', 'w') as f:
        json.dump(GENRES, f)
    
    print("Training complete!")
