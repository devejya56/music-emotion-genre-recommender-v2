import librosa
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib

class AudioEmotionDetector:
    def __init__(self):
        """
        Initialize audio emotion detector
        Uses audio features (MFCC, spectral, rhythm) for emotion detection
        """
        self.model = None
        self.scaler = StandardScaler()
        
        # Emotion labels for audio
        self.emotions = ['happy', 'sad', 'angry', 'neutral', 'calm', 'fearful', 'surprised']
        
        # Initialize with a default model
        self._initialize_default_model()
    
    def _initialize_default_model(self):
        """Initialize a default SVM model"""
        self.model = SVC(kernel='rbf', probability=True, random_state=42)
    
    def extract_audio_features(self, audio_path, sr=22050, duration=30):
        """
        Extract comprehensive audio features for emotion detection
        Args:
            audio_path: Path to audio file
            sr: Sample rate
            duration: Max duration to load
        Returns:
            Feature vector as numpy array
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=sr, duration=duration)
            
            features = []
            
            # 1. MFCC features (13 coefficients)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            features.extend(mfcc_mean)
            features.extend(mfcc_std)
            
            # 2. Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            features.append(np.mean(spectral_centroid))
            features.append(np.std(spectral_centroid))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            features.append(np.mean(spectral_rolloff))
            features.append(np.std(spectral_rolloff))
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            features.append(np.mean(spectral_bandwidth))
            features.append(np.std(spectral_bandwidth))
            
            # 3. Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)
            features.append(np.mean(zcr))
            features.append(np.std(zcr))
            
            # 4. Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features.append(np.mean(chroma))
            features.append(np.std(chroma))
            
            # 5. RMS Energy
            rms = librosa.feature.rms(y=y)
            features.append(np.mean(rms))
            features.append(np.std(rms))
            
            # 6. Tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features.append(tempo)
            
            return np.array(features)
            
        except Exception as e:
            print(f"Error extracting features from {audio_path}: {e}")
            return None
    
    def train(self, audio_paths, labels):
        """
        Train the emotion detection model
        Args:
            audio_paths: List of paths to audio files
            labels: List of emotion labels (indices)
        """
        print("Extracting features from audio files...")
        X = []
        y = []
        
        for path, label in zip(audio_paths, labels):
            features = self.extract_audio_features(path)
            if features is not None:
                X.append(features)
                y.append(label)
        
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        print("Training model...")
        self.model.fit(X_scaled, y)
        print("Training complete!")
        
        return self.model
    
    def predict(self, audio_path):
        """
        Predict emotion from audio file
        Args:
            audio_path: Path to audio file
        Returns:
            Dictionary with emotion and confidence
        """
        if self.model is None:
            return {'emotion': 'neutral', 'confidence': 0.0, 'all_scores': {}}
        
        features = self.extract_audio_features(audio_path)
        if features is None:
            return {'emotion': 'neutral', 'confidence': 0.0, 'all_scores': {}}
        
        try:
            # Scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Get predictions
            probabilities = self.model.predict_proba(features_scaled)[0]
            predicted_idx = np.argmax(probabilities)
            
            # Create scores dictionary
            all_scores = {self.emotions[i]: float(prob) 
                         for i, prob in enumerate(probabilities)}
            
            return {
                'emotion': self.emotions[predicted_idx],
                'confidence': float(probabilities[predicted_idx]),
                'all_scores': all_scores
            }
        except Exception as e:
            print(f"Error predicting emotion: {e}")
            return {'emotion': 'neutral', 'confidence': 0.0, 'all_scores': {}}
    
    def predict_batch(self, audio_paths):
        """
        Predict emotions for multiple audio files
        Args:
            audio_paths: List of audio file paths
        Returns:
            List of prediction dictionaries
        """
        results = []
        for path in audio_paths:
            result = self.predict(path)
            results.append(result)
        return results
    
    def save_model(self, model_path='models/audio_emotion_model.pkl', 
                   scaler_path='models/audio_emotion_scaler.pkl'):
        """
        Save trained model and scaler
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
    
    def load_model(self, model_path='models/audio_emotion_model.pkl',
                   scaler_path='models/audio_emotion_scaler.pkl'):
        """
        Load trained model and scaler
        """
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print(f"Model loaded from {model_path}")
            print(f"Scaler loaded from {scaler_path}")
        else:
            print(f"Model files not found. Using default model.")

if __name__ == "__main__":
    # Example usage
    detector = AudioEmotionDetector()
    
    # For training (example)
    # Assuming you have organized audio files by emotion
    # audio_paths = ['path/to/happy1.wav', 'path/to/sad1.wav', ...]
    # labels = [0, 1, ...]  # Indices corresponding to emotions
    # detector.train(audio_paths, labels)
    # detector.save_model()
    
    # For prediction
    # result = detector.predict('path/to/test_audio.wav')
    # print(f"Detected emotion: {result['emotion']} (confidence: {result['confidence']:.4f})")
    # print(f"All scores: {result['all_scores']}")
    
    print("Audio Emotion Detector initialized.")
    print(f"Supported emotions: {detector.emotions}")
