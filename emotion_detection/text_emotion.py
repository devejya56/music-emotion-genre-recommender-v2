from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

class TextEmotionDetector:
    def __init__(self, model_name='bhadresh-savani/distilbert-base-uncased-emotion'):
        """
        Initialize text emotion detector using HuggingFace BERT model
        Args:
            model_name: Pre-trained model from HuggingFace hub
        """
        self.device = 0 if torch.cuda.is_available() else -1
        print(f"Using device: {'GPU' if self.device == 0 else 'CPU'}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Create pipeline
        self.classifier = pipeline(
            'text-classification',
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            return_all_scores=True
        )
        
        # Emotion labels (model specific)
        self.emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        
    def detect_emotion(self, text):
        """
        Detect emotion from text input
        Args:
            text: Input text string
        Returns:
            Dictionary with emotion and confidence
        """
        if not text or not isinstance(text, str):
            return {'emotion': 'neutral', 'confidence': 0.0, 'all_scores': {}}
        
        try:
            # Get predictions
            results = self.classifier(text)[0]
            
            # Convert to dictionary
            all_scores = {item['label']: item['score'] for item in results}
            
            # Get top emotion
            top_emotion = max(results, key=lambda x: x['score'])
            
            return {
                'emotion': top_emotion['label'],
                'confidence': top_emotion['score'],
                'all_scores': all_scores
            }
        except Exception as e:
            print(f"Error detecting emotion: {e}")
            return {'emotion': 'neutral', 'confidence': 0.0, 'all_scores': {}}
    
    def detect_batch(self, texts):
        """
        Detect emotions for a batch of texts
        Args:
            texts: List of text strings
        Returns:
            List of emotion dictionaries
        """
        if not texts or not isinstance(texts, list):
            return []
        
        try:
            results = self.classifier(texts)
            
            emotions = []
            for result in results:
                all_scores = {item['label']: item['score'] for item in result}
                top_emotion = max(result, key=lambda x: x['score'])
                
                emotions.append({
                    'emotion': top_emotion['label'],
                    'confidence': top_emotion['score'],
                    'all_scores': all_scores
                })
            
            return emotions
        except Exception as e:
            print(f"Error in batch emotion detection: {e}")
            return [{'emotion': 'neutral', 'confidence': 0.0, 'all_scores': {}} for _ in texts]
    
    def get_dominant_emotion(self, texts):
        """
        Get the dominant emotion from multiple texts
        Args:
            texts: List of text strings
        Returns:
            Dictionary with dominant emotion and average confidence
        """
        emotions = self.detect_batch(texts)
        
        if not emotions:
            return {'emotion': 'neutral', 'confidence': 0.0}
        
        # Aggregate scores
        emotion_scores = {emotion: [] for emotion in self.emotions}
        
        for result in emotions:
            for emotion, score in result['all_scores'].items():
                if emotion in emotion_scores:
                    emotion_scores[emotion].append(score)
        
        # Calculate average scores
        avg_scores = {emotion: np.mean(scores) if scores else 0.0 
                     for emotion, scores in emotion_scores.items()}
        
        # Get dominant emotion
        dominant_emotion = max(avg_scores.items(), key=lambda x: x[1])
        
        return {
            'emotion': dominant_emotion[0],
            'confidence': dominant_emotion[1],
            'all_scores': avg_scores
        }

if __name__ == "__main__":
    # Test the emotion detector
    detector = TextEmotionDetector()
    
    # Single text example
    test_texts = [
        "I am so happy today! This is amazing!",
        "I feel really sad and lonely.",
        "This makes me so angry!",
        "I am afraid of what might happen.",
        "I love spending time with my family.",
        "Wow, that was unexpected!"
    ]
    
    print("Single text emotion detection:")
    for text in test_texts:
        result = detector.detect_emotion(text)
        print(f"Text: {text}")
        print(f"Emotion: {result['emotion']} (confidence: {result['confidence']:.4f})")
        print()
    
    # Batch detection
    print("\nBatch emotion detection:")
    batch_results = detector.detect_batch(test_texts)
    for text, result in zip(test_texts, batch_results):
        print(f"Text: {text}")
        print(f"Emotion: {result['emotion']} (confidence: {result['confidence']:.4f})")
    
    # Dominant emotion
    print("\nDominant emotion from all texts:")
    dominant = detector.get_dominant_emotion(test_texts)
    print(f"Dominant Emotion: {dominant['emotion']} (avg confidence: {dominant['confidence']:.4f})")
