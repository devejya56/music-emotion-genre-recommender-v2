import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Tuple

class HybridRecommender:
    def __init__(self, songs_database_path='data/songs_database.json'):
        """
        Initialize hybrid recommendation system
        Maps detected emotions to genres and recommends songs
        Args:
            songs_database_path: Path to songs database JSON file
        """
        self.songs_database_path = songs_database_path
        self.songs_df = None
        
        # Emotion to genre mapping
        self.emotion_genre_map = {
            # Text emotions
            'joy': ['pop', 'disco', 'reggae'],
            'love': ['pop', 'jazz', 'classical'],
            'sadness': ['blues', 'classical', 'country'],
            'anger': ['metal', 'rock', 'hiphop'],
            'fear': ['classical', 'blues'],
            'surprise': ['disco', 'pop', 'reggae'],
            
            # Audio emotions
            'happy': ['pop', 'disco', 'reggae'],
            'calm': ['classical', 'jazz', 'blues'],
            'sad': ['blues', 'classical', 'country'],
            'angry': ['metal', 'rock', 'hiphop'],
            'neutral': ['jazz', 'pop', 'classical'],
            'fearful': ['classical', 'blues'],
            'surprised': ['disco', 'pop', 'reggae']
        }
        
        # Load or create songs database
        self.load_songs_database()
    
    def load_songs_database(self):
        """
        Load songs database from JSON file or create sample database
        """
        if os.path.exists(self.songs_database_path):
            with open(self.songs_database_path, 'r') as f:
                songs_data = json.load(f)
                self.songs_df = pd.DataFrame(songs_data)
            print(f"Loaded {len(self.songs_df)} songs from database.")
        else:
            # Create sample database
            self.songs_df = self.create_sample_database()
            self.save_songs_database()
            print("Created sample songs database.")
    
    def create_sample_database(self) -> pd.DataFrame:
        """
        Create a sample songs database
        Returns:
            DataFrame with sample songs
        """
        sample_songs = [
            # Pop songs
            {'title': 'Happy Together', 'artist': 'The Turtles', 'genre': 'pop', 'mood': 'happy', 'popularity': 85},
            {'title': 'Shake It Off', 'artist': 'Taylor Swift', 'genre': 'pop', 'mood': 'happy', 'popularity': 90},
            {'title': 'Love Story', 'artist': 'Taylor Swift', 'genre': 'pop', 'mood': 'love', 'popularity': 88},
            
            # Blues songs
            {'title': 'The Thrill Is Gone', 'artist': 'B.B. King', 'genre': 'blues', 'mood': 'sad', 'popularity': 82},
            {'title': 'Stormy Monday', 'artist': 'T-Bone Walker', 'genre': 'blues', 'mood': 'sad', 'popularity': 78},
            
            # Classical songs
            {'title': 'Moonlight Sonata', 'artist': 'Beethoven', 'genre': 'classical', 'mood': 'calm', 'popularity': 95},
            {'title': 'Clair de Lune', 'artist': 'Debussy', 'genre': 'classical', 'mood': 'calm', 'popularity': 92},
            {'title': 'Requiem', 'artist': 'Mozart', 'genre': 'classical', 'mood': 'sad', 'popularity': 90},
            
            # Rock songs
            {'title': 'Breaking the Law', 'artist': 'Judas Priest', 'genre': 'rock', 'mood': 'angry', 'popularity': 80},
            {'title': "Livin' on a Prayer", 'artist': 'Bon Jovi', 'genre': 'rock', 'mood': 'energetic', 'popularity': 88},
            
            # Metal songs
            {'title': 'Master of Puppets', 'artist': 'Metallica', 'genre': 'metal', 'mood': 'angry', 'popularity': 92},
            {'title': 'Paranoid', 'artist': 'Black Sabbath', 'genre': 'metal', 'mood': 'angry', 'popularity': 85},
            
            # Jazz songs
            {'title': 'Take Five', 'artist': 'Dave Brubeck', 'genre': 'jazz', 'mood': 'calm', 'popularity': 87},
            {'title': 'Autumn Leaves', 'artist': 'Bill Evans', 'genre': 'jazz', 'mood': 'calm', 'popularity': 84},
            
            # Disco songs
            {'title': 'Stayin\' Alive', 'artist': 'Bee Gees', 'genre': 'disco', 'mood': 'happy', 'popularity': 93},
            {'title': 'Le Freak', 'artist': 'Chic', 'genre': 'disco', 'mood': 'happy', 'popularity': 86},
            
            # Hip-hop songs
            {'title': 'Lose Yourself', 'artist': 'Eminem', 'genre': 'hiphop', 'mood': 'energetic', 'popularity': 94},
            {'title': 'Juicy', 'artist': 'The Notorious B.I.G.', 'genre': 'hiphop', 'mood': 'confident', 'popularity': 89},
            
            # Reggae songs
            {'title': 'Three Little Birds', 'artist': 'Bob Marley', 'genre': 'reggae', 'mood': 'happy', 'popularity': 91},
            {'title': 'No Woman, No Cry', 'artist': 'Bob Marley', 'genre': 'reggae', 'mood': 'calm', 'popularity': 90},
            
            # Country songs
            {'title': 'Jolene', 'artist': 'Dolly Parton', 'genre': 'country', 'mood': 'sad', 'popularity': 86},
            {'title': 'Ring of Fire', 'artist': 'Johnny Cash', 'genre': 'country', 'mood': 'energetic', 'popularity': 88}
        ]
        
        return pd.DataFrame(sample_songs)
    
    def save_songs_database(self):
        """
        Save songs database to JSON file
        """
        os.makedirs(os.path.dirname(self.songs_database_path), exist_ok=True)
        self.songs_df.to_json(self.songs_database_path, orient='records', indent=2)
        print(f"Saved songs database to {self.songs_database_path}")
    
    def map_emotion_to_genres(self, emotion: str) -> List[str]:
        """
        Map detected emotion to relevant genres
        Args:
            emotion: Detected emotion string
        Returns:
            List of relevant genres
        """
        emotion_lower = emotion.lower()
        return self.emotion_genre_map.get(emotion_lower, ['pop', 'jazz', 'classical'])
    
    def get_recommendations(self, emotion: str, predicted_genre: str = None, 
                           n_recommendations: int = 5) -> List[Dict]:
        """
        Get song recommendations based on emotion and optionally predicted genre
        Args:
            emotion: Detected emotion from text/audio
            predicted_genre: Genre predicted from audio CNN (optional)
            n_recommendations: Number of songs to recommend
        Returns:
            List of recommended songs
        """
        if self.songs_df is None or len(self.songs_df) == 0:
            return []
        
        # Get genres matching the emotion
        emotion_genres = self.map_emotion_to_genres(emotion)
        
        # Filter songs by emotion-matched genres
        filtered_songs = self.songs_df[self.songs_df['genre'].isin(emotion_genres)].copy()
        
        # If we have a predicted genre from CNN, boost its priority
        if predicted_genre and predicted_genre in emotion_genres:
            # Prioritize the predicted genre
            genre_songs = filtered_songs[filtered_songs['genre'] == predicted_genre]
            other_songs = filtered_songs[filtered_songs['genre'] != predicted_genre]
            
            # Get recommendations from predicted genre first
            n_from_genre = min(len(genre_songs), max(n_recommendations // 2, 2))
            n_from_others = n_recommendations - n_from_genre
            
            genre_recommendations = genre_songs.nlargest(n_from_genre, 'popularity')
            other_recommendations = other_songs.nlargest(n_from_others, 'popularity')
            
            recommendations = pd.concat([genre_recommendations, other_recommendations])
        else:
            # Just use popularity-based sorting
            recommendations = filtered_songs.nlargest(n_recommendations, 'popularity')
        
        # Convert to list of dictionaries
        return recommendations.head(n_recommendations).to_dict('records')
    
    def get_hybrid_recommendations(self, text_emotion: str = None, 
                                  audio_emotion: str = None,
                                  predicted_genre: str = None,
                                  n_recommendations: int = 5) -> Dict:
        """
        Get hybrid recommendations using both text and audio emotions
        Args:
            text_emotion: Emotion detected from text
            audio_emotion: Emotion detected from audio
            predicted_genre: Genre predicted from audio
            n_recommendations: Number of recommendations
        Returns:
            Dictionary with recommendations and metadata
        """
        # Determine primary emotion
        if text_emotion and audio_emotion:
            # Use both emotions - prioritize text emotion
            primary_emotion = text_emotion
            secondary_emotion = audio_emotion
        elif text_emotion:
            primary_emotion = text_emotion
            secondary_emotion = None
        elif audio_emotion:
            primary_emotion = audio_emotion
            secondary_emotion = None
        else:
            primary_emotion = 'neutral'
            secondary_emotion = None
        
        # Get recommendations
        recommendations = self.get_recommendations(
            emotion=primary_emotion,
            predicted_genre=predicted_genre,
            n_recommendations=n_recommendations
        )
        
        return {
            'primary_emotion': primary_emotion,
            'secondary_emotion': secondary_emotion,
            'predicted_genre': predicted_genre,
            'recommended_genres': self.map_emotion_to_genres(primary_emotion),
            'recommendations': recommendations,
            'total_recommendations': len(recommendations)
        }

if __name__ == "__main__":
    # Test the recommender
    recommender = HybridRecommender()
    
    # Test emotion to genre mapping
    print("\nEmotion to Genre Mapping:")
    test_emotions = ['joy', 'sadness', 'anger', 'love']
    for emotion in test_emotions:
        genres = recommender.map_emotion_to_genres(emotion)
        print(f"{emotion}: {genres}")
    
    # Test recommendations
    print("\nRecommendations for 'joy' emotion:")
    recs = recommender.get_recommendations('joy', n_recommendations=5)
    for i, song in enumerate(recs, 1):
        print(f"{i}. {song['title']} by {song['artist']} ({song['genre']})")
    
    # Test hybrid recommendations
    print("\nHybrid recommendations (text: 'sadness', audio: 'calm', genre: 'classical'):")
    hybrid_result = recommender.get_hybrid_recommendations(
        text_emotion='sadness',
        audio_emotion='calm',
        predicted_genre='classical',
        n_recommendations=5
    )
    print(f"Primary emotion: {hybrid_result['primary_emotion']}")
    print(f"Predicted genre: {hybrid_result['predicted_genre']}")
    print(f"Recommended genres: {hybrid_result['recommended_genres']}")
    print(f"\nRecommended songs:")
    for i, song in enumerate(hybrid_result['recommendations'], 1):
        print(f"{i}. {song['title']} by {song['artist']} ({song['genre']})")
