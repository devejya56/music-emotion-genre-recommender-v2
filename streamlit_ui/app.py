import streamlit as st
import sys
import os
import numpy as np

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emotion_detection.text_emotion import TextEmotionDetector
from emotion_detection.audio_emotion import AudioEmotionDetector
from audio_genre_cnn.train import AudioGenreCNN
from recommendation_engine.recommend import HybridRecommender
import json

# Page configuration
st.set_page_config(
    page_title="Music Emotion & Genre Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .recommendation-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
    st.session_state.text_detector = None
    st.session_state.audio_detector = None
    st.session_state.genre_cnn = None
    st.session_state.recommender = None

@st.cache_resource
def load_models():
    """Load all models (cached to prevent reloading)"""
    with st.spinner('Loading models... This may take a minute.'):
        try:
            # Load text emotion detector
            text_detector = TextEmotionDetector()
            
            # Load audio emotion detector
            audio_detector = AudioEmotionDetector()
            
            # Load genre CNN
            genre_cnn = AudioGenreCNN(n_genres=10)
            # Try to load pre-trained model if exists
            try:
                genre_cnn.load_model('models/genre_cnn_model.h5')
            except:
                st.warning("Genre CNN model not found. Genre prediction will be unavailable.")
            
            # Load recommender
            recommender = HybridRecommender()
            
            return text_detector, audio_detector, genre_cnn, recommender
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return None, None, None, None

def main():
    # Title and description
    st.title("🎵 Music Emotion & Genre Recommender")
    st.markdown("""
    This application detects emotions from text and audio inputs, predicts music genres, 
    and provides personalized song recommendations based on your emotional state.
    """)
    
    # Load models
    if not st.session_state.models_loaded:
        models = load_models()
        if all(models):
            st.session_state.text_detector = models[0]
            st.session_state.audio_detector = models[1]
            st.session_state.genre_cnn = models[2]
            st.session_state.recommender = models[3]
            st.session_state.models_loaded = True
            st.success("Models loaded successfully!")
        else:
            st.error("Failed to load models. Please check your setup.")
            return
    
    # Sidebar for input options
    st.sidebar.header("Input Options")
    input_mode = st.sidebar.radio(
        "Select Input Mode:",
        ["Text Only", "Audio Only", "Text + Audio (Hybrid)"]
    )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    text_emotion = None
    audio_emotion = None
    predicted_genre = None
    
    with col1:
        st.header("📝 Input Section")
        
        # Text input
        if input_mode in ["Text Only", "Text + Audio (Hybrid)"]:
            st.subheader("Text Input")
            user_text = st.text_area(
                "Enter your thoughts or feelings:",
                height=150,
                placeholder="E.g., I'm feeling really happy today! Everything is going great."
            )
            
            if st.button("Analyze Text Emotion") and user_text:
                with st.spinner('Analyzing text emotion...'):
                    result = st.session_state.text_detector.detect_emotion(user_text)
                    text_emotion = result['emotion']
                    
                    st.success(f"**Detected Emotion:** {text_emotion.upper()}")
                    st.write(f"**Confidence:** {result['confidence']:.2%}")
                    
                    # Show all emotion scores
                    with st.expander("View All Emotion Scores"):
                        for emotion, score in result['all_scores'].items():
                            st.progress(score, text=f"{emotion}: {score:.2%}")
        
        # Audio input
        if input_mode in ["Audio Only", "Text + Audio (Hybrid)"]:
            st.subheader("Audio Input")
            audio_file = st.file_uploader(
                "Upload an audio file (MP3, WAV):",
                type=['mp3', 'wav', 'ogg', 'm4a']
            )
            
            if audio_file is not None:
                st.audio(audio_file, format='audio/wav')
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    if st.button("Analyze Audio Emotion"):
                        # Save uploaded file temporarily
                        temp_path = f"temp_{audio_file.name}"
                        with open(temp_path, 'wb') as f:
                            f.write(audio_file.read())
                        
                        with st.spinner('Analyzing audio emotion...'):
                            result = st.session_state.audio_detector.predict(temp_path)
                            audio_emotion = result['emotion']
                            
                            st.success(f"**Detected Emotion:** {audio_emotion.upper()}")
                            st.write(f"**Confidence:** {result['confidence']:.2%}")
                            
                            # Show all emotion scores
                            with st.expander("View All Emotion Scores"):
                                for emotion, score in result['all_scores'].items():
                                    st.progress(score, text=f"{emotion}: {score:.2%}")
                        
                        # Clean up temp file
                        os.remove(temp_path)
                
                with col_b:
                    if st.button("Predict Genre"):
                        # Save uploaded file temporarily
                        temp_path = f"temp_{audio_file.name}"
                        with open(temp_path, 'wb') as f:
                            f.write(audio_file.read())
                        
                        with st.spinner('Predicting genre...'):
                            try:
                                # Load genre labels
                                with open('models/genres.json', 'r') as f:
                                    genres = json.load(f)
                                
                                genre, confidence = st.session_state.genre_cnn.predict(temp_path, genres)
                                predicted_genre = genre
                                
                                st.success(f"**Predicted Genre:** {genre.upper()}")
                                st.write(f"**Confidence:** {confidence:.2%}")
                            except Exception as e:
                                st.error(f"Genre prediction error: {e}")
                        
                        # Clean up temp file
                        os.remove(temp_path)
    
    with col2:
        st.header("🎵 Recommendations")
        
        if st.button("Get Song Recommendations", type="primary"):
            if text_emotion or audio_emotion:
                with st.spinner('Generating recommendations...'):
                    # Get hybrid recommendations
                    result = st.session_state.recommender.get_hybrid_recommendations(
                        text_emotion=text_emotion,
                        audio_emotion=audio_emotion,
                        predicted_genre=predicted_genre,
                        n_recommendations=5
                    )
                    
                    # Display results
                    st.subheader("🎯 Analysis Summary")
                    col_r1, col_r2, col_r3 = st.columns(3)
                    
                    with col_r1:
                        if result['primary_emotion']:
                            st.metric("Primary Emotion", result['primary_emotion'].upper())
                    
                    with col_r2:
                        if result['secondary_emotion']:
                            st.metric("Secondary Emotion", result['secondary_emotion'].upper())
                    
                    with col_r3:
                        if result['predicted_genre']:
                            st.metric("Predicted Genre", result['predicted_genre'].upper())
                    
                    # Show recommended genres
                    st.subheader("🎸 Recommended Genres")
                    st.write(", ".join([g.upper() for g in result['recommended_genres']]))
                    
                    # Show song recommendations
                    st.subheader("🎼 Recommended Songs")
                    if result['recommendations']:
                        for i, song in enumerate(result['recommendations'], 1):
                            with st.container():
                                st.markdown(f"""
                                <div class='recommendation-card'>
                                    <h4>{i}. {song['title']}</h4>
                                    <p><strong>Artist:</strong> {song['artist']}</p>
                                    <p><strong>Genre:</strong> {song['genre'].upper()}</p>
                                    <p><strong>Popularity:</strong> {song['popularity']}/100</p>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.warning("No recommendations available. Try adjusting your inputs.")
            else:
                st.warning("Please analyze at least one input (text or audio) before getting recommendations.")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About
    This system uses:
    - **BERT** for text emotion detection
    - **SVM** for audio emotion detection
    - **CNN** for genre classification
    - **Hybrid Recommender** for personalized suggestions
    """)

if __name__ == "__main__":
    main()
