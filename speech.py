"""
Simple Voice Diarization Streamlit App using SpeechBrain with permission fixes

This version stores cached models in the user's home directory to avoid permission issues.
"""

import os
import numpy as np
import streamlit as st
import tempfile
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
import torchaudio
import speechbrain as sb
from speechbrain.inference import EncoderClassifier
import time
import base64
from pydub import AudioSegment
import warnings
import pathlib
warnings.filterwarnings("ignore")

# Set page configuration
st.set_page_config(
    page_title="Voice Diarization App",
    page_icon="🎤",
    layout="wide"
)

# Create a cache directory in the user's home directory
HOME_DIR = str(pathlib.Path.home())
CACHE_DIR = os.path.join(HOME_DIR, ".voice_diarization_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Sidebar
with st.sidebar:
    st.title("🎤 Nielit Voice Diarization")
    st.markdown("Separate speakers in audio recordings")
    
    st.markdown("---")
    st.markdown("""
    ### Instructions
    1. Upload an audio file
    2. Set the maximum number of speakers (if known)
    3. Click 'Process Audio'
    4. View the timeline and listen to segments
    
    Supported formats: WAV, MP3, OGG, FLAC
    """)
    
    st.markdown("---")
    st.markdown("Made with ❤️ using NIELIT")

class SimpleDiarization:
    def __init__(self):
        """Initialize the diarization system"""
        with st.spinner("Loading SpeechBrain model (this may take a moment)..."):
            # Load SpeechBrain speaker embedding model using the cache directory
            self.embedding_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb"
            )
            st.success("Model loaded successfully!")
    
    def segment_audio(self, audio_path, max_speakers=None, min_segment_length=3.0):
        """
        Segment audio file and perform diarization
        
        Parameters:
        - audio_path: path to the audio file
        - max_speakers: maximum number of speakers to detect
        - min_segment_length: minimum segment length in seconds
        
        Returns:
        - segments: list of (start_time, end_time, speaker_id)
        """
        try:
            # Load audio
            signal, sr = torchaudio.load(audio_path)
            signal = signal.mean(dim=0).unsqueeze(0)  # Convert to mono
            
            # Calculate segment length in samples
            segment_length = int(min_segment_length * sr)
            
            # Segment the audio file
            segments = []
            
            # Process segments
            for start_idx in range(0, signal.shape[1], segment_length):
                end_idx = min(start_idx + segment_length, signal.shape[1])
                
                # Skip very short segments at the end
                if end_idx - start_idx < sr * 1.0:  # Segments shorter than 1 second
                    break
                    
                segment = signal[:, start_idx:end_idx]
                
                # Extract embedding for this segment
                with torch.no_grad():
                    embedding = self.embedding_model.encode_batch(segment).squeeze()
                
                # Convert to numpy for storage
                embedding_np = embedding.cpu().numpy()
                
                start_time = start_idx / sr
                end_time = end_idx / sr
                
                segments.append((start_time, end_time, embedding_np))
            
            # Cluster the embeddings to assign speaker IDs
            if len(segments) > 0:
                # Extract embeddings for clustering
                embeddings = np.vstack([s[2] for s in segments])
                
                # Skip clustering if only one segment
                if len(segments) == 1:
                    return [(segments[0][0], segments[0][1], "Speaker_1")]
                
                # Determine number of speakers
                if max_speakers is None or max_speakers <= 0:
                    # If not specified, try to estimate (simplified method)
                    from sklearn.cluster import AgglomerativeClustering
                    
                    # Try different numbers of clusters and select best using silhouette score
                    from sklearn.metrics import silhouette_score
                    
                    max_possible = min(8, len(segments))  # Upper limit
                    best_score = -1
                    best_n = 2  # Default to 2 speakers
                    
                    for n in range(2, max_possible + 1):
                        if len(segments) > n:  # Ensure we have more segments than clusters
                            clustering = AgglomerativeClustering(n_clusters=n).fit(embeddings)
                            if len(set(clustering.labels_)) > 1:  # More than one cluster found
                                score = silhouette_score(embeddings, clustering.labels_)
                                if score > best_score:
                                    best_score = score
                                    best_n = n
                    
                    n_speakers = best_n
                else:
                    n_speakers = min(max_speakers, len(segments))
                
                # Perform clustering
                from sklearn.cluster import AgglomerativeClustering
                clustering = AgglomerativeClustering(n_clusters=n_speakers).fit(embeddings)
                
                # Update segments with speaker IDs
                diarized_segments = []
                for i, (start_time, end_time, _) in enumerate(segments):
                    speaker_id = f"Speaker_{clustering.labels_[i] + 1}"
                    diarized_segments.append((start_time, end_time, speaker_id))
                
                return diarized_segments
            
            return []
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")
            return []

    def visualize_diarization(self, segments):
        """Create a visualization of the diarization results"""
        if not segments:
            return None
            
        # Get unique speakers
        speakers = sorted(list(set(spk for _, _, spk in segments)))
        
        # Set up the plot
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = plt.cm.tab10(np.linspace(0, 1, len(speakers)))
        
        # Create a mapping of speaker to index
        speaker_to_idx = {speaker: i for i, speaker in enumerate(speakers)}
        
        # Plot each speaker segment
        for start, end, speaker in segments:
            i = speaker_to_idx[speaker]
            ax.add_patch(Rectangle((start, i), end-start, 0.6, color=colors[i]))
            
            # Only add text label if segment is wide enough
            if end-start > 1.0:  # Only label segments longer than 1 second
                ax.text(start + (end-start)/2, i + 0.3, speaker, 
                       ha='center', va='center', fontsize=9)
        
        # Set the plot limits and labels
        ax.set_yticks(np.arange(len(speakers)) + 0.3)
        ax.set_yticklabels(speakers)
        ax.set_xlabel('Time (seconds)')
        ax.set_title('Speaker Diarization')
        
        # Get the maximum time to set as xlim
        max_time = max(end for _, end, _ in segments)
        ax.set_xlim(0, max_time)
        
        return fig

    def separate_speakers(self, audio_path, segments):
        """Separate the audio into different files per speaker"""
        try:
            # Create a temporary directory for the separated audio files
            temp_dir = tempfile.mkdtemp()
            
            # Load the audio file
            audio = AudioSegment.from_file(audio_path)
            
            # Create a dictionary to store audio segments for each speaker
            speaker_segments = {}
            speaker_files = {}
            
            # Process each segment
            for start, end, speaker in segments:
                start_ms = int(start * 1000)
                end_ms = int(end * 1000)
                
                # Extract the segment
                segment = audio[start_ms:end_ms]
                
                # Add to the speaker's collection
                if speaker not in speaker_segments:
                    speaker_segments[speaker] = segment
                else:
                    speaker_segments[speaker] += segment
            
            # Export each speaker's audio
            for speaker, segment in speaker_segments.items():
                output_path = os.path.join(temp_dir, f"{speaker}.wav")
                segment.export(output_path, format="wav")
                speaker_files[speaker] = output_path
                
            return speaker_files
        except Exception as e:
            st.error(f"Error separating speakers: {str(e)}")
            return {}

    def create_transcript(self, segments):
        """Create a transcript with speaker labels"""
        transcript = []
        for start, end, speaker in segments:
            line = f"[{start:.2f} --> {end:.2f}] {speaker}"
            transcript.append(line)
        return transcript

def get_binary_file_downloader_html(bin_file, file_label='File'):
    """Generate a download link for a binary file"""
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        return f'<a href="data:application/octet-stream;base64,{b64}" download="{os.path.basename(bin_file)}">{file_label}</a>'
    except Exception as e:
        st.error(f"Error creating download link: {str(e)}")
        return "Download unavailable"

def main():
    st.title("Audio Speaker Diarization")
    st.markdown("Upload an audio file to separate different speakers")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'ogg', 'flac'])
    
    # Max speakers slider
    max_speakers = st.slider("Maximum number of speakers (0 for auto-detect)", 
                             min_value=0, max_value=10, value=0,
                             help="Set to 0 for automatic detection")
    
    # Min segment length
    min_segment = st.slider("Minimum segment length (seconds)", 
                           min_value=1.0, max_value=10.0, value=3.0, step=0.5)
    
    if uploaded_file is not None:
        try:
            # Save the uploaded file to a temporary file in our cache directory
            file_extension = os.path.splitext(uploaded_file.name)[1]
            temp_audio_path = os.path.join(CACHE_DIR, f"temp_audio{file_extension}")
            
            with open(temp_audio_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Display audio player
            st.subheader("Original Audio")
            st.audio(uploaded_file)
            
            # Process button
            if st.button("Process Audio"):
                try:
                    # Initialize the diarization system
                    diarization_system = SimpleDiarization()
                    
                    # Process the audio file
                    with st.spinner("Processing audio... This may take a few minutes for longer files."):
                        start_time = time.time()
                        segments = diarization_system.segment_audio(
                            temp_audio_path, 
                            max_speakers=max_speakers if max_speakers > 0 else None,
                            min_segment_length=min_segment
                        )
                        processing_time = time.time() - start_time
                    
                    if segments:
                        st.success(f"Audio processed successfully in {processing_time:.2f} seconds!")
                        
                        # Get unique speakers
                        speakers = sorted(list(set(spk for _, _, spk in segments)))
                        st.write(f"Detected {len(speakers)} speakers in the audio")
                        
                        # Visualize diarization
                        st.subheader("Speaker Diarization Visualization")
                        fig = diarization_system.visualize_diarization(segments)
                        if fig:
                            st.pyplot(fig)
                        
                        # Create transcript
                        st.subheader("Transcript")
                        transcript = diarization_system.create_transcript(segments)
                        st.text_area("Speaker Timeline", '\n'.join(transcript), height=200)
                        
                        # Separate speakers
                        st.subheader("Separated Speaker Audio")
                        with st.spinner("Separating speakers..."):
                            speaker_files = diarization_system.separate_speakers(temp_audio_path, segments)
                        
                        # Display speakers and download links
                        if speaker_files:
                            # Create columns for each speaker
                            columns = st.columns(min(len(speaker_files), 4))
                            
                            for i, (speaker, file_path) in enumerate(speaker_files.items()):
                                col = columns[i % min(len(speaker_files), 4)]
                                
                                with col:
                                    st.markdown(f"### {speaker}")
                                    # Add audio player
                                    try:
                                        with open(file_path, 'rb') as f:
                                            st.audio(f.read(), format='audio/wav')
                                        # Add download button    
                                        st.markdown(get_binary_file_downloader_html(file_path, f'Download {speaker} audio'), unsafe_allow_html=True)
                                    except Exception as e:
                                        st.error(f"Error loading audio segment: {str(e)}")
                    else:
                        st.warning("No speaker segments were detected. Try adjusting the minimum segment length or using a different audio file.")
                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.error("Check that your audio file is valid and try again.")
                
                finally:
                    # We don't delete the temp audio right away in case we need it for debugging
                    pass
        
        except Exception as e:
            st.error(f"Error handling file upload: {str(e)}")

if __name__ == "__main__":
    main()