# 🎤 Voice Diarization App

A powerful and user-friendly web application for speaker diarization (speaker separation) in audio recordings. Built with Streamlit and SpeechBrain, this app can automatically identify and separate different speakers in audio files.

## ✨ Features

- **🎯 Automatic Speaker Detection**: Automatically identifies and separates different speakers in audio recordings
- **📊 Interactive Timeline Visualization**: Visual representation of speaker segments with color-coded timeline
- **🎵 Audio Segment Playback**: Listen to individual speaker segments directly in the app
- **📝 Transcript Generation**: Generate transcripts with speaker labels
- **🔧 Configurable Parameters**: Adjustable settings for maximum speakers and segment length
- **📱 User-Friendly Interface**: Clean, intuitive web interface built with Streamlit
- **🎨 Real-time Processing**: Live processing with progress indicators
- **💾 Cached Models**: Efficient model caching to avoid repeated downloads

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- FFmpeg (for audio processing)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/KimavathBalajiNayak210/speech-diarization.git
   cd speech-diarization
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv diarization_env
   ```

3. **Activate the virtual environment**
   ```bash
   # On Windows
   diarization_env\Scripts\activate
   
   # On macOS/Linux
   source diarization_env/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the application**
   ```bash
   streamlit run speech.py
   ```

6. **Open your browser**
   Navigate to `http://localhost:8501` to access the application.

## 📖 Usage Guide

### Step 1: Upload Audio File
- Click "Browse files" to upload your audio file
- Supported formats: WAV, MP3, OGG, FLAC
- Maximum file size: 100MB

### Step 2: Configure Settings
- **Maximum Speakers**: Set the expected number of speakers (optional)
- **Minimum Segment Length**: Adjust the minimum duration for speaker segments (default: 3 seconds)

### Step 3: Process Audio
- Click "Process Audio" to start diarization
- The app will automatically:
  - Load the SpeechBrain model
  - Segment the audio
  - Extract speaker embeddings
  - Cluster speakers
  - Generate visualizations

### Step 4: View Results
- **Timeline Visualization**: See color-coded speaker segments
- **Speaker Segments**: Listen to individual speaker segments
- **Download Options**: Export separated audio files and transcripts

## 🏗️ Project Structure

```
speech-diarization/
├── speech.py                 # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                # Project documentation
├── .gitignore              # Git ignore rules
├── pretrained_models/      # Pre-trained model cache
│   └── spkrec-ecapa-voxceleb/
├── speaker-diarization/    # Additional diarization tools
└── ffmpeg-7.1.1/          # FFmpeg binaries
```

## 🔧 Technical Details

### Core Technologies

- **Streamlit**: Web application framework
- **SpeechBrain**: Speech processing toolkit
- **PyTorch**: Deep learning framework
- **TorchAudio**: Audio processing library
- **Scikit-learn**: Machine learning utilities
- **Matplotlib**: Data visualization
- **Pydub**: Audio file manipulation

### Algorithm Overview

1. **Audio Segmentation**: Divides audio into fixed-length segments
2. **Feature Extraction**: Uses ECAPA-TDNN model to extract speaker embeddings
3. **Clustering**: Applies hierarchical clustering to group similar speakers
4. **Speaker Assignment**: Assigns unique speaker IDs to segments
5. **Visualization**: Creates timeline plots and audio segments

### Model Information

- **Model**: ECAPA-TDNN (ECAPA: Emphasized Channel Attention, Propagation and Aggregation)
- **Dataset**: VoxCeleb (Speaker Recognition Dataset)
- **Features**: 192-dimensional speaker embeddings
- **Performance**: State-of-the-art speaker recognition accuracy

## 🎛️ Configuration Options

### Environment Variables

Create a `.env` file in the project root:

```env
# Hugging Face API Token (optional, for model downloads)
HUGGING_FACE_TOKEN=your_token_here
```

### Model Cache

The application automatically caches models in:
- **Windows**: `%USERPROFILE%\.voice_diarization_cache`
- **macOS/Linux**: `~/.voice_diarization_cache`

## 📊 Performance Considerations

- **Processing Time**: Depends on audio length and quality
- **Memory Usage**: Approximately 2-4GB RAM for typical usage
- **GPU Support**: Automatic GPU detection for faster processing
- **Model Loading**: First run may take 1-2 minutes to download models

## 🐛 Troubleshooting

### Common Issues

1. **FFmpeg not found**
   ```bash
   # Install FFmpeg
   # Windows: Download from https://ffmpeg.org/download.html
   # macOS: brew install ffmpeg
   # Ubuntu: sudo apt install ffmpeg
   ```

2. **CUDA/GPU issues**
   - The app works on CPU, but GPU acceleration requires CUDA installation
   - Check PyTorch installation: `python -c "import torch; print(torch.cuda.is_available())"`

3. **Memory errors**
   - Reduce audio file size or segment length
   - Close other applications to free memory

4. **Model download issues**
   - Check internet connection
   - Clear model cache: Delete `.voice_diarization_cache` folder

### Error Messages

- **"Model loading failed"**: Check internet connection and try again
- **"Audio processing error"**: Verify audio file format and integrity
- **"Memory allocation failed"**: Reduce audio file size or increase system memory

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests if applicable
5. Commit your changes: `git commit -m 'Add feature'`
6. Push to the branch: `git push origin feature-name`
7. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
pytest

# Format code
black speech.py

# Lint code
flake8 speech.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **SpeechBrain Team**: For the excellent speech processing toolkit
- **Streamlit Team**: For the amazing web app framework
- **NIELIT**: For project support and guidance
- **Open Source Community**: For the various libraries and tools used

## 📞 Support

- **Issues**: Report bugs and feature requests on [GitHub Issues](https://github.com/KimavathBalajiNayak210/speech-diarization/issues)
- **Discussions**: Join discussions on [GitHub Discussions](https://github.com/KimavathBalajiNayak210/speech-diarization/discussions)
- **Email**: Contact the development team for support

## 🔄 Version History

- **v1.0.0**: Initial release with basic diarization functionality
- **v1.1.0**: Added transcript generation and improved UI
- **v1.2.0**: Enhanced clustering algorithm and performance optimizations

---

**Made with ❤️ using NIELIT**

For more information, visit: [https://github.com/KimavathBalajiNayak210/speech-diarization](https://github.com/KimavathBalajiNayak210/speech-diarization) 