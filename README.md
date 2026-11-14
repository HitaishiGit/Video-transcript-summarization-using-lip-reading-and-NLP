### 1. Data preprocessing

-   Match video files with their corresponding alignment files.
-   Extract mouth-region frames from the videos.
-   Resize mouth crops to 96 × 96 pixels.
-   Prepare word-level frame sequences.
-   Split the data by video into training, validation, and test sets.

### 2. Lip-reading model

A CNN-LSTM architecture is used: - Time-distributed convolutional layers
extract visual features from mouth frames. - An LSTM learns temporal
information across frames. - A softmax output layer predicts one of 51
word classes.

### 3. Audio transcription

-   Extract mono audio at 16 kHz from test videos.
-   Use OpenAI Whisper `tiny.en` to generate audio transcripts.

### 4. Evaluation

-   Calculate lip-reading classification accuracy.
-   Calculate Word Error Rate (WER) for lip-reading and Whisper
    transcripts against reference transcripts.
-   Save a confusion matrix and comparison graph.

### 5. Summarization

Use `google/flan-t5-small` to generate a short summary from each Whisper
transcript.

## Experimental Results

The current experimental run produced the following results:

  Evaluation                       Result
  ------------------------------ --------
  Lip-reading test accuracy         4.17%
  Lip-reading WER                  95.83%
  Whisper average WER              69.33%
  Test videos evaluated               100
  Lip-reading word predictions        600
  Audio transcripts generated         100
  Summaries generated                 100

**Interpretation:** The lip-reading model performed poorly in this
experiment, and Whisper also made substantial transcription errors. The
results should be reported as observed; they should not be presented as
evidence of a high-performing system.

WER is an error rate, not ordinary accuracy. It can exceed 100%. The
value `1 − WER` should not be described as standard transcription
accuracy.


## Requirements

The notebooks were developed in Google Colab. Main libraries include:

-   Python
-   TensorFlow / Keras
-   OpenCV
-   MediaPipe
-   pandas and NumPy
-   OpenAI Whisper
-   Transformers and SentencePiece
-   scikit-learn
-   jiwer
-   Matplotlib


## Limitations

-   The experiment uses recordings from a limited speaker subset, which
    restricts speaker diversity.
-   Lip-reading test accuracy was 4.17% in the reported run.
-   Whisper's average WER was 69.33% in the reported run.
-   The summaries can be repetitive or inaccurate when the input
    transcript contains recognition errors or is very short.
-   Limited compute resources restricted the amount of model training.
-   The available single-speaker recordings did not provide suitable
    simultaneous multi-speaker examples; overlapping speech resolution
    was not implemented or evaluated.

## Future Work

-   Train and tune the lip-reading model for longer with suitable GPU
    resources.
-   Experiment with improved visual speech recognition architectures and
    larger, more diverse datasets.
-   Evaluate stronger speech recognition models and suitable decoding
    strategies.
-   Improve summarization quality and evaluate summaries systematically.
-   Add multi-speaker recordings with controlled overlap.
-   Implement and evaluate overlap detection, speaker diarization, and
    speech separation using appropriate reference recordings and
    metrics.
