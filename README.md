# Video Transcript Summarization Using Lip Reading and NLP

## Project Overview

This project explores an audiovisual pipeline for speech transcription
and text summarization using the GRID audiovisual corpus. It combines:

1.  **Lip-reading:** A CNN-LSTM model processes sequences of
    mouth-region images and predicts words.
2.  **Audio transcription:** OpenAI Whisper (`tiny.en`) generates
    transcripts from extracted audio.
3.  **Evaluation:** Lip-reading predictions are compared with reference
    transcripts, while Whisper is evaluated using Word Error Rate (WER).
4.  **Summarization:** FLAN-T5-small generates short summaries from
    Whisper transcripts.

> **Implementation scope:** The current implementation covers
> lip-reading, audio transcription, transcript evaluation, and
> summarization. Overlapping speech detection, speaker diarization, and
> speech separation have **not** been implemented or evaluated. They are
> proposed as future work.

## Dataset

The project uses the **GRID audiovisual corpus**. The experiments
described in this repository used recordings from speaker `s1`, with
corresponding video (`.mpg`) and alignment (`.align`) files.

The alignment files provide word-level timing and reference words used
to prepare the lip-reading sequences and evaluate transcripts.

Dataset files are not included in this repository. Obtain the GRID
corpus through its official distribution terms and follow its license
and usage restrictions.

## Methodology

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

## Repository Structure

A suggested repository layout is:

``` text
video-transcript-summarization/
├── README.md
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_lipreading_model.ipynb
│   ├── 03_audio_transcription.ipynb
│   └── 04_evaluation_and_summarization.ipynb
├── results/
│   ├── final_evaluation.csv
│   ├── lipreading_evaluation.csv
│   ├── wer_comparison.png
│   └── confusion_matrix.png
└── .gitignore
```

Adjust the notebook and result filenames to match the files actually
added to the repository. Do not upload the GRID dataset, large extracted
frame folders, audio files, or model checkpoints unless you have a clear
reason and permission to distribute them.

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

Install the packages in a Colab notebook as needed. Package versions may
need adjustment to match the runtime environment.

## How to Run

1.  Open the notebooks in Google Colab.
2.  Mount Google Drive and set the paths to your GRID dataset.
3.  Run preprocessing to create the mouth-region sequences and
    alignment-based labels.
4.  Train or load the saved CNN-LSTM model.
5.  Generate lip-reading predictions for the test set.
6.  Extract test audio and run Whisper transcription.
7.  Calculate WER and generate evaluation plots.
8.  Generate summaries with FLAN-T5-small.

The notebooks depend on the dataset and intermediate files created in
earlier stages. Update paths before running them in a different
environment.

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

## Acknowledgements

This project uses the GRID audiovisual corpus and open-source
machine-learning and NLP libraries. Please acknowledge the dataset
creators and follow the dataset's terms when using or redistributing it.

## Disclaimer

This is an academic experimental project. Reported metrics apply only to
the described experimental run and dataset split. They are not
guarantees of performance on other speakers, datasets, or real-world
conversations.
