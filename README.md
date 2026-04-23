# Deepfake Detection Project

This project implements a Deepfake Detection model using an Xception network. The workflow involves data preprocessing (face extraction), dataset splitting, and model training.

## Project Structure

- `dataset_faces/`: Output directory for extracted faces (created by `preprocess.py`).
- `dataset_split/`: Organized dataset for training (created by `create_splits.py`).
- `preprocess.py`: Extracts faces from video files using OpenCV Haar Cascades.
- `create_splits.py`: Splits the extracted faces into local train/val/test directories.
- `train.py`: Trains the Xception model using PyTorch.
- `requirements.txt`: Python dependencies.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If you encounter issues with PyTorch or MediaPipe on Windows, ensure you have the necessary system libraries (Visual C++ Redistributable).*

## Usage

### 1. Data Preprocessing

Extract faces from the raw video dataset.
```bash
python preprocess.py
```
*Configuration: You can adjust `MAX_VIDEOS` in `preprocess.py` to process more or fewer videos.*

### 2. Create Splits

Organize the extracted faces into training, validation, and test sets.
```bash
python create_splits.py
```

### 3. Training

Train the model.
```bash
python train.py
```
*Configuration: Adjust `BATCH_SIZE`, `EPOCHS`, or `LR` in `train.py` as needed.*

## Model Details

- **Architecture**: Xception (pretrained on ImageNet).
- **Framework**: PyTorch.
- **Input Size**: 299x299.
- **Loss**: Binary Cross Entropy with Logits.
- **Optimizer**: Adam.

---

## CNN + LSTM Approach (Video-Level)

An alternative approach that analyzes **sequences of frames** from each video to detect temporal inconsistencies, instead of classifying individual frames.

### Architecture

- **CNN Backbone**: ResNet-18 (pretrained on ImageNet, frozen) → 512-dim feature per frame.
- **LSTM**: 2-layer LSTM (512 → 256 hidden) processes the frame feature sequence.
- **Classifier**: FC layer from LSTM's last hidden state → 1 output (real/fake).
- **Input Size**: 224×224 × 20 frames per video.

### Usage

#### 1. Preprocess Video Sequences

Extract ordered face sequences from videos (split at the video level):
```bash
python preprocess_sequences.py
```
*Configuration: Adjust `MAX_VIDEOS`, `SEQ_LEN`, or `IMG_SIZE` in `preprocess_sequences.py`.*

#### 2. Train CNN+LSTM Model

```bash
python train_cnn_lstm.py
```
*Configuration: Adjust `BATCH_SIZE`, `EPOCHS`, `LR`, `LSTM_HIDDEN`, or `FREEZE_CNN` in `train_cnn_lstm.py`.*

#### 3. Test on a Single Video (Inference)

To see the prediction (Real/Fake probability) for a new video, run the inference script:
```bash
python predict_video.py path/to/your_video.mp4
```
*Note: By default, this uses `best_cnn_lstm_model.pth`. You can specify a different model with `--model path/to/model.pth`.*

#### 4. Web Interface (GUI)

A simple web interface is provided for drag-and-drop video testing:
```bash
python app.py
```
*This will start a local server. Open the displayed URL (usually `http://127.0.0.1:7860`) in your browser to upload videos and get predictions interactively.*

### Output Files

- `best_cnn_lstm_model.pth` — Best model checkpoint (lowest validation loss).
- `final_cnn_lstm_model.pth` — Final model after all epochs.
- `cnn_lstm_training_history.png` — Training/validation accuracy & loss curves.

---

## ST-ViT: Spatiotemporal Vision Transformer (Novel Architecture)

A **novel architecture** that chains CNN → Bidirectional LSTM → Vision Transformer for deepfake detection. The key novelty is that the ViT operates on **temporally enriched feature tokens** from the LSTM, not on raw image patches — giving it access to both spatial and sequential temporal information.

### Architecture Pipeline

```
Video Input → Frame Extraction (N=20)
    ↓
Stage 1: CNN (ResNet-50, pretrained) — per frame → 2048-dim features
    ↓
Stage 2: Bidirectional LSTM (2 layers) → 512-dim temporal tokens
    ↓
Stage 3: Vision Transformer Encoder (4 layers, 8 heads) → Global context
    ↓
Stage 4: FC(512→256) → ReLU → Dropout(0.5) → FC(256→1) → Sigmoid → Real/Fake
```

### Why This Is Novel

| Approach | Spatial | Temporal | Global Context |
|----------|---------|----------|----------------|
| CNN only | ✅ | ❌ | ❌ |
| CNN + LSTM | ✅ | ✅ | ❌ |
| ViT only | ✅ | ❌ | ✅ |
| **ST-ViT (ours)** | **✅** | **✅** | **✅** |

### Training Strategy

- **Progressive unfreezing**: CNN frozen for first 5 epochs, then fine-tuned at LR/10
- **Separate learning rates**: CNN (1e-5) vs LSTM/ViT/classifier (1e-4)
- **Gradient clipping**: max norm = 1.0 for training stability

### Usage

#### 1. Preprocess (same as CNN+LSTM)
```bash
python preprocess_sequences.py
```

#### 2. Train ST-ViT
```bash
python train_st_vit.py
```

#### 3. Predict on a Single Video
```bash
python predict_video_st_vit.py path/to/video.mp4
python predict_video_st_vit.py path/to/video.mp4 --model best_st_vit_model.pth
```

### Output Files

- `best_st_vit_model.pth` — Best model checkpoint (lowest validation loss).
- `final_st_vit_model.pth` — Final model after all epochs.
- `st_vit_training_history.png` — Training/validation accuracy, loss & F1 curves.

### Evaluation

Compare ST-ViT against baseline models using:
- **Accuracy** — Overall correct predictions
- **F1-Score** — Harmonic mean of precision and recall
- **AUC-ROC** — Area under the ROC curve
