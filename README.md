# BirdNET Batch Analysis for Audacity Integration

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![BirdNET](https://img.shields.io/badge/BirdNET-0.18.0-green.svg)](https://github.com/kahst/BirdNET-Analyzer)
[![Google Colab](https://img.shields.io/badge/Google-Colab-yellow.svg)](https://colab.research.google.com)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Abstract

This Google Colab notebook provides a **batch audio processing pipeline** for bioacoustic analysis using **BirdNET** (Cornell Lab of Ornithology). The workflow merges multiple audio recordings end-to-end, runs BirdNET species detection with confidence thresholds, generates Audacity-compatible label files for visual inspection, and produces comprehensive visualizations including waveforms, spectrograms, f0 tracks, and formant-like resonance estimates (F1-F3) via Linear Predictive Coding (LPC).

**Key Features**:
- End-to-end audio merging with timeline index generation
- BirdNET automated bird call detection (7-class CNN)
- Audacity label export for manual verification
- Per-detection waveform and spectrogram visualization
- F0 estimation via librosa pyin (YIN algorithm)
- Formant estimation via LPC autocorrelation method
- Overview spectrogram with detection overlays

**Application**: Ornithology research, biodiversity monitoring, acoustic ecology, bioacoustic annotation workflows.

---

## 1. Introduction

### 1.1 BirdNET

**BirdNET** (Kahl et al., 2021) is a deep learning model for automated bird species identification from audio recordings. Developed by the Cornell Lab of Ornithology and Chemnitz University of Technology, it uses a **convolutional neural network (CNN)** trained on millions of bird vocalizations to detect and classify over 3,000 species globally.

**Architecture**:
- Input: Mel spectrogram (time-frequency representation)
- CNN backbone: Residual blocks with batch normalization
- Output: Softmax classification with confidence scores (0-1)

### 1.2 Workflow Overview

The pipeline consists of four stages:

1. **Upload & Merge**: Concatenate audio files end-to-end → export merged WAV + Audacity labels + index CSV
2. **BirdNET Analysis**: Run detection on original files → align detections to merged timeline
3. **Visualize Detections**: Generate waveforms + spectrograms with f0/formant overlays
4. **Overview Spectrogram**: Full-file spectrogram with detection spans labeled

---

## 2. Mathematical Foundations

### 2.1 F0 Estimation (pyin / YIN Algorithm)

**YIN** (de Cheveigné & Kawahara, 2002) estimates fundamental frequency via **autocorrelation**:

**Difference function**:
$$
d(\tau) = \sum_{n=0}^{N-\tau-1} (x[n] - x[n+\tau])^2
$$

**Cumulative mean normalized difference**:
$$
d'(\tau) = \begin{cases}
1 & \tau = 0 \\
\frac{d(\tau)}{\frac{1}{\tau} \sum_{j=1}^{\tau} d(j)} & \tau > 0
\end{cases}
$$

**Period estimation**: First minimum of $d'(\tau)$ below threshold (typically 0.1).

**Fundamental frequency**:
$$
f_0 = \frac{f_s}{\hat{\tau}}, \quad \hat{\tau} = \arg\min_{\tau} d'(\tau)
$$

**pyin** (Mauch & Dixon, 2014) extends YIN with probabilistic voicing detection for more robust f0 tracking.

### 2.2 Formant Estimation (Linear Predictive Coding)

**LPC** models the vocal tract as an all-pole filter:

$$
H(z) = \frac{G}{1 - \sum_{k=1}^{p} a_k z^{-k}}
$$

**Autocorrelation method** (Levinson-Durbin recursion):

$$
R(m) = \sum_{n=0}^{N-m-1} x[n] \cdot x[n+m], \quad m = 0, 1, \ldots, p
$$

Solve for coefficients $a_1, \ldots, a_p$ via:

$$
\sum_{k=0}^{p} a_k R(|m-k|) = 0, \quad m = 1, \ldots, p
$$

**Formant frequencies** from LPC roots:

$$
f_i = \frac{\text{angle}(r_i)}{2\pi} \cdot f_s
$$

where $r_i$ are complex roots of $A(z) = 1 - \sum_{k=1}^{p} a_k z^{-k}$ with positive imaginary parts.

**Note**: For bird vocalizations, "formants" represent **spectral resonances** rather than true vocal tract formants (as in speech).

### 2.3 Mel Spectrogram

**Mel scale** (perceptually-motivated frequency scale):

$$
\text{Mel}(f) = 2595 \log_{10}\left(1 + \frac{f}{700}\right)
$$

**Mel filterbank**: Triangular filters spaced linearly on Mel scale, log-linearly on Hz.

**BirdNET preprocessing**: STFT → Mel filterbank → log magnitude → CNN input.

---

## 3. Pipeline Stages

### 3.1 Stage 1: Upload & Merge

**Process**:
1. Upload audio files to Google Colab
2. Sort by filename (alphabetical) or modification time
3. Concatenate end-to-end with optional gap (default: 0 ms)
4. Normalize loudness (per-file and final mix)
5. Export merged WAV + Audacity labels + index CSV

**Outputs**:
- `merged_no_overlap.wav` — Concatenated audio
- `merged_no_overlap_labels_audacity.txt` — File boundaries as Audacity regions
- `merged_no_overlap_index.csv` — Timeline mapping (filename → start/end times)

### 3.2 Stage 2: BirdNET Analysis

**Process**:
1. Load index CSV from Stage 1
2. Run BirdNET on each original file
3. Align detections to merged timeline using start offsets
4. Filter by confidence threshold (default: 0.25)
5. Export detections CSV + Audacity labels

**Outputs**:
- `merged_no_overlap_birdnet_detections.csv` — All detections with global timestamps
- `merged_no_overlap_birdnet_labels.txt` — Detection regions for Audacity

**Configuration**:
- `MIN_CONF`: Minimum confidence (0-1)
- `LAT`, `LON`, `DATE`: Optional geographic/temporal context for species filtering
- `SPECIES_FILTER`: Optional list of expected species (reduces false positives)

### 3.3 Stage 3: Visualize Detections

**Process** (per detection):
1. Extract audio segment (with padding: default ±0.2s)
2. Compute f0 track via librosa pyin
3. Estimate formants (F1-F3) via LPC (order = max(8, 2 + sr/1000))
4. Calculate spectral features (centroid, peak frequency)
5. Generate waveform plot (time domain)
6. Generate spectrogram with f0 overlay (frequency domain)

**Outputs**:
- `det_XXXX_waveform.png` — Time-domain amplitude plot
- `det_XXXX_spectrogram.png` — STFT spectrogram with f0 track overlay
- `birdnet_detection_summary.csv` — Metrics table (f0, F1-F3, spectral centroid, etc.)

### 3.4 Stage 4: Overview Spectrogram

**Process**:
1. Compute full-file STFT spectrogram
2. Overlay BirdNET detections as shaded time spans
3. Label each detection with species name + confidence

**Output**:
- `overview_spectrogram.png` — Full timeline spectrogram with annotations

---

## 4. Project Structure

```
bird-net-batch-analysis/
├── AudacityTaggerBatchCompiler.ipynb   # Google Colab notebook (all stages)
└── README.md
```

---

## 5. Installation and Usage

### 5.1 Google Colab (Recommended)

1. **Open notebook**: Click "Open in Colab" badge in `AudacityTaggerBatchCompiler.ipynb`
2. **Run setup cells**: Installs dependencies (BirdNET, librosa, pydub, graphviz)
3. **Upload audio files**: Use file upload widget in Cell 1
4. **Configure parameters**:
   - Stage 1: `NORMALIZE_EACH`, `GAP_BETWEEN_MS`, `FORCE_SAMPLE_RATE`
   - Stage 2: `MIN_CONF`, `LAT`, `LON`, `DATE`
   - Stage 3: `F0_MIN_HZ`, `F0_MAX_HZ`, `LPC_ORDER`, `PAD_BEFORE_S`, `PAD_AFTER_S`
5. **Run cells sequentially**: Execute Shift+Enter for each cell
6. **Download outputs**: All files saved to `/content/` (accessible via Colab file browser)

### 5.2 Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MIN_CONF` | 0.25 | Minimum BirdNET confidence (0-1) |
| `F0_MIN_HZ` | 100.0 | Lower bound for f0 search (Hz) |
| `F0_MAX_HZ` | 8000.0 | Upper bound for f0 search (Hz) |
| `LPC_ORDER` | auto | LPC order for formant estimation (typically 8-16) |
| `PAD_BEFORE_S` | 0.2 | Context before detection (seconds) |
| `PAD_AFTER_S` | 0.2 | Context after detection (seconds) |
| `GAP_BETWEEN_MS` | 0 | Silence between merged files (ms) |

### 5.3 Audacity Integration

**Import merged audio**:
1. File → Open → Select `merged_no_overlap.wav`

**Import labels**:
1. File → Import → Labels → Select `merged_no_overlap_labels_audacity.txt` (file boundaries)
2. File → Import → Labels → Select `merged_no_overlap_birdnet_labels.txt` (detections)

**Switch to spectrogram view**:
1. Click track dropdown → Spectrogram
2. Navigate detections using label regions
3. Verify BirdNET classifications visually

---

## 6. Outputs Summary

### Generated Files

| File | Description |
|------|-------------|
| `merged_no_overlap.wav` | Concatenated audio (all files end-to-end) |
| `merged_no_overlap_labels_audacity.txt` | File boundary labels for Audacity |
| `merged_no_overlap_index.csv` | Timeline index (filename → start/end) |
| `merged_no_overlap_birdnet_detections.csv` | BirdNET detections with global timestamps |
| `merged_no_overlap_birdnet_labels.txt` | Detection labels for Audacity |
| `birdnet_detection_summary.csv` | Per-detection metrics (f0, F1-F3, etc.) |
| `det_XXXX_waveform.png` | Waveform plot for detection XXXX |
| `det_XXXX_spectrogram.png` | Spectrogram plot for detection XXXX |
| `overview_spectrogram.png` | Full-file spectrogram with overlays |

---

## 7. References

### BirdNET

- Kahl, S., Wood, C. M., Eibl, M., & Klinck, H. (2021). "BirdNET: A deep learning solution for avian diversity monitoring". *Ecological Informatics*, 61, 101236. DOI: [10.1016/j.ecoinf.2021.101236](https://doi.org/10.1016/j.ecoinf.2021.101236)
- BirdNET GitHub: [https://github.com/kahst/BirdNET-Analyzer](https://github.com/kahst/BirdNET-Analyzer)

### F0 Estimation

- de Cheveigné, A., & Kawahara, H. (2002). "YIN, a fundamental frequency estimator for speech and music". *Journal of the Acoustical Society of America*, 111(4), 1917-1930. DOI: [10.1121/1.1458024](https://doi.org/10.1121/1.1458024)
- Mauch, M., & Dixon, S. (2014). "pYIN: A fundamental frequency estimator using probabilistic threshold distributions". *Proc. ICASSP*, 659-663.

### Linear Predictive Coding

- Makhoul, J. (1975). "Linear prediction: A tutorial review". *Proceedings of the IEEE*, 63(4), 561-580.
- Rabiner, L. R., & Schafer, R. W. (2011). *Theory and Applications of Digital Speech Processing*. Prentice Hall.

### Librosa

- McFee, B., et al. (2015). "librosa: Audio and Music Signal Analysis in Python". *Proc. SciPy*, 18-24. DOI: [10.25080/Majora-7b98e3ed-003](https://doi.org/10.25080/Majora-7b98e3ed-003)

---

## 8. License

MIT License (2025)

Copyright (c) 2025 George Redpath

---

## 9. Author

**George Redpath** (Ziforge)
GitHub: [@Ziforge](https://github.com/Ziforge)
Focus: Bioacoustics, ornithology, automated species detection

---

## 10. Acknowledgments

- **Cornell Lab of Ornithology** — BirdNET model development
- **Google Colab** — Cloud computing platform
- **Audacity Team** — Open-source audio editor
- **librosa Contributors** — Audio analysis library

---

## 11. Citation

```bibtex
@misc{redpath2025birdnet,
  author = {Redpath, George},
  title = {BirdNET Batch Analysis for Audacity Integration},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Ziforge/bird-net-batch-analysis}
}
```

---

Built for bioacoustic research and ornithological field studies.
