# Earthquake Diffusion Model

A conditional **Denoising Diffusion Probabilistic Model (DDPM)** that synthesizes realistic seismic waveforms from physical earthquake parameters. Trained on real recordings from Kandilli Observatory (Boğaziçi University).

---

## Overview

Seismic ML models suffer from severe data scarcity — major earthquakes are rare by definition. This project addresses that bottleneck by generating physically conditioned synthetic waveforms that can augment real training datasets.

Given earthquake metadata (depth, azimuth, magnitude, rupture distance), the model generates a full seismic waveform that matches the expected time-frequency characteristics of a real event at those conditions.

---

## Architecture

### Conditioning Strategy — `horizontal_embedding`
The key design choice: rather than using cross-attention or classifier-free guidance, conditioning information is injected **directly into the STFT spectrogram** as horizontal bands in the lower frequency region. Each of the 4 conditioning variables occupies a fixed band of frequency bins before being passed into the UNet.

```
Input: raw seismic waveform (6980 samples, fs=100Hz)
  → STFT → spectrogram [64 freq bins × 128 time bins]
  → horizontal_embedding injects [depth, azimuth, magnitude, dist] into lower 32 freq bins
  → noisy spectrogram [B, 1, 64, 128] fed into UNet
  → UNet predicts added noise (standard DDPM objective)
  → denoised spectrogram → inverse STFT → synthetic waveform
```

### Diffusion Model
- **Backbone:** `UNet2DModel` (HuggingFace Diffusers) — lightweight 3-level encoder-decoder (16→32→64 channels, ~1M params)
- **Scheduler:** `DDPMScheduler` with linear beta schedule, T=1000 timesteps
- **Loss:** MSE between predicted and actual noise
- **Training split:** 90/10 train/test

### Auxiliary MLP
A small 3-layer MLP (4→32→16→2) predicts the **mean and standard deviation** of a waveform directly from the 4 conditioning variables. Used as a fast baseline and for conditioning sanity checks.

---

## Conditioning Variables

| Variable | Description |
|---|---|
| `Depth` | Hypocenter depth (km) |
| `Azimuth` | Source-to-station azimuth (degrees, normalized /15) |
| `Magnitude` | Moment magnitude (scaled ×4) |
| `RuptureDist_km` | Rupture distance (km) |

---

## Data

- `timeseries_Z.csv` — vertical-component seismic waveforms (Z channel), 6980 samples per event
- `processed_events.csv` — earthquake metadata (EventID, magnitude, depth, azimuth, rupture distance)

Data source: **Kandilli Observatory & Earthquake Research Institute, Boğaziçi University**

> Raw data is not included in this repo due to institutional data sharing restrictions.

---

## Usage

**Run the notebook end-to-end:**
```
Earthquake_Diffusion_Model.ipynb
```

**Load a pretrained checkpoint:**
```python
checkpoint = torch.load("horizontal_model.pth", map_location=device)
unet.load_state_dict(checkpoint['unet'])
```

**Generate a synthetic waveform:**
```python
# Define conditioning: [depth, azimuth, magnitude, rupture_dist]
cond = torch.tensor([[15.0, 4.2, 4.8, 32.0]], dtype=torch.float32).to(device)

# Sample from the model at t=999 (fully noisy)
t = torch.tensor([999], dtype=torch.long).to(device)
noise = torch.randn(1, 1, 64, 128).to(device)

x_embedded = embedder(noise.squeeze(1), cond)
denoised = unet(sample=x_embedded, timestep=t).sample

# Convert back to time domain
waveform = inverse_stft_batch(denoised.squeeze(1).cpu().numpy())
```

---

## Results

The model learns to produce spectrograms that visually and structurally match real seismic recordings at the given physical conditions. Qualitative comparisons (real vs. generated STFT and time-domain waveforms) are visualized inline in the notebook.

This model was used at **Kandilli Observatory** to expand training data by **70%**, significantly reducing overfitting in downstream earthquake detection models.

---

## Requirements

```
pip install -r requirements.txt
```

See `requirements.txt` for full dependency list.

---

## Related Work

This project is part of ongoing research at **Boğaziçi University – Kandilli Observatory**. The generated waveforms were used to train the GNN-based real-time earthquake detection system described in the main project.
