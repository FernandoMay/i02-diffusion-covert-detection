# Diffusion Models for Covert Communication Detection

**Paper:** Diffusion Models for Covert Communication Detection in Distributed Systems
**Venue:** CCIOT 2026 — Cloud Computing and Internet of Things
**Authors:** Fernando May Fuentes et al.
**ORCID:** https://orcid.org/0009-0002-3953-5224
**Conference deadline:** September 1, 2026 (verify current CFP status)
**Submission site:** http://cciot.org

## Overview

This repository contains the simulation code and paper manuscript for the diffusion-based covert communication detection framework. The system uses conditional diffusion models to learn normal traffic distributions and detect covert channels through reconstruction probability analysis.

## Structure

```
├── src/
│   └── diffusion_detector.py   # Main detection code
├── tests/
│   └── test_diffusion.py       # Pytest test suite
├── latex/
│   └── paper.tex               # LaTeX manuscript
├── figures/                    # Generated figures
├── data/                       # Simulation data
├── requirements.txt
├── Dockerfile
└── README.md
```

## Quick Start

### Local Setup

```bash
pip install -r requirements.txt
python src/diffusion_detector.py
```

### Docker

```bash
docker build -t diffusion-covert .
docker run diffusion-covert
```

### Run Tests

```bash
pytest tests/ -v
```

## Method

1. **Training:** Diffusion model learns normal traffic distribution
2. **Detection:** Anomaly scoring via reconstruction error
3. **Adaptive Threshold:** RL-enhanced threshold optimization

## Results

| Method | Accuracy | Precision | F1 Score |
|--------|----------|-----------|----------|
| Reconstruction detector | 47.8% | 16.1% | 27.7% |
| Autoencoder | 82.7% | 79.1% | 80.9% |
| GAN-based | 86.7% | 84.2% | 85.4% |

## Citation

```bibtex
@inproceedings{may2026diffusion,
  title={Diffusion Models for Covert Communication Detection in Distributed Systems},
  author={May, Fernando},
  booktitle={Proc. CCIOT 2026},
  year={2026}
}
```

## License

MIT
