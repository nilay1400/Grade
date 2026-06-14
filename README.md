# GRADE

## GRADE: A Scalable Framework for Quantitative Reliability Assessment of Deep Neural Networks

Official implementation of the paper:

> GRADE: A Scalable Framework for Quantitative Reliability Assessment of Deep Neural Networks

published in IEEE Transactions on Reliability, 2026.

## Overview

GRADE is a scalable analytical framework for quantitative reliability assessment of Deep Neural Networks (DNNs). The framework estimates the reliability of DNNs under hardware faults without requiring extensive fault-injection campaigns, enabling efficient reliability analysis for large-scale models.

## Features

- Analytical reliability estimation
- Scalable to large DNN architectures
- Reduced computational cost compared to exhaustive fault injection
- Supports reliability evaluation under hardware-induced faults

## Repository Structure

```text
├── resnet_grade.py          # Reliability analysis for a sample DNN
└── README.md

## Requirements
- Python 3.10+
- PyTorch 2.x
- NumPy

## Quick Start
Run reliability analysis for ResNet:
```bash
   python3 resnet_grade.py

## Citation
If you use this work in your research, please cite:

   ```bash
      @article{nazari2026grade,
        title={GRADE: A Scalable Framework for Quantitative Reliability Assessment of Deep Neural Networks},
        author={Nazari, Samira and Azarpeyvand, Ali and Afsharchi, Mohsen},
        journal={IEEE Transactions on Reliability},
        year={2026}
      }
