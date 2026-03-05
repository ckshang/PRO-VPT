<div align="center">

# PRO-VPT: Distribution-Adaptive Visual Prompt Tuning via Prompt Relocation
[![iccv](https://img.shields.io/badge/ICCV-2025-blue)](https://iccv.thecvf.com/virtual/2025/poster/567)
[![arXiv](https://img.shields.io/badge/arXiv-2503.13227-b31b1b.svg)](https://arxiv.org/abs/2503.06901)

<img src="https://github.com/ckshang/PRO-VPT/blob/main/imgs/provpt_framework.png" width="500">

</div>

## Abstract
Visual prompt tuning (VPT), i.e., fine-tuning some lightweight prompt tokens, provides an efficient and effective approach for adapting pre-trained models to various downstream tasks. However, most prior art indiscriminately uses a fixed prompt distribution across different tasks, neglecting the importance of each block varying depending on the task. In this paper, we introduce adaptive distribution optimization (ADO) by tackling two key questions: (1) How to appropriately and formally define ADO, and (2) How to design an adaptive distribution strategy guided by this definition? Through empirical analysis, we first confirm that properly adjusting the distribution significantly improves VPT performance, and further uncover a key insight that a nested relationship exists between ADO and VPT. Based on these findings, we propose a new VPT framework, termed PRO-VPT (iterative Prompt RelOcation-based VPT), which adaptively adjusts the distribution built upon a nested optimization formulation. Specifically, we develop a prompt relocation strategy derived from this formulation, comprising two steps: pruning idle prompts from prompt-saturated blocks, followed by allocating these prompts to the most prompt-needed blocks. By iteratively performing prompt relocation and VPT, our proposal can adaptively learn the optimal prompt distribution in a nested optimization-based manner, thereby unlocking the full potential of VPT. Extensive experiments demonstrate that our proposal significantly outperforms advanced VPT methods, e.g., PRO-VPT surpasses VPT by 1.6 pp and 2.0 pp average accuracy, leading prompt-based methods to state-of-the-art performance on VTAB-1k and FGVC benchmarks.


## Datasets
See Tables ii and iii in the Appendix for dataset details.
- Visual Task Adaptation Benchmark (VTAB): The benchmark can be downloaded following the detailed instructions in [VPT](https://github.com/KMnP/vpt/blob/main/VTAB_SETUP.md).
- Fine-Grained Visual Classification tasks (FGVC): The datasets can be directly downloaded from [GPS](https://github.com/FightingFighting/GPS).

## Key Configs
- 🔥PRO-VPT related:
  - MODEL.PROMPT.ADAPTIVE: adaptive or fixed prompt distribution
  - MODEL.PROMPT.PPO: PPO (for RL) or TS (for MAB)
  - MODEL.PROMPT.NUM_TOKENS: prompt length
- Fine-tuning method specification:
  - MODEL.TRANSFER_TYPE
- Vision backbones:
  - DATA.FEATURE: specify which backbone to use
  - MODEL.TYPE: the general backbone type, e.g., "vit" or "swin"
  - MODEL.MODEL_ROOT: folder with pre-trained model checkpoints
- Optimization related: 
  - SOLVER.BASE_LR: lr = base_lr * bs / 256
  - SOLVER.WEIGHT_DECAY
  - DATA.BATCH_SIZE
- Datasets related:
  - DATA.NAME
  - DATA.DATAPATH: where you put the datasets
  - DATA.NUMBER_CLASSES
- Others:
  - OUTPUT_DIR: output dir of the final model and logs

## Open Questions
During experiments, we noticed several phenomena related to prompt-based methods that were not discussed in the paper:
- Prompt-based methods appear to be sensitive to the learning rate and weight decay, which often requires careful hyperparameter tuning for different datasets.
- During training, prompt-based methods may occasionally exhibit sudden spikes in loss/accuracy, followed by rapid recovery. Although these fluctuations do not appear to affect the final performance, their underlying causes remain unclear.
We hope these observations can inform future work and encourage more thorough investigation to gain a deeper understanding of visual prompts.

## Citation
If you find our work helpful in your research, please cite it as:
```bibtex
@inproceedings{shang2025pro,
  title={PRO-VPT: Distribution-Adaptive Visual Prompt Tuning via Prompt Relocation},
  author={Shang, Chikai and Li, Mengke and Zhang, Yiqun and Chen, Zhen and Wu, Jinlin and Gu, Fangqing and Lu, Yang and Cheung, Yiu-ming},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={1558--1568},
  year={2025}
}
```

## Acknowledgement
This repository is built upon [VPT](https://github.com/KMnP/vpt) and [PPO-PyTorch](https://github.com/nikhilbarhate99/PPO-PyTorch). We thank the authors for their excellent codebases.
