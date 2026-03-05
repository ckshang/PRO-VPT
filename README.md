<div align="center">

# PRO-VPT: Distribution-Adaptive Visual Prompt Tuning via Prompt Relocation
[![iccv](https://img.shields.io/badge/ICCV-2025-blue)](https://iccv.thecvf.com/virtual/2025/poster/567)
[![arXiv](https://img.shields.io/badge/arXiv-2503.13227-b31b1b.svg)](https://arxiv.org/abs/2503.06901)

<img src="https://github.com/ckshang/PRO-VPT/blob/main/imgs/provpt_framework.png" width="500">

</div>

## Open Questions

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

## Datasets
See Tables ii and iii in the Appendix for dataset details.
- Visual Task Adaptation Benchmark (VTAB): The benchmark can be downloaded following the detailed instructions in [VPT](https://github.com/KMnP/vpt/blob/main/VTAB_SETUP.md).
- Fine-Grained Visual Classification tasks (FGVC): The datasets can be directly downloaded from [GPS](https://github.com/FightingFighting/GPS).

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
