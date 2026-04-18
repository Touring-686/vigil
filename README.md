# VIGIL

Official code for **VIGIL: Defending LLM Agents Against Tool Stream Injection via Verify-Before-Commit**.

## News

- 🎉 Our paper has been accepted to ACL2026.
- 📢 The code is now publicly available in this repository.

## Introduction

VIGIL is a security framework for LLM agents.  
It is designed to defend against both **tool stream injection** attacks and **data stream injection** attacks with a **verify-before-commit** paradigm.

Instead of directly executing actions suggested by untrusted tools or runtime feedback, VIGIL first verifies whether an action is consistent with the user’s intent and safety constraints, then commits the action only after validation.

## Highlights

- Defends LLM agents against tool stream injection
- Supports secure and flexible agent reasoning
- Built on a verify-before-commit framework

## Repository

This repository contains the core implementation of **VIGIL**.

## Citation
If you find this project useful, please cite:

```bibtex
@article{Lin2026VIGILDL,
  title={VIGIL: Defending LLM Agents Against Tool Stream Injection via Verify-Before-Commit},
  author={Junda Lin and Zhaomeng Zhou and Zhi Zheng and Shuochen Liu and Tong Xu and Yong Chen and Enhong Chen},
  journal={ArXiv},
  year={2026},
  volume={abs/2601.05755},
  url={https://api.semanticscholar.org/CorpusID:284597109}
}
```