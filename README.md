# ERExtractor: Automated Multimodal Extraction of Enzyme-Catalyzed Reaction Data

<p align="center">
  <img src="https://github.com/user-attachments/assets/bb5d4e69-fa2b-4f0e-9fe8-704254898981" alt="ERExtractor Overview" width="800"/>
</p>


## 📌 Introduction
This repository contains the official implementation of **ERExtractor**, an automated and extensible platform for multimodal extraction of enzyme-catalyzed reaction data from scientific literature.  
The system integrates **tables, molecular diagrams, enzyme sequences, and experimental conditions** into structured, machine-readable datasets for downstream AI-driven modeling.

## 🚀 Features
- ✅ Unified framework combining **deep learning** and **large language models**  
- ✅ Supports **tables, figures, and text extraction**  
- ✅ Benchmarked on **1,000+ annotated tables and 5,000 biological fields**  
- ✅ Achieves **89.9% accuracy** on table recognition and **98%+ accuracy** on molecular recognition

## 📊 Results

<table>
<tr>
<td>

| Method       | Acc(%) | Gain   |
|--------------|--------|--------|
| TableMaster  | 77.90* | -      |
| LGPMA        | 65.74* | -      |
| SLANet       | 86.0   | -      |
| **Ours**     | **89.9** | **3.9%** |

</td>
<td>

<p align="center">
  <img src="https://github.com/user-attachments/assets/92745d5b-474f-47c8-a105-45d8cc2b69e8" alt="ERExtractor Overview" width="600"/>
</p>
</td>
</tr>

</table>
<img width="1912" height="682" alt="2025-09-05 21 25 18" src="https://github.com/user-attachments/assets/8e094f2d-1489-4eb3-abee-532c6f641cc9" />



<!-- ## 📂 Repository Structure -->

## ⚡ Quick Start
You can explore **ERExtractor** directly through our online platform:  
🔗 [ERExtractor Platform](https://zpaper.zelixir.com/)
> 🛠️ The source code will be released upon the acceptance and publication of our paper.  

## 🌐 Links
- 📄 [Preprint on arXiv](https://arxiv.org/abs/2508.09995)
- 💻 [Project Website](https://zpaper.zelixir.com/)
- 📦 [Dataset & Results on GitHub Releases](https://github.com/AIForgeRyan/ERExtractor/releases)

## 📬 Contact
Ryan（CAS）
📧 ryan5zh5@gmail.com

## 📖 Citation
If you find this work useful, please cite:
```bibtex
@article{zhou2025zerextractor,
  title={zERExtractor: An Automated Platform for Enzyme-Catalyzed Reaction Data Extraction from Scientific Literature},
  author={Zhou, Rui and Ma, Haohui and Xin, Tianle and Zou, Lixin and Hu, Qiuyue and Cheng, Hongxi and Lin, Mingzhi and Guo, Jingjing and Wang, Sheng and Zhang, Guoqing and others},
  journal={arXiv preprint arXiv:2508.09995},
  year={2025}
}


