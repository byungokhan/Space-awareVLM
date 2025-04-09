# Space-Aware Vision-Language Model (SA-VLM)

Welcome to the official repository for Space-Aware Vision-Language Model (SA-VLM), Space-Aware Instruction Tuning (SAIT) and Space-Aware Benchmark (SA-Bench), developed to enhance guide dog robots' assistance for visually impaired individuals.

## Overview
Guide dog robots hold the potential to significantly improve mobility and safety for visually impaired people. However, traditional Vision-Language Models (VLMs) often struggle with accurately interpreting spatial relationships, which is crucial for navigation in complex environments.

Our work introduces:
* **SAIT Dataset**: A dataset automatically generated using [the pipeline](https://github.com/yochin/PathGuidedVQA), designed to enhance VLMs' understanding of physical environments by focusing on virtual paths and 3D surroundings.
* **SA-Bench**: A benchmark with an evaluation protocol to assess the effectiveness of VLMs in delivering walking guidance.
By integrating spatial awareness into VLMs, our approach enables guide dog robots to provide more accurate and concise guidance, improving the safety and mobility of visually impaired users.

## Key Contributions
* **Training Dataset and Benchmark**: We release the SAIT dataset and SA-Bench, providing resources to the community for developing and evaluating space-aware VLMs.
* **Automated Data Generation Pipeline**: An innovative pipeline that automatically generates data focusing on spatial relationships and virtual paths in 3D space.
* **Improved VLM Performance**: Our space-aware instruction-tuned model outperforms state-of-the-art algorithms in providing walking guidance.

## Getting Started

### Dataset Download
1. Due to copyright restrictions, please download the SideGuide images as follows:
   - **If you are located in Korea**, download the "인도 보행 영상" dataset from [AI Hub](https://www.aihub.or.kr/).
   - **If you are outside Korea**, use [this link](https://docs.google.com/forms/d/e/1FAIpQLScBmoVoj0d-omBOVCHGjhRislXP0TYzRqaUJOmJcqN6ylQcxQ/viewform) to request access to the dataset.
2. Download the following datasets:
   - [SAIT](https://o365ust-my.sharepoint.com/:u:/g/personal/byungok_han_office_ust_ac_kr/ERKNDsdlNSlOmtfjTm7hgIUBtZNUxCFetVAp71Wd8WiEVw?e=scaLi3)
   - [SA-Bench](https://o365ust-my.sharepoint.com/:u:/g/personal/byungok_han_office_ust_ac_kr/ETcHGbBsZO9NktZHpZDFePIBgbnCH6s5UoMqAm9hcbjkcg?e=lNLhgS)
3. Dataset Preparation:
   - For the **SAIT dataset**, copy the image files listed in `llava_gd_space_aware.json` from the downloaded SideGuide dataset into the `original_images` folder.
   - For the **SA-Bench dataset**, each image should have a corresponding `.xml` file with the same filename.  
     - If an image is missing but the `.xml` file exists, copy the corresponding image from the SideGuide dataset.

### Installation & Training
* We utilized LLaVA-OneVision as the baseline network. For installation and training instructions, please refer to this [link](https://github.com/LLaVA-VL/LLaVA-NeXT).

### Evaluation
* If you would like to evaluate your VLM using SA-Bench, please use the following script.
```bash
CUDA_VISIBLE_DEVICES=${GPU_NUM} python ./eval/eval_savlm.py \
--model_ckpt_path <path-to-ckpt-dir> \
--eval_db_dir <path-to-SA-Bench-dir> \
--output_dir <path-to-output-dir>
```

### Results
Our experiments demonstrate that the space-aware instruction-tuned model:

* Provides more accurate and concise walking guidance compared to existing VLMs.
* Shows improved understanding of spatial relationships in complex environments.
For detailed results and analysis, refer to the paper.

## Citation
@misc{sait_icra2025,
  title={Space-Aware Instruction Tuning: Dataset and Benchmark for Guide Dog Robots Assisting the Visually Impaired},
  author={ByungOk Han and Woo-han Yun and Beom-Su Seo and Jaehong Kim},
  year={2025},
  eprint={2502.07183},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2502.07183},
}

## Acknowledgments
This work was supported by the Institute of Information & communications Technology Planning & Evaluation(IITP) grant funded by the Korea government(MSIT) (RS-2023-00215760, Guide Dog: Development of Navigation AI Technology of a Guidance Robot for the Visually Impaired Person). 

This research (paper) used datasets from ‘The Open AI Dataset Project (AI-Hub, S. Korea)’. All data information can be accessed through ‘AI-Hub (www.aihub.or.kr)’.

## Contact
For questions or collaborations, please contact:

* ByungOk Han: byungok.han@etri.re.kr
* Woo-han Yun: yochin@etri.re.kr

## License
We will add the license information later.
