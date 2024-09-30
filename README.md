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
### Prerequisites
* Python
* PyTorch
* CUDA

### Installation
```bash
```

### Dataset Download
* Download link: [SAIT dataset](https://o365ust-my.sharepoint.com/:u:/g/personal/byungok_han_office_ust_ac_kr/ESGRDqkurZZMmmGUAOEeIxIBjQpq5KCberHw5PS6t7QrXQ?e=ucKEVz)
* Download link: [SA-Bench](https://o365ust-my.sharepoint.com/:u:/g/personal/byungok_han_office_ust_ac_kr/Eb_LeNjmO3NJjErJMl3fYUMBSMdCGFUBI6EcnW7VgLvX6g?e=JXprgI)

### Training
```bash
```

### Evaluation
```bash
```

### Results
Our experiments demonstrate that the space-aware instruction-tuned model:

* Provides more accurate and concise walking guidance compared to existing VLMs.
* Shows improved understanding of spatial relationships in complex environments.
For detailed results and analysis, refer to the paper.

## Citation


## Acknowledgments
This work was supported by the Institute of Information & communications Technology Planning & Evaluation(IITP) grant funded by the Korea government(MSIT) (RS-2023-00215760, Guide Dog: Development of Navigation AI Technology of a Guidance Robot for the Visually Impaired Person). 

This research (paper) used datasets from ‘The Open AI Dataset Project (AI-Hub, S. Korea)’. All data information can be accessed through ‘AI-Hub (www.aihub.or.kr)’.

## Contact
For questions or collaborations, please contact:

* ByungOk Han: byungok.han@etri.re.kr
* Woo-han Yun: yochin@etri.re.kr

## License
We will add the license information later.
