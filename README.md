# Awesome VLA for Robotics

A comprehensive list of excellent research papers, models, datasets, and other resources on Vision-Language-Action (VLA) models in robotics. The relevant survey paper will be released soon.
Contributions are welcome! 

## Table of Contents

- [1. What are VLA Models in Robotics?](#1-what-are-vla-models-in-robotics)
- [2. Survey papers](#2-survey-papers)
- [3. Key VLA Models and Research Papers](#3-key-vla-models-and-research-papers)
    - [3.1 Quick Glance at Key VLA Models](#31-quick-glance-at-key-vla-models)
    - [3.2 By Application Area](#32-by-application-area)
        - [3.2.1 Manipulation](#321-manipulation)
        - [3.2.2 Navigation and Mobile Manipulation](#322-navigation-and-mobile-manipulation)
        - [3.2.3 Human-Robot Interaction (HRI)](#323-human-robot-interaction-hri)
        - [3.2.4 Task Planning / Reasoning](#324-task-planning--reasoning)
        - [3.2.5 Humanoid](#325-humanoid)
        - [3.2.6 Other](#326-other)
    - [3.3 By Technical Approach](#33-by-technical-approach)
        - [3.3.1 Model Architectures](#331-model-architectures)
        - [3.3.2 Action Representation & Generation](#332-action-representation--generation)
        - [3.3.3 Learning Paradigms](#333-learning-paradigms)
        - [3.3.4 Input Modalities & Grounding](#334-input-modalities--grounding)
        - [3.3.5 Fine-tuning Improvement Methods](#335-fine-tuning)
- [4. Datasets and Benchmarks](#4-datasets-and-benchmarks)
    - [4.1 Quick Glance at Datasets and Benchmarks](#41-quick-glance-at-datasets-and-benchmarks)
    - [4.2 Robot Learning Datasets](#42-robot-learning-datasets)
    - [4.3 Simulation Environments](#43-simulation-environments)
    - [4.4 Evaluation Benchmarks](#44-evaluation-benchmarks)
- [5. Challenges and Future Directions](#5-challenges-and-future-directions)
- [6. Related Awesome Lists](#6-related-awesome-lists)

## 1. What are VLA Models in Robotics?

- **Definition:** Vision-Language-Action (VLA) models are a class of multimodal AI systems specifically designed for robotics and Embodied AI. They integrate visual perception (from cameras/sensors), natural language understanding (from text or voice commands), and action generation (physical movements or digital tasks) into a unified framework. Unlike traditional robotic systems that often treat perception, planning, and control as separate modules, VLAs aim for end-to-end or tightly integrated processing, similar to how the human brain processes these modalities simultaneously. The term "VLA" gained prominence with the introduction of the RT-2 model. Generally, a VLA is defined as any model capable of processing multimodal inputs (vision, language) to generate robotic actions for completing embodied tasks.

- **Core Concepts:** The basic idea is to leverage the powerful capabilities of large models (LLMs and VLMs) pre-trained on internet-scale data and apply them to robot control. This involves "grounding" language instructions and visual perception information in the physical world to generate appropriate robot actions. The goal is to achieve greater versatility, dexterity, generalization ability, and robustness compared to traditional methods or early reinforcement learning approaches, enabling robots to work effectively in complex, unstructured environments like homes.

- **Key Components:**

    - **Vision Encoder:** Processes raw visual input (images, videos, sometimes 3D data), using architectures like ViT, CLIP encoders, DINOv2, SigLIP to extract meaningful features (object recognition, spatial reasoning).
    - **Language Understanding:** Employs LLM components (such as Llama, PaLM, GPT variants) to process natural language instructions, map commands to context, and perform reasoning.
    - **Action Decoder/Policy:** Generates robot actions based on integrated visual and language understanding (e.g., end-effector pose, joint velocities, gripper commands, base movements). This is a significant differentiator between VLAs and VLMs, involving techniques like action tokenization, diffusion models, or direct regression.
    - **Alignment Mechanisms:** Uses strategies like projection layers and cross-attention to bridge the gap between different modalities, aligning visual, language, and action representations.

- **Relationship to VLMs and Embodied AI:** VLAs are a specialized category within the field of Embodied AI. They extend Vision-Language Models (VLMs) by explicitly incorporating action generation capabilities. VLMs primarily focus on understanding and generating text based on visual input, while VLAs leverage this understanding to interact with the physical world.

- **Evolution from VLM Adaptation to Integrated Systems:** Early VLA research focused mainly on adapting existing VLMs by simply fine-tuning them to output action tokens (e.g., the initial concept of RT-2 ). However, the field is moving towards more integrated architectures where the action generation components are more sophisticated and co-designed (e.g., diffusion policies, specialized action modules, hierarchical systems like Helix  or NaVILA ). This evolution indicates that the definition of VLA is shifting from merely fine-tuning VLMs to designing specific VLA architectures that better address the unique requirements of robot action generation, while still leveraging the capabilities of VLMs.

## 2. Survey papers

- [2025] A Survey on Vision-Language-Action Models: An Action Tokenization Perspective[[paper](https://arxiv.org/abs/2507.01925)]
- [2025] A Survey on Vision-Language-Action Models for Embodied AI. [[paper](https://arxiv.org/abs/2405.14093)]
- ~~[2025] Survey on Vision-Language-Action Models. [[paper](https://arxiv.org/abs/2502.06851)]~~
- [2025] Vision-Language-Action Models: Concepts, Progress, Applications and Challenges [[paper](https://www.arxiv.org/abs/2505.04769)]
- [2025] OpenHelix: A Short Survey, Empirical Analysis, and Open-Source Dual-System VLA Model for Robotic Manipulation [[paper](https://arxiv.org/pdf/2505.03912)][[project](https://openhelix-robot.github.io/)]
- [2025] Exploring Embodied Multimodal Large Models: Development, Datasets, and Future Directions [[paper](https://arxiv.org/pdf/2502.15336)]
- [2025] Multimodal Fusion and Vision-Language Models: A Survey for Robot Vision [[paper](https://arxiv.org/pdf/2504.02477)] [[project](https://github.com/Xiaofeng-Han-Res/MF-RV)] 
- [2025] Generative Artificial Intelligence in Robotic Manipulation: A Survey [[paper](https://arxiv.org/pdf/2503.03464)] [[project](https://github.com/GAI4Manipulation/AwesomeGAIManipulation)]
- [2025] Neural Brain: A Neuroscience-inspired Framework for Embodied Agents [[paper](https://arxiv.org/pdf/2505.07634)] [[project](https://github.com/CNJianLiu/Neural-Brain-for-Embodied-Agents)] 
- [2024] Aligning Cyber Space with Physical World: A Comprehensive Survey on Embodied AI. [[paper](https://arxiv.org/abs/2407.06886)]
- [2024] A Survey on Robotics with Foundation Models: toward Embodied AI. [[paper](https://arxiv.org/abs/2402.02385)]
- [2024] What Foundation Models can Bring for Robot Learning in Manipulation: A Survey. [[paper](https://arxiv.org/abs/2404.18201)]
- [2024] Towards Generalist Robot Learning from Internet Video: A Survey. [[paper](https://arxiv.org/abs/2404.19664)]
- [2024] Large Multimodal Agents: A Survey. [[paper](https://arxiv.org/abs/2402.15116)]
- [2024] A Survey on Integration of Large Language Models with Intelligent Robots. [[paper](https://arxiv.org/abs/2404.09228)]
- [2024] Vision-Language Models for Vision Tasks: A Survey. [[paper](https://arxiv.org/abs/2304.00685)]
- [2024] A Survey of Embodied Learning for Object-Centric Robotic Manipulation [[paper](https://arxiv.org/pdf/2408.11537)]
- [2024] Vision-language navigation: a survey and taxonomy [[paper](https://arxiv.org/pdf/2108.11544)]
- [2023] Toward general-purpose robots via foundation models: A survey and meta-analysis. [[paper](https://arxiv.org/abs/2312.08782)]
- [2023] Robot learning in the era of foundation models: A survey. [[paper](https://arxiv.org/abs/2311.14379)]


## 3. Key VLA Models and Research Papers

This section is the heart of the resource, listing specific VLA models and influential research papers. Papers are first categorized by major application area, then by key technical contributions. A paper/model may appear in multiple subsections if it is relevant to several categories.

### **3.1 Quick Glance at Key VLA Models**

| Model Name | Key Contribution / Features | Base VLM / Architecture | Action Generation Method | Paper/ Project / Code |
|---|---|---|---|---|
| RT-1 | <details><summary>Details</summary>First large-scale Transformer robot model; Demonstrates scalability on multi-task real-world data; Action discretization</details>  | Transformer (EfficientNet-B3 vision) | Action binning + Token output | [arxiv](https://arxiv.org/abs/2212.06817) / [Project](https://robotics-transformer1.github.io/) / [Code](https://github.com/google-research/robotics_transformer) |
| RT-2 | <details><summary>Details</summary>Transfers web knowledge of VLMs to robot control; Joint fine-tuning of VLM to output action tokens; Shows emergent generalization</details>  | PaLI-X / PaLM-E (Transformer) | Action binning + Token output | [arxiv](https://arxiv.org/abs/2307.15818) / [Project](https://robotics-transformer2.github.io/) |
| PaLM-E | <details><summary>Details</summary>Embodied multimodal language model; Injects continuous sensor data (image, state) into pre-trained LLM; Usable for sequential manipulation planning, VQA, etc.</details>  | PaLM (Transformer) | Outputs subgoals or action descriptions | [ICML](https://openreview.net/pdf?id=VTpHpqM3Cf) / [Project](https://palm-e.github.io/)|
| OpenVLA | <details><summary>Details</summary>Open-source 7B parameter VLA; Based on Llama 2; Trained on OpenX dataset; Outperforms RT-2-X; Shows good generalization and PEFT ability</details>  | Llama 2 (DINOv2 + SigLIP vision) | Action binning + Token output | [arxiv](https://arxiv.org/abs/2406.09246) / [Project](https://openvla.github.io/) / [Code](https://github.com/openvla/openvla) / [HF](https://huggingface.co/collections/openvla/openvla-666b11f9e9f77a2f02a6c740)|
| Helix | <details><summary>Details</summary>General-purpose VLA for humanoid robots; Hierarchical architecture (System 1/2); Full-body control; Multi-robot collaboration; Onboard deployment</details>  | Custom VLM (System 2) + Visuomotor Policy (System 1) | Continuous action output (System 1) | [Paper](https://www.figure.ai/news/helix) / [Project](https://www.figure.ai/news/helix) |
| π0 (Pi-Zero) | <details><summary>Details</summary>General-purpose VLA; Uses Flow Matching to generate continuous action trajectories (50Hz); Cross-platform training (7 platforms, 68 tasks)</details>  | PaliGemma (Transformer) + Action Expert | Flow Matching | [arXiv](https://www.physicalintelligence.company/download/pi0.pdf) / [Project](https://www.physicalintelligence.company/blog/pi0) / [Code](https://github.com/Physical-Intelligence/openpi) / [HF](https://huggingface.co/physical-intelligence) |
| Octo | <details><summary>Details</summary>General-purpose robot model; Trained on OpenX dataset; Flexible input/output conditioning; Often used as a baseline</details>  | Transformer (ViT) | Action binning + Token output / Diffusion Head | [arXiv](https://arxiv.org/abs/2405.12213)/ [Project](https://octo-models.github.io/) / [Code](https://github.com/octo-models/octo) |
| SayCan | <details><summary>Details</summary>Grounds LLM planning in robot affordances; Uses LLM to score skill relevance + value function to score executability</details>  | PaLM (Transformer) + Value Function | Selects pre-defined skills (high-level planner) | [arXiv](https://arxiv.org/abs/2204.01691) / [Project](https://say-can.github.io/) / [Code](https://github.com/google-research/google-research/tree/master/saycan) |
| NaVILA | <details><summary>Details</summary>Two-stage framework for legged robot VLN; High-level VLA outputs mid-level language actions, low-level vision-motor policy executes</details>  | InternVL-Chat-V1.5 (VLM) + Locomotion Policy (RL) | Mid-level language action output (VLA) | [arXiv](https://arxiv.org/abs/2412.04453) / [Project](https://navila-bot.github.io/) |
| VLAS | <details><summary>Details</summary>First end-to-end VLA with direct integration of speech commands; Based on LLaVA; Three-stage fine-tuning for voice commands; Supports personalized tasks (Voice RAG)</details>  | LLaVA (Transformer) + Speech Encoder | Action binning + Token output | [arXiv](https://arxiv.org/abs/2502.13508)
| CoT-VLA | <details><summary>Details</summary>Incorporates explicit Visual Chain-of-Thought (Visual CoT); Predicts future goal images before generating actions; Hybrid attention mechanism</details>  | Llama 2 (ViT vision) | Action binning + Token output (after predicting visual goals) | [arXiv](https://arxiv.org/abs/2503.22020) / [Project](https://cot-vla.github.io/) |
| TinyVLA | <details><summary>Details</summary>Compact, fast, and data-efficient VLA; Requires no pre-training; Uses small VLM + diffusion policy decoder</details>  | MobileVLM V2 / Moondream2 + Diffusion Policy Decoder | Diffusion Policy | [arXiv](https://arxiv.org/abs/2409.12514) / [Project](https://tiny-vla.github.io/) |
| CogACT | <details><summary>Details</summary>Componentized VLA architecture; Specialized action module (Diffusion Action Transformer) conditioned on VLM output; Significantly outperforms OpenVLA / RT-2-X</details>  | InternVL-Chat-V1.5 (VLM) + Diffusion Action Transformer | Diffusion Policy | [arXiv](https://arxiv.org/abs/2411.19650) / [Project](https://cogact.github.io/) |
| TLA | <details><summary>Details</summary>Tactile-Language-Action (TLA) model; sequential tactile feedback via cross-modal language grounding to enable robust policy generation in contact-intensive scenarios.</details>  | Qwen2 7B + LoRA + Qwen2-VL | Qwen2 | [arXiv](https://arxiv.org/abs/2503.08548) / [Project](https://sites.google.com/view/tactile-language-action/) |
| OpenVLA-OFT | <details><summary>Details</summary>Optimized Fine-Tuning (OFT)  | Llama 2 (DINOv2 + SigLIP vision)</details>  | L1 regression  | [arXiv](https://arxiv.org/abs/2502.19645) | [Project](https://openvla-oft.github.io/) |
| RDT |  <details><summary>Details</summary>Robotics Diffusion | InternVL-Chat-V1.5 (VLM) + Diffusion Action Transformer</details>  | Diffusion Policy | [arXiv](https://arxiv.org/abs/2410.07864) | [Project](https://rdt-robotics.github.io/rdt-robotics/) |

### 3.2 By Application Area

#### 3.2.1 Manipulation

Focuses on tasks involving interaction with objects, ranging from simple pick-and-place to complex, dexterous, long-horizon activities. This is a major application area for VLA research.
##### 2025

- **[2025] DreamVLA: A Vision-Language-Action Model Dreamed with Comprehensive World Knowledge**
 [[paper](https://arxiv.org/abs/2507.04447)][[Code](https://github.com/Zhangwenyao1/DreamVLA)][[Project](https://zhangwenyao1.github.io/DreamVLA/)]

- **[2025] TriVLA: A Triple-System-Based Unified Vision-Language-Action Model for General Robot Control**
 [[paper](https://arxiv.org/abs/2507.01424)][[Project](https://zhenyangliu.github.io/TriVLA/)]

 - **[2025] VOTE: Vision-Language-Action Optimization with Trajectory Ensemble Voting**
 [[paper](https://arxiv.org/abs/2507.05116)]

- **[2025] VLA-RL: Towards Masterful and General Robotic Manipulation with Scalable Reinforcement Learning**
 [[paper](https://arxiv.org/abs/2505.18719)][[Code](https://github.com/GuanxingLu/vlarl)]

 - **[2025] CEED-VLA: Consistency Vision-Language-Action Model with Early-Exit Decoding**
 [[paper](https://www.arxiv.org/abs/2506.13725)][[Project](https://irpn-eai.github.io/CEED-VLA/)]

 - **[2025] SwitchVLA: Execution-Aware Task Switching for Vision-Language-Action Models**
 [[paper](https://arxiv.org/abs/2506.03574)][[Project](https://switchvla.github.io/)]

- **[2025] Helix: A Vision-Language-Action Model for Generalist Humanoid Control**
 [[project](https://www.figure.ai/news/helix)]

- **[2025] CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models**  [[paper](https://arxiv.org/pdf/2503.22020)] [[project](https://cot-vla.github.io/)]


- **[2025] Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models** [[paper](https://arxiv.org/abs/2502.19417)] [[project](https://www.physicalintelligence.company/research/hirobot)]

- **[2025] Interactive Post-Training for Vision-Language-Action Models (RIPT-VLA)** [[paper](https://arxiv.org/abs/2505.17016)] [[project](https://ariostgx.github.io/ript_vla/)]


- **[2025] UP-VLA: A Unified Understanding and Prediction Model for Embodied Agent** [[paper](https://arxiv.org/abs/2501.18867)]


- **[2025] NORA: A Small Open-Sourced Generalist Vision Language Action Model for Embodied Tasks** [[paper](https://arxiv.org/abs/2504.19854)] [[project](https://declare-lab.github.io/nora)]


- **[2025] DexVLA: Scaling Vision-Language-Action Models for Dexterous Manipulation Across Embodiments** [[paper](https://arxiv.org/pdf/2502.05855)] [[project](https://dex-vla.github.io/)]

- **[2025] Shake-VLA: Shake, Stir, and Pour with a Dual-Arm Robot: A Vision-Language-Action Model for Automated Cocktail Making** [[paper](https://arxiv.org/pdf/2501.06919)]

- **[2025] VLA Model-Expert Collaboration: Enhancing Vision-Language-Action Models with Human Corrections via Shared Autonomy** [[paper](https://arxiv.org/pdf/2503.04163)] [[project](https://aoqunjin.github.io/Expert-VLA/)]

- **[2025] FAST: Efficient Action Tokenization for Vision-Language-Action Models** [[paper](https://arxiv.org/pdf/2501.09747)] [[project](https://www.pi.website/research/fast)]

- **[2025] HybridVLA: Integrating Diffusion and Autoregressive Action Prediction for Generalist Robot Control** [[paper](https://arxiv.org/pdf/2503.10631)] [[project](https://hybrid-vla.github.io/)] [[code](https://github.com/PKU-HMI-Lab/Hybrid-VLA)]

- **[2025] Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success** [[paper](https://arxiv.org/abs/2502.19645)] [[project(OpenVLA-OFT)](https://openvla-oft.github.io/)]

- **[2025] RoboBrain: A Unified Brain Model for Robotic Manipulation from Abstract to Concrete** [[paper](https://arxiv.org/pdf/2502.21257)] [[project](https://superrobobrain.github.io/)]  [[code](https://github.com/FlagOpen/RoboBrain)]

- **[2025] GRAPE: Generalizing Robot Policy via Preference Alignment** [[paper](https://arxiv.org/pdf/2411.19309)] [[Project](https://grape-vla.github.io/)][[Code](https://github.com/aiming-lab/grape)]

- **[2025] OTTER: A Vision-Language-Action Model with Text-Aware Visual Feature Extraction** [[paper](https://arxiv.org/pdf/2503.03734)] [[project](https://ottervla.github.io/)]

- **[2025] PointVLA: Injecting the 3D World into Vision-Language-Action Models** [[paper](https://arxiv.org/pdf/2503.07511)] [[project](https://pointvla.github.io/)]

- **[2025] AgiBot World Colosseo: Large-scale Manipulation Platform for Scalable and Intelligent Embodied Systems** [[paper](https://agibot-world.com/blog/agibot_go1.pdf)] [[project](https://agibot-world.com/blog/go1)]

- **[2025] DexGraspVLA: A Vision-Language-Action Framework Towards General Dexterous Grasping** [[paper](https://arxiv.org/pdf/2502.20900)] [[project](https://dexgraspvla.github.io/)] 

- **[2025] Interleave-VLA: Enhancing Robot Manipulation with Interleaved Image-Text Instructions** [[paper](https://arxiv.org/abs/2505.02152)] 

- **[2025] EnerVerse: Envisioning Embodied Future Space for Robotics Manipulation** [[paper](https://arxiv.org/pdf/2501.01895)]

- **[2025] VLA-Cache: Towards Efficient Vision-Language-Action Model via Adaptive Token Caching in Robotic Manipulation**[[paper](https://arxiv.org/pdf/2502.02175)]

- **[2025] SafeVLA: Towards Safety Alignment of Vision-Language-Action Model via Safe Reinforcement Learning** [[paper](https://arxiv.org/pdf/2503.03480)] [[project](https://sites.google.com/view/pku-safevla)] 

- **[2025] Accelerating Vision-Language-Action Model Integrated with Action Chunking via Parallel Decoding** [[paper(PD-VLA)](https://arxiv.org/abs/2503.02310)]

- **[2025] Refined Policy Distillation: From VLA Generalists to RL Experts** [[paper(RPD)](https://arxiv.org/abs/2503.05833)]

- **[2025] MoLe-VLA: Dynamic Layer-skipping Vision Language Action Model via Mixture-of-Layers for Efficient Robot Manipulation** [[paper](https://arxiv.org/abs/2503.20384)][[project](https://sites.google.com/view/mole-vla)] [[Code](https://github.com/RoyZry98/MoLe-VLA-Pytorch/)]

- **[2025] MoManipVLA: Transferring Vision-language-action Models for General Mobile Manipulation** [[paper](https://arxiv.org/pdf/2503.13446)] [[project](https://gary3410.github.io/momanipVLA/)]

- **[2025] Dita: Scaling Diffusion Transformer for Generalist Vision-Language-Action Policy** [[paper](https://arxiv.org/pdf/2410.15959v4)] [[project](https://robodita.github.io/)]

- **[2025]  Gemini Robotics: Bringing AI into the Physical World**
 [[report](https://storage.googleapis.com/deepmind-media/gemini-robotics/gemini_robotics_report.pdf)]

- **[2025] ConRFT: A Reinforced Fine-tuning Method for VLA Models via Consistency Policy**, RSS 2025 [[Paper](https://arxiv.org/abs/2502.05450)][[Project](https://cccedric.github.io/conrft/)]

- **[2025] RoboGround: Robotic Manipulation with Grounded Vision-Language Priors** [[paper](https://arxiv.org/pdf/2504.21530)] [[project](https://robo-ground.github.io/)]

- **[2025] ViSA-Flow: Accelerating Robot Skill Learning via Large-Scale Video Semantic Action Flow** [[paper](https://arxiv.org/pdf/2505.01288)] [[project](https://visaflow-web.github.io/ViSAFLOW/)]

- **[2025] Interleave-VLA: Enhancing Robot Manipulation with Interleaved Image-Text Instructions** [[paper](https://arxiv.org/pdf/2505.02152)] [[project](https://interleave-vla-anonymous.github.io/Interleave-VLA-Anonymous/)]

- **[2025] OpenHelix: A Short Survey, Empirical Analysis, and Open-Source Dual-System VLA Model for Robotic Manipulation** [[paper](https://arxiv.org/pdf/2505.03912)] [[project](https://openhelix-robot.github.io/)]

- **[2025] ReinboT: Amplifying Robot Visual-Language Manipulation with Reinforcement Learning** 

- **[2025] CrayonRobo: Object-Centric Prompt-Driven Vision-Language-Action Model for Robotic Manipulation** [[paper](https://arxiv.org/pdf/2505.02166)]
- **[2025] Learning to Act Anywhere with Task-centric Latent Actions [[paper(UniVLA)](https://arxiv.org/pdf/2505.06111)]** [[project](https://github.com/OpenDriveLab/UniVLA)]
- **[2025] Pixel Motion as Universal Representation for Robot Control [[paper](https://arxiv.org/pdf/2505.07817)]** [[project](https://kahnchana.github.io/LangToMo/)]


##### 2024

- **[2024] OpenVLA: An Open-Source Vision-Language-Action Model**
 [[paper](https://arxiv.org/abs/2406.09246)] [[code](https://github.com/openvla/openvla)]

- **[2024] π₀ (Pi-Zero): Our First Generalist Policy**
 [[project](https://www.physicalintelligence.company/blog/pi0)] [[code](https://github.com/Physical-Intelligence/openpi)]

- **[2024] Octo: An Open-Source Generalist Robot Policy [[paper](https://arxiv.org/pdf/2405.12213)] [[project](https://octo-models.github.io/)] [[Code](https://github.com/octo-models/octo)]** 

- **[2024] RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation** [[paper](https://arxiv.org/pdf/2410.07864)]

- **[2024] ReKep: Spatio-Temporal Reasoning of Relational Keypoint Constraints for Robotic Manipulation** [[paper](https://arxiv.org/pdf/2409.01652)]  [[project](https://rekep-robot.github.io/)] [[code](https://github.com/huangwl18/ReKep)]
 
- **[2024] OK-Robot: What Really Matters in Integrating Open-Knowledge Models for Robotics**  [[paper](https://arxiv.org/pdf/2401.12202)] [[project](https://ok-robot.github.io/)] [[code](https://github.com/ok-robot/ok-robot)]

- **[2024] 3D-VLA: A 3D Vision-Language-Action Generative World Model**  [[paper](https://arxiv.org/pdf/2403.09631)]  [[code](https://github.com/UMass-Embodied-AGI/3D-VLA)]
- **[2024] TinyVLA: Towards Fast, Data-Efficient Vision-Language-Action Models for Robotic Manipulation** [[paper](https://arxiv.org/pdf/2409.12514)] [[project](https://tiny-vla.github.io/)]

- **[2024] CogACT: Componentized Vision-Language-Action Models for Robotic Control** [[paper](https://arxiv.org/pdf/2411.19650)] [[project](https://cogact.github.io/)]


- **[2024] RoboMM: All-in-One Multimodal Large Model for Robotic Manipulation** [[paper](https://arxiv.org/pdf/2412.07215v1)][[Code](https://github.com/RoboUniview/RoboMM)]

- **[2024] Diffusion-VLA: Scaling Robot Foundation Models via Unified Diffusion and Autoregression** [[paper](https://arxiv.org/abs/2412.03293)] [[project](https://diffusion-vla.github.io/)]

- **[2024] HiRT: Enhancing Robotic Control with Hierarchical Robot Transformers** [[paper](https://arxiv.org/pdf/2410.05273)]

- **[2024] GR-2: A Generative Video-Language-Action Model with Web-Scale Knowledge for Robot Manipulation** [[paper](https://arxiv.org/pdf/2410.06158)]

- **[2024] DeeR-VLA: Dynamic Inference of Multimodal Large Language Models for Efficient Robot Execution** [[paper](https://arxiv.org/pdf/2411.02359)] [[code](https://github.com/yueyang130/DeeR-VLA)]

- **[2024] RT-Affordance: Affordances are Versatile Intermediate Representations for Robot Manipulation** [[paper](https://arxiv.org/pdf/2411.02704)] 

- **[2024] Moto: Latent Motion Token as the Bridging Language for Robot Manipulation** [[paper](https://arxiv.org/pdf/2412.04445)] [[project](https://chenyi99.github.io/moto/)]

- **[2024] Bi-VLA: Vision-Language-Action Model-Based System for Bimanual Robotic Dexterous Manipulations**  [[paper](https://arxiv.org/abs/2405.06039)]

- **[2024] An Embodied Generalist Agent in 3D World**  [[paper](https://arxiv.org/abs/2311.12871)]

- **[2024] Towards Synergistic, Generalized, and Efficient Dual-System for Robotic Manipulation**  [[paper](https://arxiv.org/abs/2410.08001)][[project](https://opendrivelab.com/legacy/RoboDual/index.html)] [[code](https://github.com/OpenDriveLab/RoboDual)]

##### 2023
- **[2023] RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control**
 [[paper](https://arxiv.org/abs/2307.15818)] [[project](https://robotics-transformer2.github.io/)]

- **[2023] PaLM-E: An Embodied Multimodal Language Model**
 [[paper](https://arxiv.org/abs/2303.03378)] [[project](https://palm-e.github.io/)]

- **[2023] VIMA: General Robot Manipulation with Multimodal Prompts**
 [[paper](https://arxiv.org/abs/2210.03094)] [[project](https://vimalabs.github.io/)]

- **[2023] VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models**  [[paper](https://arxiv.org/pdf/2307.05973)] [[project](https://voxposer.github.io/)] [[code](https://github.com/huangwl18/VoxPoser)]

##### 2022

- **[2022] RT-1: Robotics Transformer for Real-World Control at Scale**
 [[paper](https://arxiv.org/abs/2212.06817)] [[code](https://github.com/google-research/robotics_transformer)]

- **[2022] Do As I Can, Not As I Say: Grounding Language in Robotic Affordances (SayCan)**
 [[paper](https://arxiv.org/abs/2204.01691)] [[project](https://say-can.github.io/)] [[code](https://github.com/google-research/google-research/tree/master/saycan)]

- **[2022] Perceiver-Actor: A Multi-Task Transformer for Robotic Manipulation** [[paper](https://arxiv.org/abs/2209.05451v2)] [[project](https://peract.github.io/)] [[code](https://github.com/peract/peract)]


#### 3.2.2 Navigation and Mobile Manipulation

Focuses on tasks where a robot moves through an environment based on visual input and language instructions. Includes Vision-Language Navigation (VLN) and applications for legged robots.

- **[2025] OctoNav: Towards Generalist Embodied Navigation**
 [[paper](https://arxiv.org/abs/2506.09839)] [[project](https://buaa-colalab.github.io/OctoNav/)]

- **[2025] Do Visual Imaginations Improve Vision-and-Language Navigation Agents?**
 [[paper](https://arxiv.org/pdf/2503.16394)] [[project](https://www.akhilperincherry.com/VLN-Imagine-website/)]

 - **[2025] SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Models**
   [[paper](https://arxiv.org/abs/2501.15830)] [[project](https://spatialvla.github.io/)]

- **[2025] FlexVLN: Flexible Adaptation for Diverse Vision-and-Language Navigation Tasks**
   [[paper](https://arxiv.org/abs/2503.13966)] 

- **[2025] MoManipVLA: Transferring Vision-language-action Models for General Mobile Manipulation** [[paper](https://arxiv.org/pdf/2503.13446)] [[project](https://gary3410.github.io/momanipVLA/)]

- **[2024] NaVILA: Legged Robot Vision-Language-Action Model for Navigation**
   [[paper](https://arxiv.org/abs/2412.04453)] [[project](https://navila-bot.github.io/)]

- **[2024] QUAR-VLA: Vision-Language-Action Model for Quadruped Robots**
   [[paper](https://arxiv.org/abs/2312.14457)] [[project](https://sites.google.com/view/quar-vla)]

- **[2024] NaviLLM: Towards Learning a Generalist Model for Embodied Navigation**
   [[paper](https://arxiv.org/abs/2312.02010)] [[code](https://github.com/zd11024/NaviLLM)]

- **[2024] NaVid: Video-based VLM Plans the Next Step for Vision-and-Language Navigation**
   [[paper](https://arxiv.org/abs/2402.15852)] [[project](https://pku-epic.github.io/NaVid/)]

- **[2023] VLN-SIG: Improving Vision-and-Language Navigation by Generating Future-View Image Semantics**
   [[paper](https://arxiv.org/abs/2304.04907)] [[project](https://jialuli-luka.github.io/VLN-SIG)]

- **[2023] PanoGen: Text-Conditioned Panoramic Environment Generation for Vision-and-Language Navigation**
   [[paper](https://arxiv.org/abs/2305.19195)] [[project](https://pano-gen.github.io/)]



#### 3.2.3 Human-Robot Interaction (HRI)

Focuses on enabling more natural and effective interactions between humans and robots, often using language (text or speech) as the primary interface.

- **[2025] Unveiling the Potential of Vision-Language-Action Models with Open-Ended Multimodal Instructions** [[paper](https://arxiv.org/abs/2505.11214)] (OE-VLA)

- **[2025] VLAS: Vision-Language-Action Model With Speech Instructions For Customized Robot Manipulation** [[paper](https://arxiv.org/abs/2502.13508)]

- **[2025] Shake-VLA: Vision-Language-Action Model-Based System for Bimanual Robotic Manipulations and Liquid Mixing**
   [[paper](https://arxiv.org/abs/2501.06919)]
  
- **[2025] VLA Model-Expert Collaboration for Bi-directional Manipulation Learning**
   [[paper](https://arxiv.org/abs/2503.04163)]
  
- **[2025] Helix: A Vision-Language-Action Model for Generalist Humanoid Control**
   [[project](https://www.figure.ai/news/helix)]
  
- **[2025] CognitiveDrone: A VLA Model and Evaluation Benchmark for Real-Time Cognitive Task Solving and Reasoning in UAVs**
   [[paper](https://arxiv.org/abs/2503.01378)][[project](https://cognitivedrone.github.io/)]

- **[2024] TalkWithMachines: Enhancing Human-Robot Interaction for Interpretable Industrial Robotics Through Large/Vision Language Models**
   [[paper](https://arxiv.org/abs/2412.15462)][[project](https://talk-machines.github.io)]


  
#### 3.2.4 Task Planning / Reasoning

Focuses on using VLA/LLM components for high-level task decomposition, planning, and reasoning, often bridging the gap between complex instructions and low-level actions.

- **[2025] Agentic Robot: A Brain-Inspired Framework for Vision-Language-Action Models in Embodied Agents**  [[paper](https://arxiv.org/abs/2505.23450)]  [[project](https://agentic-robot.github.io/)]

- **[2025] Training Strategies for Efficient Embodied Reasoning (ECoT-Lite)**  [[paper](https://arxiv.org/abs/2505.08243)]

- **[2025] OneTwoVLA: A Unified Vision-Language-Action Model with Adaptive Reasoning** [[paper](https://arxiv.org/abs/2505.11917)] [[project](https://one-two-vla.github.io/)]

- **[2025] Vision-Language-Action Model with Open-World Embodied Reasoning from Pretrained Knowledge** [[paper](https://arxiv.org/abs/2505.21906)] [[project](https://chatvla-2.github.io/)]

- **[2025] Hume: Introducing System-2 Thinking in Visual-Language-Action Model** [[paper](https://arxiv.org/abs/2505.21432)] [[project](https://hume-vla.github.io/)]

 - **[2025] Robotic Control via Embodied Chain-of-Thought Reasoning** [[paper](https://arxiv.org/abs/2407.08693)] [[project](https://embodied-cot.github.io/)][[code](https://github.com/MichalZawalski/embodied-CoT/)]


- **[2025] GR00T N1: An Open Foundation Model for Generalist Humanoid Robots** [[paper](https://arxiv.org/pdf/2503.14734)] [[Code](https://github.com/NVIDIA/Isaac-GR00T)]

- **[2025]  Gemini Robotics: Bringing AI into the Physical World**
 [[report](https://storage.googleapis.com/deepmind-media/gemini-robotics/gemini_robotics_report.pdf)]

- **[2025] GRAPE: Generalizing Robot Policy via Preference Alignment** [[paper](https://arxiv.org/pdf/2411.19309)]

- **[2025] HAMSTER: Hierarchical Action Models For Open-World Robot Manipulation** [[paper](https://arxiv.org/pdf/2502.05485)]

- **[2025] π0.5: A Vision-Language-Action Model with Open-World Generalization**[[paper](https://arxiv.org/pdf/2504.16054)] [[project](https://www.pi.website/blog/pi05)]

- **[2025] Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models**
   [[paper](https://arxiv.org/pdf/2502.19417)] [[project](https://www.physicalintelligence.company/research/hirobot)]
- **[2025] CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models**
   [[paper](https://arxiv.org/abs/2503.22020)] [[project](https://cot-vla.github.io/)]

- **[2025] Embodied-R: Collaborative Framework for Activating Embodied Spatial Reasoning in Foundation Models via Reinforcement Learning** [[paper](https://arxiv.org/pdf/2504.12680)] [[project](https://embodiedcity.github.io/Embodied-R/)] 

- **[2025] RoboBrain: A Unified Brain Model for Robotic Manipulation from Abstract to Concrete** [[paper](https://arxiv.org/pdf/2502.21257)] [[project](https://superrobobrain.github.io/)]  [[code](https://github.com/FlagOpen/RoboBrain)]


- **[2025] Probing a Vision-Language-Action Model for Symbolic States and Integration into a Cognitive Architecture** [[paper](https://arxiv.org/abs/2502.04558)]


- **[2024] RoboMamba: Efficient Vision-Language-Action Model for Robotic Reasoning and Manipulation**
   [[paper](https://arxiv.org/abs/2406.04339)] [[project](https://sites.google.com/view/robomamba-web)]

- **[2024] Improving Vision-Language-Action Models via Chain-of-Affordance**
   [[paper](https://arxiv.org/abs/2412.20451)] [[project](https://chain-of-affordance.github.io/)]

- **[2023] PaLM-E: An Embodied Multimodal Language Model**
   [[paper](https://arxiv.org/abs/2303.03378)] [[project](https://palm-e.github.io/)]

- **[2023] EmbodiedGPT: Vision-Language Pre-Training via Embodied Chain of Thought**
   [[paper](https://arxiv.org/abs/2305.15021)] [[code](https://github.com/OpenGVLab/EmbodiedGPT)]

- **[2022] LLM-Planner: Few-Shot Grounded Planning with Large Language Models**
   [[paper](https://arxiv.org/pdf/2212.04088)] [[project](https://dki-lab.github.io/LLM-Planner/)]

- **[2022] Code as Policies: Language Model Programs for Embodied Control**
   [[paper](https://arxiv.org/abs/2209.07753)] [[project](https://code-as-policies.github.io/)]

- **[2022] Inner Monologue: Embodied Reasoning through Planning with Language Models**
   [[paper](https://arxiv.org/abs/2207.05608)] [[project](https://inner-monologue.github.io/)]
- **[2022] Do As I Can, Not As I Say: Grounding Language in Robotic Affordances (SayCan)**
   [[paper](https://arxiv.org/abs/2204.01691)] [[project](https://say-can.github.io/)] [[code](https://github.com/google-research/google-research/tree/master/saycan)]

#### 3.2.5 Humanoid
- **[2025] GR00T N1: An Open Foundation Model for Generalist Humanoid Robots**
 [[paper](https://arxiv.org/pdf/2503.14734)] [[Code](https://github.com/NVIDIA/Isaac-GR00T)]
- **[2025] Helix: A Vision-Language-Action Model for Generalist Humanoid Control**
 [[project](https://www.figure.ai/news/helix)]

- **[2025] Humanoid-VLA: Towards Universal Humanoid Control with Visual Integration**
   [[paper](https://arxiv.org/pdf/2502.14795)]

#### 3.2.6 Other
- **[2025] Adversarial Attacks on Robotic Vision Language Action Models**
   [[paper](https://arxiv.org/abs/2506.03350)][[Code](https://github.com/eliotjones1/robogcg)]

- **[2025] Vision-Language-Action Model with Open-World Embodied Reasoning from Pretrained Knowledge**
   [[paper](https://arxiv.org/pdf/2505.21906)][[Project](https://chatvla-2.github.io/)](ChatVLA-2)

- **[2025] ChatVLA: Unified Multimodal Understanding and Robot Control with Vision-Language-Action Model**
   [[paper](https://arxiv.org/abs/2502.14420)][[Project](https://chatvla.github.io/)]

- **[2025] OpenDriveVLA: Towards End-to-end Autonomous Driving with Large Vision Language Action Model**
   [[paper](https://arxiv.org/abs/2503.23463)][[Project](https://drivevla.github.io/)][[Code](https://github.com/DriveVLA/OpenDriveVLA)]

- **[2024] OccLLaMA: An Occupancy-Language-Action Generative World Model for Autonomous Driving**
   [[paper](https://arxiv.org/abs/2409.03272)]

- **[2024] DriveVLM: The Convergence of Autonomous Driving and Large Vision-Language Models**
   [[paper](https://arxiv.org/abs/2402.12289)][[Project](https://tsinghua-mars-lab.github.io/DriveVLM/)]

- **[2024] EMMA: End-to-End Multimodal Model for Autonomous Driving**
   [[paper](https://arxiv.org/abs/2410.23262)][[Code](https://github.com/taco-group/OpenEMMA)]





### 3.3 By Technical Approach

#### 3.3.1 Model Architectures

Focuses on the core neural network architectures used in VLA models.

- **Transformer-based:** The dominant architecture, leveraging self-attention mechanisms to integrate vision, language, and action sequences. 
Applications:
    - [RT-1](https://arxiv.org/abs/2212.06817), [RT-2](https://arxiv.org/abs/2307.15818), [Octo](https://arxiv.org/abs/2405.12213), [OpenVLA](https://arxiv.org/abs/2406.09246), [VIMA](https://arxiv.org/abs/2210.03094),[ QUART](https://arxiv.org/abs/2312.14457).

- **Diffusion-based:** Primarily for the action generation component, utilizing the ability of diffusion models to model complex distributions. 
Often combined with a Transformer backbone. Applications:
    -  [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/), [Octo](https://arxiv.org/abs/2405.12213) (can use diffusion head), [3D Diffuser Actor](https://arxiv.org/abs/2402.10885), [SUDD](https://arxiv.org/abs/2307.14535v2), [MDT](https://arxiv.org/abs/2407.05996v1), [RDT-1B](https://arxiv.org/abs/2410.07864v2), [DexVLA](https://arxiv.org/pdf/2502.05855), 
[DiVLA](https://diffusion-vla.github.io/), [TinyVLA](https://arxiv.org/abs/2409.12514), [Hybrid VLA+Diffusion](https://arxiv.org/abs/2503.10631).


- **Hierarchical / Decoupled:** Architectures that separate high-level reasoning/planning (often VLM/LLM-based) from low-level control/execution 
(which may be a separate policy). Applications:
    - [Helix](https://www.figure.ai/news/helix) (System 1/2), [NaVILA](https://arxiv.org/abs/2412.04453) (VLA + Locomotion Policy), [Hi Robot](https://arxiv.org/pdf/2502.19417) (VLM + π0), [SayCan](https://arxiv.org/abs/2204.01691) (LLM + Value Function).
    - [TriVLA](https://arxiv.org/abs/2507.01424)

- **State-Space Models (SSM):** Emerging architectures like Mamba are being explored for their efficiency. Applications:
    - [RoboMamba](https://arxiv.org/abs/2406.04339)

- **Mixture-of-Experts (MoE / MoLE):** Using sparsely activated expert modules for task adaptation or efficiency. 
Applications:
    - [MoRE](https://arxiv.org/abs/2503.08007) (Mixture-of-Robotic-Experts using LoRA), [CogACT](https://arxiv.org/abs/2411.19650), [π0](https://www.physicalintelligence.company/blog/pi0) (uses an MoE-like structure).
    - [MolE-VLA](https://arxiv.org/abs/2503.20384), [ChatVLA](https://arxiv.org/abs/2502.14420)


#### 3.3.2 Action Representation & Generation

Focuses on how robot actions are represented (e.g., discrete tokens vs. continuous vectors) and how models generate them. This is a key area differentiating VLAs from VLMs.

- **Action Tokenization / Discretization:** Representing continuous actions (e.g., joint angles, end-effector pose) as discrete tokens, often via binning.
Used in early/many Transformer-based VLAs like [RT-1](https://arxiv.org/abs/2212.06817), [RT-2](https://arxiv.org/abs/2307.15818) to fit the language modeling paradigm. May have limitations in precision and high-frequency control.

- **Continuous Action Regression:** Directly predicting continuous action vectors. 
Sometimes used in conjunction with other methods or implemented via specific heads. L1 regression is used in [OpenVLA-OFT](https://openvla-oft.github.io/).

- **Diffusion Policies for Actions:** Modeling action generation as a denoising diffusion process. 
Good at capturing multi-modality and continuous spaces. Applications:
    - [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/), [Octo](https://arxiv.org/abs/2405.12213) (diffusion head), [SUDD](https://arxiv.org/abs/2307.14535v2), [MDT](https://arxiv.org/abs/2407.05996v1), [RDT-1B](https://arxiv.org/abs/2410.07864v2), [DexVLA](https://arxiv.org/pdf/2502.05855), [DiVLA](https://diffusion-vla.github.io/), [TinyVLA](https://arxiv.org/abs/2409.12514). Can be slow due to iterative sampling.

- **Flow Matching:** An alternative generative method for continuous actions, used in [π0](https://www.physicalintelligence.company/blog/pi0) for efficient, high-frequency (50Hz) trajectory generation.

- **Action Chunking:** Predicting multiple future actions in a single step, for efficiency and temporal consistency. Increases action dimensionality and inference time when using AR decoding.
Applications:
    -  [CogACT](https://cogact.github.io/), [RoboAgent](https://robopen.github.io/), [π0](https://www.physicalintelligence.company/blog/pi0), [PD-VLA](https://arxiv.org/abs/2503.02310). 

- **Better Decoding Strategy:** Techniques to speed up autoregressive decoding of action chunks. 
    - **Parallel Decoding:**  [PD-VLA](https://arxiv.org/abs/2503.02310).
    - **Early-Exit Decoding:** [CEED-VLA](https://www.arxiv.org/abs/2506.13725)

- **Specialized Tokenizers:** Developing better ways to tokenize continuous action sequences. 
Applications:
    - [FAST](https://arxiv.org/pdf/2501.09747) (designed for dexterous, high-frequency tasks).

- **Point-based Actions:** Using VLMs to predict keypoints or goal locations rather than full trajectories. 
Applications:
    -  [PIVOT](https://arxiv.org/abs/2402.07872), [RoboPoint](https://arxiv.org/abs/2406.10721), [ReKep](https://arxiv.org/pdf/2409.01652).

- **Mid-Level Language Actions:** Generating actions as natural language commands to be consumed by a lower-level policy. Applications:
    - [NaVILA](https://arxiv.org/abs/2412.04453).


#### 3.3.3 Learning Paradigms

Focuses on how VLA models are trained and adapted.

- **Imitation Learning (IL) / Behavior Cloning (BC):** Dominant paradigm, training VLAs to mimic expert demonstrations (often from teleoperation). Heavily reliant on large-scale, diverse, high-quality datasets. Performance is often limited by the quality of the demonstrations.
Applications:
    -  [RT-1](https://arxiv.org/abs/2212.06817), [RT-2](https://arxiv.org/abs/2307.15818), [OpenVLA](https://arxiv.org/abs/2406.09246) （pre-training part）, [Octo](https://arxiv.org/abs/2405.12213), [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/), etc. 
- **Reinforcement Learning (RL):** Used to fine-tune VLAs or train components, allowing models to learn from interaction and potentially exceed demonstrator performance. Challenges include stability and sample efficiency with large models. Applications:
    - [iRe-VLA](https://arxiv.org/abs/2501.16664) (iterative RL/SFT), [MoRE](https://arxiv.org/abs/2503.08007) (RL objective for MoE VLAs handling mixed data), [RPD](https://arxiv.org/abs/2503.05833) (RL-based policy distillation), ConRFT (RL fine-tuning with consistency policies), [SafeVLA](https://arxiv.org/pdf/2503.03480) (Constrained RL for safety), [RIPT-VLA](https://arxiv.org/abs/2505.17016),[VLA-RL](https://arxiv.org/abs/2505.18719),[SimpleVLA-RL](https://github.com/PRIME-RL/SimpleVLA-RL).
    - [Robot-R1](https://arxiv.org/abs/2506.00070)

- **Pre-training & Fine-tuning:** Standard approach, involving pre-training on large datasets (web data for VLM backbones, large robot datasets like OpenX for VLAs) and then fine-tuning on specific tasks or robots.
    - **Fine-tuning by RL**
        - **[2025] ConRFT: A Reinforced Fine-tuning Method for VLA Models via Consistency Policy**, RSS 2025 [[Paper](https://arxiv.org/abs/2502.05450)][[Project](https://cccedric.github.io/conrft/)]

- **Parameter-Efficient Fine-Tuning (PEFT):** Techniques like LoRA to efficiently adapt large VLAs without retraining the entire model, crucial for practical deployment and customization. [MoRE](https://arxiv.org/abs/2503.08007) uses LoRA modules as experts.

- **Distillation:** Training smaller, faster models (students) to mimic the behavior of larger, slower models (teachers). 
Applications:
    - [RPD](https://arxiv.org/abs/2503.05833) (distilling a VLA to an RL policy), [OneDP](https://arxiv.org/abs/2410.21257) (distilling a diffusion policy).

- **Curriculum Learning:** Structuring the learning process, e.g., by embodiment complexity. 
Applications:
    -  [DexVLA](https://arxiv.org/pdf/2502.05855) uses embodied curriculum.

- **Learning from Mixed-Quality Data:** Using techniques (e.g., RL in MoRE) to learn effectively even when demonstration data is suboptimal or contains failures.


#### 3.3.4 Input Modalities & Grounding

Focuses on input data types beyond standard RGB images and text used by VLAs, and how they ground these inputs.

- **Integrating Speech:** Control via spoken commands, potentially capturing nuances missed by text. Requires handling the speech modality directly or via ASR. Applications:
    -  [VLAS](https://arxiv.org/abs/2502.13508) (direct integration), [Shake-VLA](https://arxiv.org/abs/2501.06919) (uses external STT/TTS).

- **Integrating 3D Vision:** Using point clouds, voxels, depth maps, or implicit representations (NeRFs, 3DGS) to provide richer spatial understanding. Applications:
    -  [3D-VLA](https://arxiv.org/pdf/2403.09631), [PerAct](https://arxiv.org/abs/2209.05451v2), [Act3D](https://arxiv.org/abs/2306.17817v2), [RVT](https://arxiv.org/abs/2306.14896v1), [RVT-2](https://arxiv.org/abs/2406.08545v1), [RoboUniView](https://arxiv.org/pdf/2406.18977), DP3, [3D Diffuser Actor](https://arxiv.org/abs/2402.10885), [LEO](https://arxiv.org/abs/2311.12871v3), [3D-LLM](https://arxiv.org/abs/2307.12981v1), [LLM-Grounder](https://arxiv.org/abs/2309.12311), [SpatialVLA](https://spatialvla.github.io/).
    - [Bridge VLA](https://www.arxiv.org/abs/2506.07961)
- **Integrating Proprioception / State:** Incorporating the robot's own state (joint angles, velocities, end-effector pose) as input. Common in many policies, explicitly mentioned in [VLAS](https://arxiv.org/abs/2502.13508), [PaLM-E](https://arxiv.org/abs/2303.03378), [π0](https://www.physicalintelligence.company/blog/pi0) (evaluation requires Simpler fork with proprioception support). OpenVLA initially lacked this, noted as a limitation/future work.

- **Multimodal Prompts:** Handling instructions that include images or video in addition to text. Applications:
    - [VIMA](https://arxiv.org/abs/2210.03094).

- **Grounding:** The process of linking language descriptions or visual perceptions to specific entities, locations, or actions in the physical world or robot representation. Addressed via various techniques like similarity matching, leveraging common-sense knowledge, multimodal alignment, or interaction. LLM-Grounder focuses on open-vocabulary 3D visual grounding.


### 3.3.5 Fine-tuning
- **FT by RL**
    - [ConRFT: A Reinforced Fine-tuning Method for VLA Models via Consistency Policy](https://cccedric.github.io/conrft/), RSS 2025

## 4. Datasets and Benchmarks

This section lists key resources for training and evaluating VLA models. Large-scale, diverse datasets and standardized benchmarks are crucial for progress in the field.

### **4.1 Quick Glance at Datasets and Benchmarks**

| Name | Type | Focus Area | Key Features / Environment | Link | Key Publication |
|---|---|---|---|---|---|
| Open X-Embodiment (OpenX) | Dataset | General Manipulation | <details><summary>Details</summary>Aggregates 20+ datasets, cross-embodiment/task/environment, >1M trajectories | [Project](https://robotics-transformer-x.github.io/) | [arXiv](https://arxiv.org/abs/2310.08864) |
| DROID | Dataset | Real-world Manipulation | <details><summary>Details</summary>Large-scale human-collected data (500+ tasks, 26k hours) | [Project](https://droid-dataset.github.io/) | [arxiv](https://arxiv.org/abs/2403.06037) |
| CALVIN | Dataset / Benchmark | Long-Horizon Manipulation | <details><summary>Details</summary>Long-horizon tasks with language conditioning, Franka arm, PyBullet simulation | [Project](http://calvin.cs.uni-freiburg.de/)| [arxiv](https://arxiv.org/abs/2112.03227) |
| QUARD | Dataset | Quadruped Robot Tasks | <details><summary>Details</summary>Large-scale multi-task dataset (sim + real) for navigation and manipulation | [Project](https://sites.google.com/view/quar-vla) | [ECCV 2024](https://arxiv.org/abs/2312.14457) |
| BEHAVIOR-1K | Dataset / Benchmark | Household Activities | <details><summary>Details</summary>1000 simulated human household activities | [Project](https://behavior.stanford.edu/) | [arxiv](https://arxiv.org/abs/2108.03332) |
| Isaac Sim / Orbit / OmniGibson | Simulator | High-fidelity Robot Simulation | <details><summary>Details</summary>NVIDIA Omniverse-based, physically realistic | [Isaac-sim](https://developer.nvidia.com/isaac-sim), [Orbit](https://isaac-orbit.github.io/), [OmniGibson](https://omnigibson.stanford.edu/) | - |
| Habitat Sim | Simulator | Embodied AI Navigation | <details><summary>Details</summary>Flexible, high-performance 3D simulator | [Project](https://aihabitat.org/) | [arxiv](https://arxiv.org/abs/1904.01201) |
| MuJoCo | Simulator | Physics Engine | <details><summary>Details</summary>Popular physics engine for robotics and RL | [Website](https://mujoco.org/) | - |
| PyBullet | Simulator | Physics Engine | <details><summary>Details</summary>Open-source physics engine, used for CALVIN, etc. | [Website](https://pybullet.org/) | - |
| ManiSkill (1, 2, 3) | Benchmark | Generalizable Manipulation Skills | <details><summary>Details</summary>Large-scale manipulation benchmark based on SAPIEN | [Project](https://maniskill.ai/) | [arxiv](https://arxiv.org/abs/2107.14483) |
| Meta-World | Benchmark | Multi-task / Meta RL Manipulation | <details><summary>Details</summary>50 Sawyer arm manipulation tasks, MuJoCo | [Project](https://meta-world.github.io/)| [arxiv](https://arxiv.org/abs/1910.10897) |
| RLBench | Benchmark | Robot Learning Manipulation | <details><summary>Details</summary>100+ manipulation tasks, CoppeliaSim (V-REP) | [Project](https://sites.google.com/view/rlbench) | [arxiv](https://arxiv.org/abs/1909.12271) |
| VLN-CE / R2R / RxR | Benchmark | Vision-Language Nav | <details><summary>Details</summary>Standard VLN benchmarks, often run in Habitat | [VLN-CE](https://github.com/jacobkrantz/VLN-CE),[R2R-EnvDrop](https://github.com/airsplay/R2R-EnvDrop),[RxR](https://github.com/google-research-datasets/RxR) | - |

### 4.2 Robot Learning Datasets

Large-scale datasets of robot interaction trajectories, often with accompanying language instructions and visual observations. Crucial for training general-purpose policies via imitation learning.

- **Open X-Embodiment (OpenX)** [[Project](https://robotics-transformer-x.github.io/)] - Open X-Embodiment Collaboration. 
    <details><summary>Details</summary>A massive, standardized dataset aggregating data from 20+ existing robot datasets, spanning diverse embodiments, tasks, and environments. Used to train major VLAs like RT-X, Octo, OpenVLA, π0. Contains over 1 million trajectories. </details>

    
- **BridgeData V2** [[Project](https://rail-berkeley.github.io/bridgedata/)] - Walke, H., et al.

    <details><summary>Details</summary>Large dataset collected on a WidowX robot, used for OpenVLA evaluation. </details>

- **DROID** [[Project](https://droid-dataset.github.io/)]  - Manuelli, L., et al.

    <details><summary>Details</summary>Large-scale, diverse, human-collected manipulation dataset (500+ tasks, 26k hours). Used to fine-tune/evaluate OpenVLA, π0. </details>

- **RH20T** [[Project](https://rh20t.github.io/)] - Shao, L., et al.

    <details><summary>Details</summary>Comprehensive dataset with 110k robot clips, 110k human demonstrations, and 140+ tasks.  </details>

- **CALVIN (Composing Actions from Language and Vision)** [[Project](http://calvin.cs.uni-freiburg.de/)] - Mees, O., et al.

    <details><summary>Details</summary>Benchmark and dataset for long-horizon language-conditioned manipulation with a simulated Franka arm in PyBullet.  </details>

- **QUARD (QUAdruped Robot Dataset)** [[Project](https://sites.google.com/view/quar-vla)]  - Tang, J., et al.

    <details><summary>Details</summary>arge-scale multi-task dataset (sim + real) for quadruped navigation and manipulation, released with QUAR-VLA. Contains 348k sim + 3k real clips or 246k sim + 3k real clips.  </details>

- **RoboNet** [[Project](https://www.robonet.wiki/)]  - Dasari, S., et al.

    <details><summary>Details</summary>Early large-scale dataset aggregating data from multiple robot platforms.  </details>

- **BEHAVIOR-1K** [[Project](https://behavior.stanford.edu/)]  - Srivastava, S., et al.

    <details><summary>Details</summary>Dataset of 1000 simulated human household activities, useful for high-level task understanding.  </details>

- **SQA & CSI Datasets** [[arXiv](https://arxiv.org/abs/2502.13508)]- Zhao, W., et al.

    <details><summary>Details</summary>Curated datasets with speech instructions, released with the VLAS model, for speech-vision-action alignment and fine-tuning.  </details>

- **Libero** [[Project](https://libero-project.github.io/datasets)]  - Li, Z., et al.

    <details><summary>Details</summary>* Benchmark suite for robot lifelong learning with procedurally generated tasks. Used in π0 fine-tuning examples.  </details>

- **D4RL (Datasets for Deep Data-Driven Reinforcement Learning)** [[Code](https://github.com/Farama-Foundation/D4RL)]  - Fu, J., et al.

    <details><summary>Details</summary>Standardized datasets for offline RL research, potentially useful for RL-based VLA methods.   </details>

### 4.3 Simulation Environments

Physics-based simulators used to train agents, generate synthetic data, and evaluate policies in controlled settings before real-world deployment.

- **NVIDIA Isaac Sim / Orbit / OmniGibson** [[Isaac-sim](https://developer.nvidia.com/isaac-sim), [Orbit](https://isaac-orbit.github.io/), [OmniGibson](https://omnigibson.stanford.edu/)].

    <details><summary>Details</summary>High-fidelity, physically realistic simulators based on NVIDIA Omniverse. Used for QUAR-VLA, ReKep, ARNOLD, etc.   </details>

- **Habitat Sim** [[Project](https://aihabitat.org/)] - Facebook AI Research (Meta AI).

    <details><summary>Details</summary>Flexible, high-performance 3D simulator for Embodied AI research, especially navigation.   </details>

- **MuJoCo (Multi-Joint dynamics with Contact)** [[Project](https://mujoco.org/)].

    <details><summary>Details</summary>Popular physics engine widely used for robot simulation and RL benchmarks (dm\_control, robosuite, Meta-World, RoboHive).   </details>

- **PyBullet** [[Project](https://pybullet.org/).]

    <details><summary>Details</summary>Open-source physics engine, used for CALVIN and other benchmarks (panda-gym).   </details>

- **SAPIEN** [[Project](https://sapien.ucsd.edu/).]

    <details><summary>Details</summary>Physics simulator focused on articulated objects and interaction. Used for the ManiSkill benchmark.   </details>

- **Gazebo** [[Project](https://gazebosim.org/).]

    <details><summary>Details</summary>Widely used open-source robot simulator, especially in the ROS ecosystem.   </details>

- **Webots** [[Project](https://cyberbotics.com/)].

    <details><summary>Details</summary>Open-source desktop robot simulator.   </details>

- **Genesis** ([GitHub](https://github.com/Genesis-Embodied-AI/Genesis)).

    <details><summary>Details</summary>A newer platform aimed at general robot/Embodied AI simulation.   </details>

- **UniSim** [[arXiv](https://universal-simulator.github.io/unisim/)] - Yang, G., et al

    <details><summary>Details</summary>Learns interactive simulators from real-world videos.   </details>

### 4.4 Evaluation Benchmarks

Standardized suites of environments and tasks used to evaluate and compare the performance of VLA models and other robot learning algorithms.

- **CALVIN** [[Project](https://github.com/mees/calvin)].

    <details><summary>Details</summary>Benchmark for long-horizon language-conditioned manipulation.   </details>

- **ManiSkill (1, 2, 3)** [[Project](https://maniskill.ai/)]

    <details><summary>Details</summary>Large-scale benchmark for generalizable manipulation skills, based on SAPIEN.   </details>

- **Meta-World** [[Project](https://meta-world.github.io/)].

    <details><summary>Details</summary>Multi-task and meta-RL benchmark with 50 different manipulation tasks using a Sawyer arm in MuJoCo.   </details>

- **RLBench** [[Project](https://sites.google.com/view/rlbench)].

    <details><summary>Details</summary>Large-scale benchmark with 100+ manipulation tasks in CoppeliaSim (V-REP).   </details>

- **Franka Kitchen** [[GitHub](https://robotics.farama.org/envs/franka_kitchen/franka_kitchen/)].

    <details><summary>Details</summary>dm\_control-based benchmark involving kitchen tasks with a Franka arm. Used in iRe-VLA.   </details>

- **LIBERO** [[Project](https://libero-project.github.io/datasets)].

    <details><summary>Details</summary>Benchmark for lifelong/continual learning in robot manipulation.   </details>

- **VIMA-Bench** [[Project](https://vimalabs.github.io/)].

    <details><summary>Details</summary>Multimodal few-shot prompting benchmark for robot manipulation.   </details>

- **BEHAVIOR-1K** [[Project](https://behavior.stanford.edu/)].

    <details><summary>Details</summary>Benchmark focused on long-horizon household activities.   </details>

- **VLN-CE / R2R / RxR** [[VLN-CE](https://github.com/jacobkrantz/VLN-CE),[R2R-EnvDrop](https://github.com/airsplay/R2R-EnvDrop),[RxR](https://github.com/google-research-datasets/RxR)].

    <details><summary>Details</summary>Standard benchmarks for Vision-Language Navigation, often run in Habitat. NaVILA is evaluated on these.   </details>

- **Safety-CHORES** [[paper](https://arxiv.org/abs/2503.03480)].

    <details><summary>Details</summary>A new simulated benchmark with safety constraints, proposed for evaluating safe VLA learning.   </details>

- **OK-VQA** [[Project](https://okvqa.allenai.org/)].

    <details><summary>Details</summary>Visual question answering benchmark requiring external knowledge, used to evaluate the general VLM abilities of [PaLM-E](https://arxiv.org/abs/2303.03378).   </details>


## 5. Challenges and Future Directions

- **Data Efficiency & Scalability:** Reducing reliance on massive, expensive, expert-driven datasets. Improving the ability to learn from limited, mixed-quality, or internet-sourced data. Efficiently scaling models and training processes. 

    * Future directions: Improved sample efficiency (RL, self-supervision), sim-to-real transfer, automated data generation, efficient architectures (SSMs, MoEs), data filtering/weighting.
- **Inference Speed & Real-Time Control:** Current large VLAs may be too slow for the high-frequency control loops needed for dynamic tasks or dexterous manipulation. 

    * Future directions: Smaller/compact models ([TinyVLA](https://arxiv.org/abs/2409.12514)), efficient architectures (RoboMamba), parallel decoding ([PD-VLA](https://arxiv.org/abs/2503.02310)), action chunking optimization ([FAST](https://arxiv.org/abs/2501.09747)), model distillation ([OneDP](https://arxiv.org/abs/2410.21257), [RPD](https://arxiv.org/abs/2503.05833) ), hardware acceleration.
- **Robustness & Reliability:** Ensuring consistent performance across variations in environment, lighting, object appearance, disturbances, and unexpected events. Current models can be brittle. 

    * Future directions: Adversarial training, improved grounding, better 3D understanding, closed-loop feedback, anomaly detection, incorporating physical priors, testing frameworks (VLATest).
- **Generalization:** Improving the ability to generalize to new tasks, objects, instructions, environments, and embodiments beyond the training distribution. This is a core promise of VLAs, but remains a challenge. 

    * Future directions: Training on more diverse data (OpenX), effective utilization of VLM pre-training knowledge, compositional reasoning, continual/lifelong learning, better action representations.
- **Safety & Alignment:** Explicitly incorporating safety constraints to prevent harm to the robot, the environment, or humans. Ensuring alignment with user intent. Crucial for real-world deployment. 

    * Future directions: Constrained reinforcement learning ([SafeVLA](https://arxiv.org/pdf/2503.03480)), formal verification, human oversight mechanisms, robust failure detection/recovery, ethical considerations.
- **Dexterity & Contact-Rich Tasks:** Improving performance on tasks requiring fine motor skills, precise force control, and handling complex object interactions. Current VLAs often lag behind specialized methods in this area. 

    * Future directions: Better action representations ([FAST](https://arxiv.org/pdf/2501.09747), Diffusion), integration of tactile sensing, improved physical understanding/simulation, hybrid control approaches.
- **Reasoning & Long-Horizon Planning:** Enhancing the ability for multi-step reasoning, long-horizon planning, and handling complex instructions. 

    * Future directions: Hierarchical architectures, explicit planning modules, chain-of-thought reasoning (visual/textual), memory mechanisms, world models.
- **Multimodality Expansion:** Integrating richer sensory inputs beyond vision + language, such as audio/speech, touch, force, 3D. 

    * Future directions: Developing architectures and alignment techniques for diverse modalities.

## 6. Related Awesome Lists

* Awesome-VLA:

    * https://github.com/yueen-ma/Awesome-VLA
    * https://github.com/OpenHelix-robot/awesome-dual-system-vla
    * https://github.com/Orlando-CS/Awesome-VLA
        
* Awesome-Embodied-AI:
    * https://github.com/HCPLab-SYSU/Embodied_AI_Paper_List
    * https://github.com/dustland/awesome-embodied-ai
    * https://github.com/haoranD/Awesome-Embodied-AI
    * https://github.com/zchoi/Awesome-Embodied-Robotics-and-Agent
* Awesome-Robot-Learning:
    * https://github.com/RayYoh/Awesome-Robot-Learning
    * https://github.com/jonzamora/awesome-robot-learning-envs
    * https://github.com/JadeCong/Awesome-Robot-Learning
* Awesome-Vision-Language-Models:
    * https://github.com/jingyi0000/VLM_survey

## Citation

If you find this repository useful, please consider citing this list:

```
@misc{liu2025vlaroboticspaperslist,
    title = {Awesome-VLA-Robotics},
    author = {Jiaqi Liu},
    journal = {GitHub repository},
    url = {https://github.com/Jiaaqiliu/Awesome-VLA-Robotics},
    year = {2025},
}

```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Jiaaqiliu/Awesome-VLA-Robotics&type=Date)](https://www.star-history.com/#Jiaaqiliu/Awesome-VLA-Robotics&Date)