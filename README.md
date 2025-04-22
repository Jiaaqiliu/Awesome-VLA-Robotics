# Awesome VLA for Robotics

A comprehensive list of excellent research papers, models, datasets, and other resources on Vision-Language-Action (VLA) models in robotics. Contributions are welcome! 

## Table of Contents

- [What are VLA Models in Robotics?](#what-are-vla-models-in-robotics)
- [Survey papers](#survey-papers)
- [Key VLA Models and Research Papers](#key-vla-models-and-research-papers)
    - [Quick Glance at Key VLA Models](#q)
- [By Application Area](#by-application-area)
    - [Manipulation](#manipulation)
    - [Navigation](#navigation)
    - [Human-Robot Interaction (HRI)](#human-robot-interaction-hri)
    - [Task Planning / Reasoning](#task-planning--reasoning)
- [By Technical Approach](#by-technical-approach)
    - [Model Architectures](#model-architectures)
    - [Action Representation & Generation](#action-representation--generation)
    - [Learning Paradigms](#learning-paradigms)
    - [Input Modalities & Grounding](#input-modalities--grounding)
- [Datasets and Benchmarks](#datasets-and-benchmarks)
    - [Quick Glance at Datasets and Benchmarks](#quick-glance-at-datasets-and-benchmarks)
    - [Robot Learning Datasets](#robot-learning-datasets)
    - [Simulation Environments](#simulation-environments)
    - [Evaluation Benchmarks](#evaluation-benchmarks)
- [Related Awesome Lists](#related-awesome-lists)
- [Challenges and Future Directions](#challenges-and-future-directions)

## What are VLA Models in Robotics?

* **Definition:** Vision-Language-Action (VLA) models are a class of multimodal AI systems specifically designed for robotics and Embodied AI. They integrate visual perception (from cameras/sensors), natural language understanding (from text or voice commands), and action generation (physical movements or digital tasks) into a unified framework. Unlike traditional robotic systems that often treat perception, planning, and control as separate modules, VLAs aim for end-to-end or tightly integrated processing, similar to how the human brain processes these modalities simultaneously. The term "VLA" gained prominence with the introduction of the RT-2 model. Generally, a VLA is defined as any model capable of processing multimodal inputs (vision, language) to generate robotic actions for completing embodied tasks.

* **Core Concepts:** The basic idea is to leverage the powerful capabilities of large models (LLMs and VLMs) pre-trained on internet-scale data and apply them to robot control. This involves "grounding" language instructions and visual perception information in the physical world to generate appropriate robot actions. The goal is to achieve greater versatility, dexterity, generalization ability, and robustness compared to traditional methods or early reinforcement learning approaches, enabling robots to work effectively in complex, unstructured environments like homes.

* **Key Components:**

    * **Vision Encoder:** Processes raw visual input (images, videos, sometimes 3D data), using architectures like ViT, CLIP encoders, DINOv2, SigLIP to extract meaningful features (object recognition, spatial reasoning).
    * **Language Understanding:** Employs LLM components (such as Llama, PaLM, GPT variants) to process natural language instructions, map commands to context, and perform reasoning.
    * **Action Decoder/Policy:** Generates robot actions based on integrated visual and language understanding (e.g., end-effector pose, joint velocities, gripper commands, base movements). This is a significant differentiator between VLAs and VLMs, involving techniques like action tokenization, diffusion models, or direct regression.
    * **Alignment Mechanisms:** Uses strategies like projection layers and cross-attention to bridge the gap between different modalities, aligning visual, language, and action representations.

* **Relationship to VLMs and Embodied AI:** VLAs are a specialized category within the field of Embodied AI. They extend Vision-Language Models (VLMs) by explicitly incorporating action generation capabilities. VLMs primarily focus on understanding and generating text based on visual input, while VLAs leverage this understanding to interact with the physical world.

* **Evolution from VLM Adaptation to Integrated Systems:** Early VLA research focused mainly on adapting existing VLMs by simply fine-tuning them to output action tokens (e.g., the initial concept of RT-2 ). However, the field is moving towards more integrated architectures where the action generation components are more sophisticated and co-designed (e.g., diffusion policies, specialized action modules, hierarchical systems like Helix  or NaVILA ). This evolution indicates that the definition of VLA is shifting from merely fine-tuning VLMs to designing specific VLA architectures that better address the unique requirements of robot action generation, while still leveraging the capabilities of VLMs.

## Survey papers

* A Survey on Vision-Language-Action Models for Embodied AI. [[paper](https://arxiv.org/abs/2405.14093)]
* Aligning Cyber Space with Physical World: A Comprehensive Survey on Embodied AI. [[paper](https://arxiv.org/abs/2407.06886)]
* Robot learning in the era of foundation models: A survey. [[paper](https://arxiv.org/abs/2311.14379)]
* A Survey on Robotics with Foundation Models: toward Embodied AI. [[paper](https://arxiv.org/abs/2402.02385)]
* Toward general-purpose robots via foundation models: A survey and meta-analysis. [[paper](https://arxiv.org/abs/2312.08782)]
* What Foundation Models can Bring for Robot Learning in Manipulation: A Survey. [[paper](https://arxiv.org/abs/2404.18201)]
* Towards Generalist Robot Learning from Internet Video: A Survey. [[paper](https://arxiv.org/abs/2404.19664)]
* Large Multimodal Agents: A Survey. [[paper](https://arxiv.org/abs/2402.15116)]
* A Survey on Large Language Models for Automated Planning. [[paper](https://arxiv.org/abs/2502.12435)]
* A Survey on Integration of Large Language Models with Intelligent Robots. [[paper](https://arxiv.org/abs/2404.09228)]
* Vision-Language Models for Vision Tasks: A Survey. [[paper](https://arxiv.org/abs/2304.00685)]
* Survey on Vision-Language-Action Models. [[paper](https://arxiv.org/abs/2502.06851)]
* From Screens to Scenes: A Survey of Embodied AI in Healthcare. [[paper](https://arxiv.org/abs/2501.07468)]


## Key VLA Models and Research Papers

This section is the heart of the resource, listing specific VLA models and influential research papers. Papers are first categorized by major application area, then by key technical contributions. A paper/model may appear in multiple subsections if it is relevant to several categories.

### **Quick Glance at Key VLA Models**

| Model Name | Key Contribution / Features | Base VLM / Architecture | Action Generation Method | Key Publication(s) | Project / Code |
|---|---|---|---|---|---|
| RT-1 | First large-scale Transformer robot model; Demonstrates scalability on multi-task real-world data; Action discretization | Transformer (EfficientNet-B3 vision) | Action binning + Token output | [arxiv](https://arxiv.org/abs/2212.06817) | [Project](https://robotics-transformer1.github.io/) / [Code](https://github.com/google-research/robotics_transformer) |
| RT-2 | Transfers web knowledge of VLMs to robot control; Joint fine-tuning of VLM to output action tokens; Shows emergent generalization | PaLI-X / PaLM-E (Transformer) | Action binning + Token output | [arxiv](https://arxiv.org/abs/2307.15818) | [Project](https://robotics-transformer2.github.io/) |
| PaLM-E | Embodied multimodal language model; Injects continuous sensor data (image, state) into pre-trained LLM; Usable for sequential manipulation planning, VQA, etc. | PaLM (Transformer) | Outputs text (subgoals or action descriptions), needs low-level policy to execute | [ICML 2023](https://openreview.net/pdf?id=VTpHpqM3Cf) | [Project](https://palm-e.github.io/)|
| OpenVLA | Open-source 7B parameter VLA; Based on Llama 2; Trained on OpenX dataset; Outperforms RT-2-X; Shows good generalization and PEFT ability | Llama 2 (DINOv2 + SigLIP vision) | Action binning + Token output (raw) / L1 regression (OFT) | [arxiv](https://arxiv.org/abs/2406.09246) | [Project](https://openvla.github.io/) / [Code](https://github.com/openvla/openvla) / [HF](https://huggingface.co/collections/openvla/openvla-666b11f9e9f77a2f02a6c740)|
| Helix | General-purpose VLA for humanoid robots; Hierarchical architecture (System 1/2); Full-body control; Multi-robot collaboration; Onboard deployment | Custom VLM (System 2) + Visuomotor Policy (System 1) | Continuous action output (System 1) | https://www.figure.ai/news/helix | [Project](https://www.figure.ai/news/helix) |
| π0 (Pi-Zero) | General-purpose VLA; Uses Flow Matching to generate continuous action trajectories (50Hz); Cross-platform training (7 platforms, 68 tasks) | PaliGemma (Transformer) + Action Expert | Flow Matching | [arXiv 2024](https://arxiv.org/abs/2410.08532) | [Project](https://www.physicalintelligence.company/research/pi-zero) / [Code](https://github.com/Physical-Intelligence/openpi) / [HF](https://huggingface.co/physical-intelligence) |
| Octo | General-purpose robot model; Trained on OpenX dataset; Flexible input/output conditioning; Often used as a baseline | Transformer (ViT) | Action binning + Token output / Diffusion Head | [arXiv 2024](https://arxiv.org/abs/2405.12213)| [Project](https://octo-models.github.io/) / [Code](https://github.com/octo-models/octo) |
| SayCan | Grounds LLM planning in robot affordances; Uses LLM to score skill relevance + value function to score executability | PaLM (Transformer) + Value Function | Selects pre-defined skills (high-level planner) | [arXiv 2024](https://arxiv.org/abs/2204.01691) | [Project](https://say-can.github.io/) / [Code](https://github.com/google-research/google-research/tree/master/saycan) |
| NaVILA | Two-stage framework for legged robot VLN; High-level VLA outputs mid-level language actions, low-level vision-motor policy executes | InternVL-Chat-V1.5 (VLM) + Locomotion Policy (RL) | Mid-level language action output (VLA) | [arXiv 2024](https://arxiv.org/abs/2412.04453) | [Project](https://navila-bot.github.io/) |
| VLAS | First end-to-end VLA with direct integration of speech commands; Based on LLaVA; Three-stage fine-tuning for voice commands; Supports personalized tasks (Voice RAG) | LLaVA (Transformer) + Speech Encoder | Action binning + Token output | [arXiv 2024](https://arxiv.org/abs/2502.13508) | - |
| CoT-VLA | Incorporates explicit Visual Chain-of-Thought (Visual CoT); Predicts future goal images before generating actions; Hybrid attention mechanism | Llama 2 (ViT vision) | Action binning + Token output (after predicting visual goals) | [arXiv 2024](https://arxiv.org/abs/2503.22020) | [Project](https://cot-vla.github.io/) |
| TinyVLA | Compact, fast, and data-efficient VLA; Requires no pre-training; Uses small VLM + diffusion policy decoder | MobileVLM V2 / Moondream2 + Diffusion Policy Decoder | Diffusion Policy | [arXiv 2024](https://arxiv.org/abs/2409.12514) | [Project](https://tiny-vla.github.io/) |
| CogACT | Componentized VLA architecture; Specialized action module (Diffusion Action Transformer) conditioned on VLM output; Significantly outperforms OpenVLA / RT-2-X | InternVL-Chat-V1.5 (VLM) + Diffusion Action Transformer | Diffusion Policy | [arXiv 2024](https://arxiv.org/abs/2411.19650) | [Project](https://cogact.github.io/) |

## By Application Area

### Manipulation

Focuses on tasks involving interaction with objects, ranging from simple pick-and-place to complex, dexterous, long-horizon activities. This is a major application area for VLA research.

* [**RT-1 (Robotics Transformer 1)**](https://arxiv.org/abs/2212.06817) - Brohan, A., et al. (Google) [[Code](https://github.com/google-research/robotics_transformer)]

    * Early influential Transformer-based model demonstrating scalability on multi-task real-world data (13 robots, 130k trajectories). Uses discretized action tokens input to a Transformer. Shows improved generalization and robustness. 

* [**RT-2 (Robotics Transformer 2)**](https://arxiv.org/abs/2307.15818) - Brohan, A., et al. (Google DeepMind) [[Project](https://robotics-transformer2.github.io/)]

    * A landmark VLA model demonstrating the ability to transfer knowledge from web-scale VLMs to robotics by joint fine-tuning a VLM (PaLI-X, PaLM-E) to output action tokens. Shows emergent generalization to new objects, commands, and basic reasoning.  Defines the modern VLA concept.

* **PaLM-E** ([ICML 2023 / arXiv 2023](https://arxiv.org/abs/2303.03378))  - Driess, D., et al. (Google) [[Project](https://palm-e.github.io/)]

    * Embodied multimodal language model, injecting continuous sensor data (image, state) into a pre-trained LLM (PaLM). Validated on sequential manipulation, VQA, and captioning, showing positive transfer from vision-language data to robotic tasks. 

* **OpenVLA** ([arXiv 2024](https://arxiv.org/abs/2406.09246))  - Kim, M. J., et al, (Stanford, Berkeley, TRI, Google, Physical Intelligence, MIT), [[Code](https://github.com/openvla/openvla)]

    * State-of-the-art 7B open-source VLA at the time of release, based on Llama 2, DINOv2, SigLIP. Trained on 970k Open X-Embodiment trajectories. Outperforms RT-2-X with fewer parameters. Shows strong generalization and effective fine-tuning (PEFT) ability. 

* **Helix** （[Figure AI blog post 2024](https://www.figure.ai/news/helix)） - Figure AI. [[Project](https://www.figure.ai/news/helix)]

    * General-purpose VLA for humanoid robot (Figure 01) control. Features include full-body (including hands) control, multi-robot collaboration, arbitrary object grasping, all behaviors using a single network, and onboard deployment. Uses a hierarchical "System 1 (fast visuomotor) / System 2 (slow VLM reasoning)" architecture. 

* **π0 (Pi-Zero)** (Physical Intelligence blog post / arXiv 2024) - Physical Intelligence Team. [[Project](https://www.physicalintelligence.company/blog/pi0)] / [[Code](https://github.com/Physical-Intelligence/openpi)] / [[HuggingFace](https://huggingface.co/physical-intelligence)]

    * General-purpose VLA using flow matching to generate continuous actions (50Hz). Trained on data from 7 platforms, 68 tasks. Demonstrates complex tasks like laundry folding and table clearing. 

* **Hi Robot** (arXiv 2025) - Physical Intelligence Team. [[Project](https://www.physicalintelligence.company/research/hirobot)]

    * Hierarchical system using π0 as "System 1" and a VLM as "System 2" for reasoning and task decomposition (via self-talk), improving handling of complex prompts. 

* **SayCan (Do As I Can, Not As I Say)** ((https://arxiv.org/abs/2204.01691)) - Ahn, M., et al, (Google). [[Project](https://say-can.github.io/)] / [[Code](https://github.com/google-research/google-research/tree/master/saycan)]

    * Pioneering work grounding LLM planning in robot affordances. Uses an LLM (PaLM) to score potential skills by instruction relevance and a value function to score executability.  Primarily a high-level planner.

* **VIMA (Visual Matching Agent)** ([ICML 2023 / arXiv 2022](https://arxiv.org/abs/2210.03094)) - Jiang, Y., et al, [[Project](https://vimalabs.github.io/)]

    * Transformer-based agent that processes multimodal prompts (text, images, video) for manipulation tasks. Introduces VIMA-Bench. 

* **Octo** ([arXiv 2024](https://arxiv.org/abs/2405.12213))  - Octo Model Team (UC Berkeley, Google, TRI, et al.), [[Project](https://octo-models.github.io/)] / [[Code](https://github.com/octo-models/octo)]

    * General-purpose robot model trained on Open X-Embodiment. Transformer architecture with flexible input/output conditioning. Often used as a strong baseline model. 

* **VoxPoser** ((https://arxiv.org/abs/2307.05973)) - Huang, W., et al, [[Project](https://voxposer.github.io/) ]/ [[Code](https://github.com/huangwl18/VoxPoser)]

    * Uses LLM/VLM to synthesize 3D value maps (affordances) in perceptual space for zero-shot manipulation.  Focuses on the motion planning aspect.

* **ReKep (Relational Keypoint Constraints)** ((https://arxiv.org/abs/2409.01652)) - Huang, W., et al, [[Project](https://rekep-robot.github.io/)] / [[Code](https://github.com/huangwl18/ReKep)]

    * Uses LVM (DINOv2, SAM2) + VLM (GPT-4o) for spatio-temporal reasoning via keypoint constraints for manipulation tasks.  Point-based action approach.

* **OK-Robot** ([arXiv 2024](https://arxiv.org/abs/2401.12202))  - Singh, N., et al, [[Project](https://ok-robot.github.io/) ]/ [[Code](https://github.com/ok-robot/ok-robot)].

    * Integrates open knowledge models (VLM, LLM) for mobile manipulation arm (Hello Robot) navigation, perception, and manipulation in home environments. 

* **CoT-VLA (Chain-of-Thought VLA)** ((https://arxiv.org/abs/2503.22020)) - Wu, J., et al, [[Project](https://cot-vla.github.io/)]

    * Incorporates explicit visual chain-of-thought reasoning by predicting future goal images before generating actions. Uses hybrid attention (causal for vision/text, full for actions). 

* **3D-VLA** ((https://arxiv.org/abs/2403.09631)) - Zhen, Z., et al, [[Code](https://github.com/UMass-Embodied-AGI/3D-VLA)]

    * Introduces 3D perception (point clouds) and generative world models into VLAs, connecting 3D perception, reasoning, and action. 

* **TinyVLA** ([arXiv 2024](https://arxiv.org/abs/2409.12514)) - Liu, H., et al.

    * [Project](https://tiny-vla.github.io/).

    * Focuses on faster inference speed and higher data efficiency, eliminating the pre-training stage. Uses a smaller VLM backbone + diffusion policy decoder. Outperforms OpenVLA in speed/data efficiency. 

* **CogACT** ([arXiv 2024](https://arxiv.org/abs/2411.19650))  - Li, Q., et al, [[Project](https://cogact.github.io/)]

    * Componentized VLA architecture with a specialized action module (Diffusion Action Transformer) conditioned on VLM output. Significantly outperforms OpenVLA and RT-2-X. 

* **DexVLA** [(arXiv 2025)](https://arxiv.org/abs/2502.05855) - Li, Z., et al, [[Project](https://dex-vla.github.io/)]

    * Improves VLA efficiency/generalization via a large (1B parameter) diffusion-based action expert and an embodied curriculum learning strategy. Focuses on dexterity across different embodiments (single-arm / dual-arm / dexterous hand). 

* **Shake-VLA** [(arXiv 2025)](https://arxiv.org/abs/2501.06919) - Abdelkader, H., et al.

    * VLA-based system for automated cocktail making with a dual-arm robot, integrating vision (YOLOv8, EasyOCR), speech-to-text, and LLM instruction generation.  Application-specific system.

* **VLA Model-Expert Collaboration** [(arXiv 2025)](https://arxiv.org/abs/2503.04163) - Xiang, T.-Y., et al, [[Project](https://aoqunjin.github.io/Expert-VLA/)]
    * Enables human experts to collaborate with VLA models by providing corrective actions via shared autonomy. Achieves bi-directional learning (VLA improves, humans also improve). 

### Navigation

Focuses on tasks where a robot moves through an environment based on visual input and language instructions. Includes Vision-Language Navigation (VLN) and applications for legged robots.

* **NaVILA** ((https://arxiv.org/abs/2412.04453)) - Chen, X., et al, [[Project](https://navila-bot.github.io/)]

    * Two-stage framework for legged robot VLN. High-level VLA outputs actions in mid-level language form (e.g., "move forward 75cm"), and a low-level vision-motor policy executes them. Decouples high-level reasoning from low-level control. 

* **QUAR-VLA / QUART** ([ECCV 2024 / arXiv 2023](https://arxiv.org/abs/2312.14457)) - Tang, J., et al, [[Project](https://sites.google.com/view/quar-vla)]

    * Paradigm and VLA model family (QUART) for quadruped robots, integrating vision and language for navigation, complex terrain traversal, and manipulation. Includes the QUARD dataset. 

* **NaviLLM** ([arXiv 2023](https://arxiv.org/abs/2312.02010))  - Shah, D., et al, [[Code](https://github.com/zd11024/NaviLLM)]

    * General navigation model using LLMs for planning and interpreting instructions in diverse environments.

* **NaVid** ([arXiv 2024](https://arxiv.org/abs/2402.15852)) - Chen, X., et al, [[Project](https://pku-epic.github.io/NaVid/)]

    * Focuses on next-step planning in navigation using VLMs. Earlier work related to NaVILA.

### Human-Robot Interaction (HRI)

Focuses on enabling more natural and effective interactions between humans and robots, often using language (text or speech) as the primary interface.

* **VLAS (Vision-Language-Action-Speech)** ((https://arxiv.org/abs/2502.13508)) - Zhao, W., et al.

    * First end-to-end VLA with direct integration of speech commands, without needing an external ASR. Built upon LLaVA. Includes the SQA and CSI datasets. Uses Voice RAG to handle personalized tasks.

* **Shake-VLA** (arXiv 2025) - Abdelkader, H., et al.

    * Integrates voice commands for a dual-arm cocktail-making robot.

* **VLA Model-Expert Collaboration** (arXiv 2025) - Xiang, T.-Y., et al.

    * Enables human-robot collaboration through shared autonomy, improving both VLA and human performance.

* **Helix** (Figure AI blog post 2024) - Figure AI, [[Project](https://www.figure.ai/news/helix)]

    * Uses a single VLA model to enable multiple robots to collaborate on shared tasks (e.g., tidying up groceries).

### Task Planning / Reasoning

Focuses on using VLA/LLM components for high-level task decomposition, planning, and reasoning, often bridging the gap between complex instructions and low-level actions.

* **SayCan (Do As I Can, Not As I Say)** ((https://arxiv.org/abs/2204.01691)) - Ahn, M., et al,(Google), [[Project](https://say-can.github.io/)] / [[Code](https://github.com/google-research/google-research/tree/master/saycan)]

    * Grounds LLM planning in robot affordances.

* **PaLM-E** (ICML 2023 / arXiv 2023) - Driess, D., et al, (Google), [[Project](https://palm-e.github.io/)]

    * Can perform sequential manipulation planning end-to-end or output language subgoals. Shows visual chain-of-thought reasoning abilities.

* **EmbodiedGPT** (arXiv 2023) - Mu, Y., et al, [[Code](https://github.com/OpenGVLab/EmbodiedGPT)]

    * Multimodal model that performs end-to-end planning and reasoning for embodied tasks.

* **CoT-VLA (Chain-of-Thought VLA)** ((https://arxiv.org/abs/2503.22020)) - Wu, J., et al, [[Project](https://cot-vla.github.io/)]

    * Explicitly incorporates visual CoT reasoning by predicting future goal images.

* **Hi Robot** (arXiv 2025) - Physical Intelligence Team, [[Project](https://www.physicalintelligence.company/research/hirobot)]

    * Hierarchical VLA where a high-level VLM reasons and decomposes tasks for a low-level VLA (π0) executor.

* **LLM-Planner** ((https://arxiv.org/pdf/2212.04088)) - Liu, B., et al, [[Project](https://dki-lab.github.io/LLM-Planner/)]
    * Modular planner using LLMs.

* **Code as Policies (CaP)** ((https://arxiv.org/abs/2209.07753)) - Liang, J., et al. (Google), [[Project](https://code-as-policies.github.io/)]

    * Uses LLMs to directly generate robot policy code.


* **Inner Monologue** ((https://arxiv.org/abs/2207.05608)) - Huang, W., et al, [[Project](https://inner-monologue.github.io/)]

    * Uses language feedback from VLM/LLMs to guide robot policies.

* **The Critical Role of Hierarchical Reasoning for Complexity:** Many successful approaches to complex, long-horizon tasks employ hierarchical structures. This can be explicit (e.g., NaVILA's VLA + motor policy; Helix's System 1/2; Hi Robot's VLM planner + VLA executor), or implicit (e.g., SayCan grounding LLM plans in affordances; CoT-VLA generating intermediate visual goals). This suggests that monolithic end-to-end VLAs may struggle with deep reasoning or long-term planning compared to approaches that leverage emergent abilities of a single large model. Architectures that separate high-level planning/reasoning from low-level reactive control appear more effective. This architectural trend reflects the inherent complexity of linking semantic understanding to robust physical execution over extended periods.

## By Technical Approach

### Model Architectures

Focuses on the core neural network architectures used in VLA models.

* **Transformer-based:** The dominant architecture, leveraging self-attention mechanisms to integrate vision, language, and action sequences. Used in RT-1, RT-2, Octo, OpenVLA, VIMA, QUART, etc.
* **Diffusion-based:** Primarily for the action generation component, utilizing the ability of diffusion models to model complex distributions. Often combined with a Transformer backbone. E.g., Diffusion Policy, Octo (can use diffusion head), 3D Diffuser Actor, SUDD, MDT, RDT-1B, DexVLA, DiVLA, TinyVLA, DTP, Hybrid VLA+Diffusion.
* **Hierarchical / Decoupled:** Architectures that separate high-level reasoning/planning (often VLM/LLM-based) from low-level control/execution (which may be a separate policy). E.g., Helix (System 1/2), NaVILA (VLA + Locomotion Policy), Hi Robot (VLM + π0), SayCan (LLM + Value Function).
* **State-Space Models (SSM):** Emerging architectures like Mamba are being explored for their efficiency. E.g., RoboMamba.
* **Mixture-of-Experts (MoE / MoLE):** Using sparsely activated expert modules for task adaptation or efficiency. E.g., MoRE (Mixture-of-Robotic-Experts using LoRA). Componentized architecture in CogACT. π0 uses an MoE-like structure.

* **Architectural Diversification for Capability and Efficiency:** While Transformers are foundational, their limitations in handling continuous actions, computational cost, and reasoning depth are driving researchers to explore alternative or hybrid architectures. Diffusion models excel at action generation, hierarchical systems improve reasoning/control separation, SSMs promise efficiency, and MoEs aim for adaptive specialization. This diversification indicates an active search for architectures better suited to the specific constraints and needs of robotics than those designed purely for vision-language tasks. This has led to the emergence of hybrid and specialized designs to address the unique challenges of real-time control, action modeling, efficiency, and complex reasoning in robotics.

### Action Representation & Generation

Focuses on how robot actions are represented (e.g., discrete tokens vs. continuous vectors) and how models generate them. This is a key area differentiating VLAs from VLMs.

* **Action Tokenization / Discretization:** Representing continuous actions (e.g., joint angles, end-effector pose) as discrete tokens, often via binning. Used in early/many Transformer-based VLAs like RT-1, RT-2 to fit the language modeling paradigm. May have limitations in precision and high-frequency control.
* **Continuous Action Regression:** Directly predicting continuous action vectors. Sometimes used in conjunction with other methods or implemented via specific heads. L1 regression is used in OpenVLA-OFT.
* **Diffusion Policies for Actions:** Modeling action generation as a denoising diffusion process. Good at capturing multi-modality and continuous spaces. E.g., Diffusion Policy, Octo (diffusion head), SUDD, MDT, RDT-1B, DexVLA, DiVLA, TinyVLA, DTP. Can be slow due to iterative sampling.
* **Flow Matching:** An alternative generative method for continuous actions, used in π0 for efficient, high-frequency (50Hz) trajectory generation.
* **Action Chunking:** Predicting multiple future actions in a single step, for efficiency and temporal consistency. Used in ACT, RoboAgent, π0, PD-VLA. Increases action dimensionality and inference time when using AR decoding.
* **Parallel Decoding:** Techniques to speed up autoregressive decoding of action chunks. E.g., PD-VLA.
* **Specialized Tokenizers:** Developing better ways to tokenize continuous action sequences. E.g., FAST (Frequency-domain Action Sequence Tokenization), designed for dexterous, high-frequency tasks.
* **Point-based Actions:** Using VLMs to predict keypoints or goal locations rather than full trajectories. E.g., PIVOT, RoboPoint, ReKep.
* **Mid-Level Language Actions:** Generating actions as natural language commands to be consumed by a lower-level policy. E.g., NaVILA.

* **Action Generation as a Core VLA Challenge:** The diversity and rapid evolution of action representation/generation techniques highlight its importance and difficulty. The limitations of simple tokenization are driving innovations like diffusion models, flow matching, specialized tokenizers, and parallel decoding to balance precision, efficiency, and compatibility with large sequence models. This focus indicates that effectively translating high-level understanding into low-level physical control may be *the* core challenge that VLAs must address to move beyond standard VLM capabilities. Success requires a shift from simple VLM adaptation to action modeling techniques specifically designed for robotics.

### Learning Paradigms

Focuses on how VLA models are trained and adapted.

* **Imitation Learning (IL) / Behavior Cloning (BC):** Dominant paradigm, training VLAs to mimic expert demonstrations (often from teleoperation). Used for RT-1, RT-2, OpenVLA pre-training, Octo, Diffusion Policy, etc. Heavily reliant on large-scale, diverse, high-quality datasets. Performance is often limited by the quality of the demonstrations.
* **Reinforcement Learning (RL):** Used to fine-tune VLAs or train components, allowing models to learn from interaction and potentially exceed demonstrator performance. Challenges include stability and sample efficiency with large models. E.g., iRe-VLA (iterative RL/SFT), MoRE (RL objective for MoE VLAs handling mixed data), RPD (RL-based policy distillation), ConRFT (RL fine-tuning with consistency policies), SafeVLA (Constrained RL for safety).
* **Pre-training & Fine-tuning:** Standard approach, involving pre-training on large datasets (web data for VLM backbones, large robot datasets like OpenX for VLAs) and then fine-tuning on specific tasks or robots.
* **Parameter-Efficient Fine-Tuning (PEFT):** Techniques like LoRA to efficiently adapt large VLAs without retraining the entire model, crucial for practical deployment and customization. MoRE uses LoRA modules as experts.
* **Distillation:** Training smaller, faster models (students) to mimic the behavior of larger, slower models (teachers). E.g., RPD (distilling a VLA to an RL policy), OneDP (distilling a diffusion policy).
* **Curriculum Learning:** Structuring the learning process, e.g., by embodiment complexity. E.g., DexVLA uses embodied curriculum.
* **Learning from Mixed-Quality Data:** Using techniques (e.g., RL in MoRE) to learn effectively even when demonstration data is suboptimal or contains failures.

* **Bridging Imitation and Interaction:** While Imitation Learning (IL) on large-scale datasets like OpenX is foundational for creating general-purpose VLAs, there's a growing trend towards combining it with interactive learning (RL) for fine-tuning and improvement. This hybrid approach aims to leverage the broad knowledge of IL datasets while overcoming IL's limitations (suboptimality, dataset cost) by enabling robots to adapt and potentially surpass human demonstrators through environmental interaction. The challenge lies in making RL stable and efficient for large VLA models. This trend indicates a direction towards combining the strengths of both: broad generalization via large-scale IL pre-training, and targeted refinement and adaptation via efficient and stable RL fine-tuning.

### Input Modalities & Grounding

Focuses on input data types beyond standard RGB images and text used by VLAs, and how they ground these inputs.

* **Integrating Speech:** Control via spoken commands, potentially capturing nuances missed by text. Requires handling the speech modality directly or via ASR. E.g., VLAS (direct integration), Shake-VLA (uses external STT/TTS).
* **Integrating 3D Vision:** Using point clouds, voxels, depth maps, or implicit representations (NeRFs, 3DGS) to provide richer spatial understanding. E.g., 3D-VLA, PerAct, Act3D, RVT, RVT-2, RoboUniView, DP3, 3D Diffuser Actor, LEO, 3D-LLM, LLM-Grounder, SpatialVLA.
* **Integrating Proprioception / State:** Incorporating the robot's own state (joint angles, velocities, end-effector pose) as input. Common in many policies, explicitly mentioned in VLAS, PaLM-E, π0 (evaluation requires Simpler fork with proprioception support). OpenVLA initially lacked this, noted as a limitation/future work.
* **Multimodal Prompts:** Handling instructions that include images or video in addition to text. E.g., VIMA.
* **Grounding:** The process of linking language descriptions or visual perceptions to specific entities, locations, or actions in the physical world or robot representation. Addressed via various techniques like similarity matching, leveraging common-sense knowledge, multimodal alignment, or interaction. LLM-Grounder focuses on open-vocabulary 3D visual grounding.

* **The Need for Richer World Representations:** The increasing integration of 3D vision and speech indicates that standard RGB images and text may be insufficient for robust, nuanced robot interaction in complex environments. 3D data provides crucial spatial context missing from 2D images, while speech offers a more natural HRI modality. This trend suggests that future VLAs will become truly "multi-sensory" agents, moving beyond just vision and language. The VLA paradigm is expanding beyond its name ("Vision-Language-Action") to incorporate the richer sensory inputs (3D, speech, proprioception) needed for effective manipulation and interaction in the complex physical world, moving towards more general-purpose multimodal embodied agents.

## Datasets and Benchmarks

This section lists key resources for training and evaluating VLA models. Large-scale, diverse datasets and standardized benchmarks are crucial for progress in the field.

### **Quick Glance at Datasets and Benchmarks**

| Name | Type | Focus Area | Key Features / Environment | Link | Key Publication |
|---|---|---|---|---|---|
| Open X-Embodiment (OpenX) | Dataset | General Manipulation | Aggregates 20+ datasets, cross-embodiment/task/environment, >1M trajectories | [Project](https://robotics-transformer-x.github.io/) | [arXiv 2023](https://arxiv.org/abs/2310.08864) |
| DROID | Dataset | Real-world Manipulation | Large-scale human-collected data (500+ tasks, 26k hours) | [Project](https://droid-dataset.github.io/) | (https://arxiv.org/abs/2403.06037) |
| CALVIN | Dataset / Benchmark | Long-Horizon Manipulation | Long-horizon tasks with language conditioning, Franka arm, PyBullet simulation | [Project](http://calvin.cs.uni-freiburg.de/)| (https://arxiv.org/abs/2112.03227) |
| QUARD | Dataset | Quadruped Robot Tasks | Large-scale multi-task dataset (sim + real) for navigation and manipulation | [Project](https://sites.google.com/view/quar-vla) | [ECCV 2024](https://arxiv.org/abs/2312.14457) |
| BEHAVIOR-1K | Dataset / Benchmark | Household Activities | 1000 simulated human household activities | [Project](https://behavior.stanford.edu/) | (https://arxiv.org/abs/2108.03332) |
| Isaac Sim / Orbit / OmniGibson | Simulator | High-fidelity Robot Simulation | NVIDIA Omniverse-based, physically realistic | (https://developer.nvidia.com/isaac-sim), [Orbit](https://isaac-orbit.github.io/), [OmniGibson](https://omnigibson.stanford.edu/) | - |
| Habitat Sim | Simulator | Embodied AI Navigation | Flexible, high-performance 3D simulator | [Project](https://aihabitat.org/) | (https://arxiv.org/abs/1904.01201) |
| MuJoCo | Simulator | Physics Engine | Popular physics engine for robotics and RL | [Website](https://mujoco.org/) | - |
| PyBullet | Simulator | Physics Engine | Open-source physics engine, used for CALVIN, etc. | [Website](https://pybullet.org/) | - |
| ManiSkill (1, 2, 3) | Benchmark | Generalizable Manipulation Skills | Large-scale manipulation benchmark based on SAPIEN | [Project](https://maniskill.ai/) | (https://arxiv.org/abs/2107.14483) |
| Meta-World | Benchmark | Multi-task / Meta RL Manipulation | 50 Sawyer arm manipulation tasks, MuJoCo | [Project](https://meta-world.github.io/)| (https://arxiv.org/abs/1910.10897) |
| RLBench | Benchmark | Robot Learning Manipulation | 100+ manipulation tasks, CoppeliaSim (V-REP) | [Project](https://sites.google.com/view/rlbench) | (https://arxiv.org/abs/1909.12271) |
| VLN-CE / R2R / RxR | Benchmark | Vision-Language Nav | Standard VLN benchmarks, often run in Habitat | [VLN-CE](https://github.com/jacobkrantz/VLN-CE), https://github.com/airsplay/R2R-EnvDrop, https://github.com/google-research-datasets/RxR) | Various |

### Robot Learning Datasets

Large-scale datasets of robot interaction trajectories, often with accompanying language instructions and visual observations. Crucial for training general-purpose policies via imitation learning.

* **Open X-Embodiment (OpenX)** ([Project](https://robotics-transformer-x.github.io/)) - Open X-Embodiment Collaboration.

    * A massive, standardized dataset aggregating data from 20+ existing robot datasets, spanning diverse embodiments, tasks, and environments.

    * Used to train major VLAs like RT-X, Octo, OpenVLA, π0.

    * Contains over 1 million trajectories. 

* RT-X: A family of models released along with the Open X-Embodiment dataset. 
    
* **BridgeData V2** ([Project](https://rail-berkeley.github.io/bridgedata/)) - Walke, H., et al.

    * Large dataset collected on a WidowX robot, used for OpenVLA evaluation. 

* **DROID** ([Project](https://droid-dataset.github.io/))  - Manuelli, L., et al.

    * Large-scale, diverse, human-collected manipulation dataset (500+ tasks, 26k hours).

    * Used to fine-tune/evaluate OpenVLA, π0. 

* **RH20T** ([Project](https://rh20t.github.io/)) - Shao, L., et al.

    * Comprehensive dataset with 110k robot clips, 110k human demonstrations, and 140+ tasks. 

* **CALVIN (Composing Actions from Language and Vision)** ([Project](http://calvin.cs.uni-freiburg.de/)) - Mees, O., et al.

    * (Uni Freiburg).

    * Benchmark and dataset for long-horizon language-conditioned manipulation with a simulated Franka arm in PyBullet. 

* **QUARD (QUAdruped Robot Dataset)** ([Project](https://sites.google.com/view/quar-vla))  - Tang, J., et al.

    * Large-scale multi-task dataset (sim + real) for quadruped navigation and manipulation, released with QUAR-VLA.

    * Contains 348k sim + 3k real clips or 246k sim + 3k real clips. 

* **RoboNet** ([Project](https://www.robonet.wiki/))  - Dasari, S., et al.

    * Early large-scale dataset aggregating data from multiple robot platforms. 

* **BEHAVIOR-1K** ([Project](https://behavior.stanford.edu/))  - Srivastava, S., et al.

    * Dataset of 1000 simulated human household activities, useful for high-level task understanding. 

* **SQA & CSI Datasets** ([arXiv 2025](https://arxiv.org/abs/2502.13508))- Zhao, W., et al.

    * Curated datasets with speech instructions, released with the VLAS model, for speech-vision-action alignment and fine-tuning. 

* **Libero** ([Project](https://libero-project.github.io/datasets))  - Li, Z., et al.

    * Benchmark suite for robot lifelong learning with procedurally generated tasks.

    * Used in π0 fine-tuning examples. 

* **D4RL (Datasets for Deep Data-Driven Reinforcement Learning)** ([Code](https://github.com/Farama-Foundation/D4RL))  - Fu, J., et al.

    * Standardized datasets for offline RL research, potentially useful for RL-based VLA methods. 

### Simulation Environments

Physics-based simulators used to train agents, generate synthetic data, and evaluate policies in controlled settings before real-world deployment.

* **NVIDIA Isaac Sim / Orbit / OmniGibson** ((https://developer.nvidia.com/isaac-sim), Orbit, OmniGibson).

    * High-fidelity, physically realistic simulators based on NVIDIA Omniverse.

    * Used for QUAR-VLA, ReKep, ARNOLD, etc. 

* **Habitat Sim** ([Project](https://aihabitat.org/)) - Facebook AI Research (Meta AI).

    * Flexible, high-performance 3D simulator for Embodied AI research, especially navigation. 

* **MuJoCo (Multi-Joint dynamics with Contact)** ([Project](https://mujoco.org/)).

    * Popular physics engine widely used for robot simulation and RL benchmarks (dm\_control, robosuite, Meta-World, RoboHive). 

* **PyBullet** ([Project](https://pybullet.org/)).

    * Open-source physics engine, used for CALVIN and other benchmarks (panda-gym). 

* **SAPIEN** ([Project](https://sapien.ucsd.edu/)).

    * Physics simulator focused on articulated objects and interaction.

    * Used for the ManiSkill benchmark. 

* **Gazebo** ([Project](https://gazebosim.org/)).

    * Widely used open-source robot simulator, especially in the ROS ecosystem. 

* **Webots** ([Project](https://cyberbotics.com/)).

    * Open-source desktop robot simulator. 

* **Genesis** ([GitHub](https://github.com/Genesis-Embodied-AI/Genesis)).

    * A newer platform aimed at general robot/Embodied AI simulation. 

* **UniSim** ([arXiv 2023](https://universal-simulator.github.io/unisim/)) - Yang, G., et al.

    * Learns interactive simulators from real-world videos. 

### Evaluation Benchmarks

Standardized suites of environments and tasks used to evaluate and compare the performance of VLA models and other robot learning algorithms.

* **CALVIN** ([Project](https://github.com/mees/calvin)).

    * Benchmark for long-horizon language-conditioned manipulation. 

* **ManiSkill (1, 2, 3)** ([Project](https://maniskill.ai/)).

    * Large-scale benchmark for generalizable manipulation skills, based on SAPIEN. 

* **Meta-World** ([Project](https://meta-world.github.io/)).

    * Multi-task and meta-RL benchmark with 50 different manipulation tasks using a Sawyer arm in MuJoCo. 

* **RLBench** ([Project](https://sites.google.com/view/rlbench)).

    * Large-scale benchmark with 100+ manipulation tasks in CoppeliaSim (V-REP). 

* **Franka Kitchen** ([GitHub](https://robotics.farama.org/envs/franka_kitchen/franka_kitchen/)).

    * dm\_control-based benchmark involving kitchen tasks with a Franka arm.

    * Used in iRe-VLA. 

* **LIBERO** ([Project](https://libero-project.github.io/datasets)).

    * Benchmark for lifelong/continual learning in robot manipulation. 

* **VIMA-Bench** ([Project](https://vimalabs.github.io/)).

    * Multimodal few-shot prompting benchmark for robot manipulation. 

* **BEHAVIOR-1K** ([Project](https://behavior.stanford.edu/)).

    * Benchmark focused on long-horizon household activities. 

* **VLN-CE / R2R / RxR** ([VLN-CE](https://github.com/jacobkrantz/VLN-CE),[https://github.com/airsplay/R2R-EnvDrop](https://github.com/airsplay/R2R-EnvDrop),[https://github.com/google-research-datasets/RxR](https://github.com/google-research-datasets/RxR)).

    * Standard benchmarks for Vision-Language Navigation, often run in Habitat.

    * NaVILA is evaluated on these. 

* **Safety-CHORES** ((https://arxiv.org/abs/2503.03480)).

    * A new simulated benchmark with safety constraints, proposed for evaluating safe VLA learning. 

* **OK-VQA** ([Project](https://okvqa.allenai.org/)).

    * Visual question answering benchmark requiring external knowledge, used to evaluate the general VLM abilities of PaLM-E. 

* **Symbiotic Relationship of Models, Data, and Simulation:** VLA progress is tightly coupled with the availability of large-scale datasets (OpenX being crucial) and powerful simulators (Isaac Sim, MuJoCo enabling large-scale training). Benchmarks (CALVIN, ManiSkill) drive standardized evaluation. However, the cost of real-world data collection and the persistent sim-to-real gap remain major bottlenecks, driving research into data augmentation, sim-to-real techniques, data-efficient learning, and automated data collection. The ecosystem of models, datasets, simulators, and benchmarks co-evolves, with limitations in one area (e.g., real data cost) driving innovation in others (e.g., simulation, data efficiency). Overcoming data/simulation limitations is key to unlocking the potential of VLAs.

## Challenges and Future Directions

Summarizes key limitations of current VLA models and highlights promising directions for future research, based on analysis of surveys and recent papers.

* **Data Efficiency & Scalability:** Reducing reliance on massive, expensive, expert-driven datasets. Improving the ability to learn from limited, mixed-quality, or internet-sourced data. Efficiently scaling models and training processes. 

    * Future directions: Improved sample efficiency (RL, self-supervision), sim-to-real transfer, automated data generation, efficient architectures (SSMs, MoEs), data filtering/weighting.
* **Inference Speed & Real-Time Control:** Current large VLAs may be too slow for the high-frequency control loops needed for dynamic tasks or dexterous manipulation. 

    * Future directions: Smaller/compact models (TinyVLA), efficient architectures (RoboMamba), parallel decoding (PD-VLA), action chunking optimization (FAST), model distillation (OneDP, RPD), hardware acceleration.
* **Robustness & Reliability:** Ensuring consistent performance across variations in environment, lighting, object appearance, disturbances, and unexpected events. Current models can be brittle. 

    * Future directions: Adversarial training, improved grounding, better 3D understanding, closed-loop feedback, anomaly detection, incorporating physical priors, testing frameworks (VLATest).
* **Generalization:** Improving the ability to generalize to new tasks, objects, instructions, environments, and embodiments beyond the training distribution. This is a core promise of VLAs, but remains a challenge. 

    * Future directions: Training on more diverse data (OpenX), effective utilization of VLM pre-training knowledge, compositional reasoning, continual/lifelong learning, better action representations.
* **Safety & Alignment:** Explicitly incorporating safety constraints to prevent harm to the robot, the environment, or humans. Ensuring alignment with user intent. Crucial for real-world deployment. 

    * Future directions: Constrained reinforcement learning (SafeVLA), formal verification, human oversight mechanisms, robust failure detection/recovery, ethical considerations.
* **Dexterity & Contact-Rich Tasks:** Improving performance on tasks requiring fine motor skills, precise force control, and handling complex object interactions. Current VLAs often lag behind specialized methods in this area. 

    * Future directions: Better action representations (FAST, Diffusion), integration of tactile sensing, improved physical understanding/simulation, hybrid control approaches.
* **Reasoning & Long-Horizon Planning:** Enhancing the ability for multi-step reasoning, long-horizon planning, and handling complex instructions. 

    * Future directions: Hierarchical architectures, explicit planning modules, chain-of-thought reasoning (visual/textual), memory mechanisms, world models.
* **Multimodality Expansion:** Integrating richer sensory inputs beyond vision + language, such as audio/speech, touch, force, 3D. 

    * Future directions: Developing architectures and alignment techniques for diverse modalities.
* **The Tension Between Generalization and Specialization/Performance:** While a core promise of VLAs is to leverage large pre-trained models for generalization, achieving high success rates on specific, complex, or novel tasks often requires significant fine-tuning or specialized components. This creates a tension: how to achieve expert-level performance while maintaining broad generalization? Future research needs to balance general capabilities with task-specific proficiency, potentially through more effective adaptation techniques (e.g., PEFT), modular architectures (e.g., MoEs), or methods that combine general priors with task-specific learning.

## Related Awesome Lists

* Awesome-VLA:

    * https://github.com/yueen-ma/Awesome-VLA
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
    * https://github.com/geoaigroup/awesome-vision-language-models-for-earth-observation

## Citation

If you find this repository useful, please consider citing this list:

```
@misc{kira2022llmroboticspaperslist,
    title = {Awesome-VAL-Robotics},
    author = {Jiaqi Liu},
    journal = {GitHub repository},
    url = {https://github.com/Jiaaqiliu/Awesome-VLA-Robotics},
    year = {2025},
}