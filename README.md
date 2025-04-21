# Awesome VLA for Robotics

A comprehensive list of excellent research papers, models, datasets, and other resources on Vision-Language-Action (VLA) models in robotics. Contributions are welcome! 

## Table of Contents

- [What are VLA Models in Robotics?](#what-are-vla-models-in-robotics)
- [Surveys and Overviews](#surveys-and-overviews)
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

## Surveys and Overviews

This section lists key survey papers that provide broad overviews of the VLA field, related concepts (such as Embodied AI), and the role of foundation models in robotics. 

### **Overview of Key Survey Papers**
|                                      Title                                      |         Authors        |                 Focus Area                 |                                       Key Taxonomy/Contributions                                      |       Link       |     Venue/Year     |
|:-------------------------------------------------------------------------------:|:----------------------:|:------------------------------------------:|:-----------------------------------------------------------------------------------------------------:|:----------------:|:------------------:|
| A Survey on Vision-Language-Action Models for Embodied AI                       | Ma, Y., et al.         | VLA for Embodied AI                        | First VLA survey; detailed taxonomy (components, control strategies, task planners); resource summary | [arXiv:2405.14093](https://arxiv.org/abs/2405.14093) | arXiv 2024         |
| Aligning Cyber Space with Physical World: A Comprehensive Survey on Embodied AI | Liu, Y., et al.        | Embodied AI (General)                      | Broad Embodied AI background                                                                          | [arXiv:2407.06886](https://arxiv.org/abs/2407.06886) | arXiv 2024         |
| Robot learning in the era of foundation models: A survey                        | Yuan, W., et al.       | Foundation Models in Robot Learning        | Focuses on the application of foundation models in robot learning                                     | [arXiv:2311.14379](https://arxiv.org/abs/2311.14379) | arXiv 2023         |
| A Survey on Robotics with Foundation Models: toward Embodied AI                 | Xu, Z., et al.         | Foundation Models for Embodied AI          | Connects foundation models with Embodied AI                                                           | [arXiv:2402.02385](https://arxiv.org/abs/2402.02385) | arXiv 2024         |
| Toward general-purpose robots via foundation models: A survey and meta-analysis | Hu, Y., et al.         | General-Purpose Robots & Foundation Models | Focuses on general-purpose robots and foundation models                                               | [arXiv:2312.08782](https://arxiv.org/abs/2312.08782) | Machines 2023      |
| What Foundation Models can Bring for Robot Learning in Manipulation : A Survey  | Yao, L., et al.        | Foundation Models for Manipulation         | Focuses on manipulation tasks                                                                         | [arXiv:2404.18201](https://arxiv.org/abs/2404.18201) | arXiv 2024         |
| Towards Generalist Robot Learning from Internet Video: A Survey                 | McCarthy, R., et al.   | Learning from Internet Video               | Focuses on learning from video data                                                                   | [arXiv:2404.19664](https://arxiv.org/abs/2404.19664) | arXiv 2024         |
| Large Multimodal Agents: A Survey                                               | Xie, J., et al.        | Multimodal Agents                          | Broader background of multimodal agents                                                               | [arXiv:2402.15116](https://arxiv.org/abs/2402.15116) | arXiv 2024         |
| A Survey on Large Language Models for Automated Planning                        | Aghzal, M., et al.     | LLMs for Automated Planning                | Focuses on the application of LLMs in planning, related to VLA task planners                          | [arXiv:2502.12435](https://arxiv.org/abs/2502.12435) | arXiv 2025         |
| Integrating Large Language Models in Robotics: A Survey                         | Kim, D., et al.        | LLM Integration in Robotics                | Focuses on the integration of LLMs in various components of robotics                                  | [arXiv:2404.09228](https://arxiv.org/abs/2404.09228) | arXiv 2024         |
| Vision-Language Models for Vision Tasks: A Survey                               | Zhang, J., et al.      | VLMs for Vision Tasks                      | Background knowledge of VLMs, the foundation of many VLAs                                             | [arXiv:2304.00685](https://arxiv.org/abs/2304.00685) | TPAMI 2024         |
| Survey on Vision-Language-Action Models                                         | Adilkhanov, A., et al. | VLA Models (AI-Generated)                  | (Note: AI-generated demonstration survey, use with caution)                                           | [arXiv:2502.06851](https://arxiv.org/abs/2502.06851) | arXiv 2025         |
| From Screens to Scenes: A Survey of Embodied AI in Healthcare                   | Liu, Y., et al.        | Embodied AI in Healthcare                  | Survey for a specific application area                                                                | [arXiv:2501.07468](https://arxiv.org/abs/2501.07468) | Information Fusion |
| Towards Robust and Secure Embodied AI: A Survey on Vulnerabilities and Attacks  | Xing, W., et al.       | Security & Robustness in Embodied AI       | Focuses on the security and robustness challenges of Embodied AI                                      | [arXiv:2502.13175](https://arxiv.org/abs/2502.13175) | arXiv 2025         |

## Key VLA Models and Research Papers

This section is the heart of the resource, listing specific VLA models and influential research papers. Papers are first categorized by major application area, then by key technical contributions. A paper/model may appear in multiple subsections if it is relevant to several categories.

### **Quick Glance at Key VLA Models**

| Model Name | Key Contribution / Features | Base VLM / Architecture | Action Generation Method | Key Publication(s) | Project / Code |
|---|---|---|---|---|---|
| RT-1 | First large-scale Transformer robot model; Demonstrates scalability on multi-task real-world data; Action discretization | Transformer (EfficientNet-B3 vision) | Action binning + Token output | https://arxiv.org/abs/2212.06817 | [Project](https://robotics-transformer1.github.io/) / [Code](https://github.com/google-research/robotics_transformer) |
| RT-2 | Transfers web knowledge of VLMs to robot control; Joint fine-tuning of VLM to output action tokens; Shows emergent generalization | PaLI-X / PaLM-E (Transformer) | Action binning + Token output | https://arxiv.org/abs/2307.15818 | [Project](https://robotics-transformer2.github.io/) |
| PaLM-E | Embodied multimodal language model; Injects continuous sensor data (image, state) into pre-trained LLM; Usable for sequential manipulation planning, VQA, etc. | PaLM (Transformer) | Outputs text (subgoals or action descriptions), needs low-level policy to execute | [ICML 2023](https://openreview.net/pdf?id=VTpHpqM3Cf) | [Project](https://palm-e.github.io/)|
| OpenVLA | Open-source 7B parameter VLA; Based on Llama 2; Trained on OpenX dataset; Outperforms RT-2-X; Shows good generalization and PEFT ability | Llama 2 (DINOv2 + SigLIP vision) | Action binning + Token output (raw) / L1 regression (OFT) | [Projects](https://openvla.github.io/) | [Project](https://openvla.github.io/) / [Code](https://github.com/openvla/openvla) / [HF](https://huggingface.co/collections/openvla/openvla-666b11f9e9f77a2f02a6c740)|
| Helix | General-purpose VLA for humanoid robots; Hierarchical architecture (System 1/2); Full-body control; Multi-robot collaboration; Onboard deployment | Custom VLM (System 2) + Visuomotor Policy (System 1) | Continuous action output (System 1) | https://www.figure.ai/news/helix | [Project](https://www.figure.ai/news/helix) |
| π0 (Pi-Zero) | General-purpose VLA; Uses Flow Matching to generate continuous action trajectories (50Hz); Cross-platform training (7 platforms, 68 tasks) | PaliGemma (Transformer) + Action Expert | Flow Matching | [arXiv 2024](https://arxiv.org/abs/2410.08532) | [Project](https://www.physicalintelligence.company/research/pi-zero) / [Code](https://github.com/Physical-Intelligence/openpi) / [HF](https://huggingface.co/physical-intelligence) |
| Octo | General-purpose robot model; Trained on OpenX dataset; Flexible input/output conditioning; Often used as a baseline | Transformer (ViT) | Action binning + Token output / Diffusion Head | [Project Website 2023](https://octo-models.github.io/)| [Code](https://github.com/octo-models/octo) |
| SayCan | Grounds LLM planning in robot affordances; Uses LLM to score skill relevance + value function to score executability | PaLM (Transformer) + Value Function | Selects pre-defined skills (high-level planner) | (https://arxiv.org/abs/2204.01691) | [Project](https://say-can.github.io/) / [Code](https://github.com/google-research/google-research/tree/master/saycan) |
| NaVILA | Two-stage framework for legged robot VLN; High-level VLA outputs mid-level language actions, low-level vision-motor policy executes | InternVL-Chat-V1.5 (VLM) + Locomotion Policy (RL) | Mid-level language action output (VLA) | https://arxiv.org/abs/2412.04453 | [Project](https://navila-bot.github.io/) |
| VLAS | First end-to-end VLA with direct integration of speech commands; Based on LLaVA; Three-stage fine-tuning for voice commands; Supports personalized tasks (Voice RAG) | LLaVA (Transformer) + Speech Encoder | Action binning + Token output | https://arxiv.org/abs/2502.13508 | - |
| CoT-VLA | Incorporates explicit Visual Chain-of-Thought (Visual CoT); Predicts future goal images before generating actions; Hybrid attention mechanism | Llama 2 (ViT vision) | Action binning + Token output (after predicting visual goals) | (https://arxiv.org/abs/2503.22020) | [Project](https://visualcot.github.io/) |
| TinyVLA | Compact, fast, and data-efficient VLA; Requires no pre-training; Uses small VLM + diffusion policy decoder | MobileVLM V2 / Moondream2 + Diffusion Policy Decoder | Diffusion Policy | [arXiv 2024](https://arxiv.org/abs/2409.12514) | [Project](https://tiny-vla.github.io/) |
| CogACT | Componentized VLA architecture; Specialized action module (Diffusion Action Transformer) conditioned on VLM output; Significantly outperforms OpenVLA / RT-2-X | InternVL-Chat-V1.5 (VLM) + Diffusion Action Transformer | Diffusion Policy | [arXiv 2024](https://arxiv.org/abs/2411.19650) | [Project](https://cogact-vla.github.io/) |

## By Application Area

### Manipulation

Focuses on tasks involving interaction with objects, ranging from simple pick-and-place to complex, dexterous, long-horizon activities. This is a major application area for VLA research.

* [**RT-1 (Robotics Transformer 1)**](https://arxiv.org/abs/2212.06817) - Brohan, A., et al. (Google).

    * [Code](https://github.com/google-research/robotics_transformer).

    * Early influential Transformer-based model demonstrating scalability on multi-task real-world data (13 robots, 130k trajectories). Uses discretized action tokens input to a Transformer. Shows improved generalization and robustness. 

* [**RT-2 (Robotics Transformer 2)**](https://arxiv.org/abs/2307.15818) - Brohan, A., et al.

    * (Google DeepMind).

    * [Project](https://robotics-transformer2.github.io/).

    * A landmark VLA model demonstrating the ability to transfer knowledge from web-scale VLMs to robotics by joint fine-tuning a VLM (PaLI-X, PaLM-E) to output action tokens. Shows emergent generalization to new objects, commands, and basic reasoning.  Defines the modern VLA concept.

* **PaLM-E** ([ICML 2023 / arXiv 2023](https://arxiv.org/abs/2303.03378))  - Driess, D., et al.

    * (Google).

    * [Project](https://palm-e.github.io/).

    * Embodied multimodal language model, injecting continuous sensor data (image, state) into a pre-trained LLM (PaLM). Validated on sequential manipulation, VQA, and captioning, showing positive transfer from vision-language data to robotic tasks. 

* **OpenVLA** ([arXiv 2024](https://arxiv.org/abs/2406.09246))  - Kim, M. J., et al.

    * (Stanford, Berkeley, TRI, Google, Physical Intelligence, MIT).

    * [Code](https://github.com/mlresearch/openvla).

    * State-of-the-art 7B open-source VLA at the time of release, based on Llama 2, DINOv2, SigLIP. Trained on 970k Open X-Embodiment trajectories. Outperforms RT-2-X with fewer parameters. Shows strong generalization and effective fine-tuning (PEFT) ability. 

* **Helix** （[Figure AI blog post 2024](https://www.figure.ai/news/helix)） - Figure AI.

    * [Project](https://www.figure.ai/news/helix).

    * General-purpose VLA for humanoid robot (Figure 01) control. Features include full-body (including hands) control, multi-robot collaboration, arbitrary object grasping, all behaviors using a single network, and onboard deployment. Uses a hierarchical "System 1 (fast visuomotor) / System 2 (slow VLM reasoning)" architecture. 

* **π0 (Pi-Zero)** (Physical Intelligence blog post / arXiv 2024) - Physical Intelligence Team.

    * [Project](https://www.physicalintelligence.company/research/pi-zero) / [Code](https://github.com/Physical-Intelligence/openpi) / [HuggingFace](https://huggingface.co/physical-intelligence)

    * General-purpose VLA using flow matching to generate continuous actions (50Hz). Trained on data from 7 platforms, 68 tasks. Demonstrates complex tasks like laundry folding and table clearing. 

* **Hi Robot** (arXiv 2025) - Physical Intelligence Team.

    * [Project](https://www.physicalintelligence.company/research/hirobot).

    * Hierarchical system using π0 as "System 1" and a VLM as "System 2" for reasoning and task decomposition (via self-talk), improving handling of complex prompts. 

* **SayCan (Do As I Can, Not As I Say)** ((https://arxiv.org/abs/2204.01691)) - Ahn, M., et al.

    * (Google).

    * [Project](https://say-can.github.io/) / [Code](https://github.com/google-research/google-research/tree/master/saycan)

    * Pioneering work grounding LLM planning in robot affordances. Uses an LLM (PaLM) to score potential skills by instruction relevance and a value function to score executability.  Primarily a high-level planner.

* **VIMA (Visual Matching Agent)** (ICML 2023 / arXiv 2022) - Jiang, Y., et al.

    * [Project](https://vima.cs.princeton.edu/).

    * Transformer-based agent that processes multimodal prompts (text, images, video) for manipulation tasks. Introduces VIMA-Bench. 

* **Octo** (Project website 2023) - Octo Model Team (UC Berkeley, Google, TRI, et al.).

    * [Code](https://github.com/google-research/octo).

    * General-purpose robot model trained on Open X-Embodiment. Transformer architecture with flexible input/output conditioning. Often used as a strong baseline model. 

* **VoxPoser** ((https://arxiv.org/abs/2307.05973)) - Huang, W., et al.

    * [Project](https://voxposer.github.io/) / [Code](https://github.com/huangwl18/VoxPoser)

    * Uses LLM/VLM to synthesize 3D value maps (affordances) in perceptual space for zero-shot manipulation.  Focuses on the motion planning aspect.

* **ReKep (Relational Keypoint Constraints)** ((https://arxiv.org/abs/2409.01652)) - Huang, W., et al.

    * [Project](https://rekep-robot.github.io/) / [Code](https://github.com/huangwl18/ReKep).

    * Uses LVM (DINOv2, SAM2) + VLM (GPT-4o) for spatio-temporal reasoning via keypoint constraints for manipulation tasks.  Point-based action approach.

* **OK-Robot** (arXiv 2024) - Singh, N., et al.

    * [Project](https://ok-robot.github.io/) / [Code](https://github.com/ok-robot/ok-robot).

    * Integrates open knowledge models (VLM, LLM) for mobile manipulation arm (Hello Robot) navigation, perception, and manipulation in home environments. 

* **CoT-VLA (Chain-of-Thought VLA)** ((https://arxiv.org/abs/2503.22020)) - Wu, J., et al.

    * [Project](https://cot-vla.github.io/).

    * Incorporates explicit visual chain-of-thought reasoning by predicting future goal images before generating actions. Uses hybrid attention (causal for vision/text, full for actions). 

* **3D-VLA** ((https://arxiv.org/abs/2403.09599)) - Zhen, Z., et al.

    * [[Project](https://3d-vla.github.io/) / [Code](https://github.com/zhen-zx/3D-VLA).

    * Introduces 3D perception (point clouds) and generative world models into VLAs, connecting 3D perception, reasoning, and action. 

* **TinyVLA** (arXiv 2024) - Liu, H., et al.

    * [Project](https://tiny-vla.github.io/).

    * Focuses on faster inference speed and higher data efficiency, eliminating the pre-training stage. Uses a smaller VLM backbone + diffusion policy decoder. Outperforms OpenVLA in speed/data efficiency. 

* **CogACT** (arXiv 2024) - Li, Q., et al.

    * [Project](https://cog-act.github.io/).

    * Componentized VLA architecture with a specialized action module (Diffusion Action Transformer) conditioned on VLM output. Significantly outperforms OpenVLA and RT-2-X. 

* **DexVLA** (arXiv 2025) - Li, Z., et al.

    * [Project](https://diffusion-vla.github.io/).

    * Improves VLA efficiency/generalization via a large (1B parameter) diffusion-based action expert and an embodied curriculum learning strategy. Focuses on dexterity across different embodiments (single-arm / dual-arm / dexterous hand). 

* **Shake-VLA** (arXiv 2025) - Abdelkader, H., et al.

    * VLA-based system for automated cocktail making with a dual-arm robot, integrating vision (YOLOv8, EasyOCR), speech-to-text, and LLM instruction generation.  Application-specific system.

* **VLA Model-Expert Collaboration** (arXiv 2025) - Xiang, T.-Y., et al.

    * Enables human experts to collaborate with VLA models by providing corrective actions via shared autonomy. Achieves bi-directional learning (VLA improves, humans also improve). 

### Navigation

Focuses on tasks where a robot moves through an environment based on visual input and language instructions. Includes Vision-Language Navigation (VLN) and applications for legged robots.

* **NaVILA** ((https://arxiv.org/abs/2412.04453)) - Chen, X., et al.

    * [Project](https://navila-vln.github.io/).

    * Two-stage framework for legged robot VLN. High-level VLA outputs actions in mid-level language form (e.g., "move forward 75cm"), and a low-level vision-motor policy executes them. Decouples high-level reasoning from low-level control. 

* **QUAR-VLA / QUART** (ECCV 2024 / arXiv 2023) - Tang, J., et al.

    * [Project](https://sites.google.com/view/quar-vla).

    * Paradigm and VLA model family (QUART) for quadruped robots, integrating vision and language for navigation, complex terrain traversal, and manipulation. Includes the QUARD dataset. 

* **NaviLLM** (arXiv 2023) - Shah, D., et al.

    * [Project / Code](https://github.com/allenai/navillm).

    * General navigation model using LLMs for planning and interpreting instructions in diverse environments.

* **NaVid** (arXiv 2024) - Chen, X., et al.

    * [Project](https://navid-vlmap.github.io/).

    * Focuses on next-step planning in navigation using VLMs. Earlier work related to NaVILA.

### Human-Robot Interaction (HRI)

Focuses on enabling more natural and effective interactions between humans and robots, often using language (text or speech) as the primary interface.

* **VLAS (Vision-Language-Action-Speech)** ((https://arxiv.org/abs/2502.13508)) - Zhao, W., et al. ([https://openreview.net/forum?id=K4FAFNRpko](https://openreview.net/forum?id=K4FAFNRpko)).

    * First end-to-end VLA with direct integration of speech commands, without needing an external ASR. Built upon LLaVA. Includes the SQA and CSI datasets. Uses Voice RAG to handle personalized tasks.

* **Shake-VLA** (arXiv 2025) - Abdelkader, H., et al.

    * Integrates voice commands for a dual-arm cocktail-making robot.

* **VLA Model-Expert Collaboration** (arXiv 2025) - Xiang, T.-Y., et al.

    * Enables human-robot collaboration through shared autonomy, improving both VLA and human performance.

* **Helix** (Figure AI blog post 2024) - Figure AI.

    * [Project](https://www.figure.ai/news/helix).

    * Uses a single VLA model to enable multiple robots to collaborate on shared tasks (e.g., tidying up groceries).

### Task Planning / Reasoning

Focuses on using VLA/LLM components for high-level task decomposition, planning, and reasoning, often bridging the gap between complex instructions and low-level actions.

* **SayCan (Do As I Can, Not As I Say)** ((https://arxiv.org/abs/2204.01691)) - Ahn, M., et al.

    * (Google).

    * [Project](https://say-can.github.io/) / [Code](https://github.com/google-research/google-research/tree/master/saycan)

    * Grounds LLM planning in robot affordances.

* **PaLM-E** (ICML 2023 / arXiv 2023) - Driess, D., et al. (Google).

    * [Project](https://palm-e.github.io/).

    * Can perform sequential manipulation planning end-to-end or output language subgoals. Shows visual chain-of-thought reasoning abilities.

* **EmbodiedGPT** (arXiv 2023) - Mu, Y., et al.

    * [Code](https://github.com/OpenGVLab/EmbodiedGPT).

    * Multimodal model that performs end-to-end planning and reasoning for embodied tasks.

* **CoT-VLA (Chain-of-Thought VLA)** ((https://arxiv.org/abs/2503.22020)) - Wu, J., et al.

    * [Project](https://cot-vla.github.io/).

    * Explicitly incorporates visual CoT reasoning by predicting future goal images.

* **Hi Robot** (arXiv 2025) - Physical Intelligence Team.

    * [Project](https://www.physicalintelligence.company/research/hirobot).

    * Hierarchical VLA where a high-level VLM reasons and decomposes tasks for a low-level VLA (π0) executor.

* **LLM-Planner** ((https://ieeexplore.ieee.org/document/10160437)) - Liu, B., et al.

    * Modular planner using LLMs.

* **Code as Policies (CaP)** ((https://arxiv.org/abs/2212.06081)) - Liang, J., et al. (Google).

    * [Project](https://code-as-policies.github.io/).

    * Uses LLMs to directly generate robot policy code.

* **ConceptGraphs** ((https://arxiv.org/abs/2306.08999)) - Vemprala, S., et al.

    * [Project](https://concept-graphs.github.io/).

    * Uses LLMs to build and query scene graphs for planning and grounding.

* **Inner Monologue** ((https://arxiv.org/abs/2207.05608)) - Huang, W., et al.

    * [Project](https://inner-monologue.github.io/).

    * Uses language feedback from VLM/LLMs to guide robot policies.

* **The Critical Role of Hierarchical Reasoning for Complexity:** Many successful approaches to complex, long-horizon tasks employ hierarchical structures. This can be explicit (e.g., NaVILA's VLA + motor policy; Helix's System 1/2; Hi Robot's VLM planner + VLA executor), or implicit (e.g., SayCan grounding LLM plans in affordances; CoT-VLA generating intermediate visual goals). This suggests that monolithic end-to-end VLAs may struggle with deep reasoning or long-term planning compared to approaches that leverage emergent abilities of a single large model. Architectures that separate high-level planning/reasoning from low-level reactive control appear more effective. This architectural trend reflects the inherent complexity of linking semantic understanding to robust physical execution over extended periods.