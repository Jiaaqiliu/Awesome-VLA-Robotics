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
