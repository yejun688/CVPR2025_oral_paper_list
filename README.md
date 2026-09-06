# 🎉 CVPR 2025 Oral Paper List

A collection of **95 oral papers** from CVPR 2025, organized by topic, with links to papers, project pages, and code.

[![CVPR 2025](https://img.shields.io/badge/CVPR-2025-2563eb)](https://cvpr.thecvf.com/Conferences/2025)
[![Oral Papers](https://img.shields.io/badge/Oral_Papers-95-16a34a)](https://cvpr.thecvf.com/virtual/2025/papers.html)

**Official resources:** [Program](https://cvpr.thecvf.com/virtual/2025/papers.html) · [Proceedings](https://openaccess.thecvf.com/CVPR2025?day=all) · [Awards](https://cvpr.thecvf.com/Conferences/2025/BestPapersDemos)

## 📚 Browse by Topic

| Topic | Papers |
| --- | ---: |
| [3D Geometry & Reconstruction](#geometry) | 13 |
| [Rendering & 3D/4D Generation](#rendering) | 10 |
| [Humans, Avatars & Motion](#humans) | 5 |
| [Image & Video Generation](#generation) | 15 |
| [Vision-Language & Video Understanding](#multimodal) | 17 |
| [Embodied AI & Autonomous Driving](#embodied) | 6 |
| [Segmentation & Detection](#segmentation) | 5 |
| [Computational Imaging & Restoration](#imaging) | 8 |
| [Learning, Efficiency & Trustworthiness](#learning) | 13 |
| [Medical & Scientific Vision](#applications) | 3 |

## 🏆 Awards

Award labels follow the [official CVPR 2025 results](https://cvpr.thecvf.com/Conferences/2025/BestPapersDemos).

| Award | Paper |
| --- | --- |
| Best Paper | [VGGT](https://cvpr.thecvf.com/virtual/2025/oral/35294) |
| Best Student Paper | [Neural Inverse Rendering from Propagating Light](https://cvpr.thecvf.com/virtual/2025/oral/35315) |
| Best Paper Honorable Mention | [MegaSaM](https://cvpr.thecvf.com/virtual/2025/oral/35311) |
| Best Paper Honorable Mention | [Navigation World Models](https://cvpr.thecvf.com/virtual/2025/oral/35338) |
| Best Paper Honorable Mention | [Molmo and PixMo](https://cvpr.thecvf.com/virtual/2025/oral/35281) |
| Best Paper Honorable Mention | [3D Student Splatting and Scooping](https://cvpr.thecvf.com/virtual/2025/oral/35367) |
| Best Student Paper Honorable Mention | [Generative Multimodal Pretraining with Discrete Diffusion Timestep Tokens](https://cvpr.thecvf.com/virtual/2025/oral/35376) |

<a id="geometry"></a>

## 📐 3D Geometry & Reconstruction

| Paper | Links |
| --- | --- |
| VGGT: Visual Geometry Grounded Transformer | [Paper](https://arxiv.org/abs/2503.11651) · [Project](https://vgg-t.github.io/) · [Code](https://github.com/facebookresearch/vggt) |
| **CUT3R** — Continuous 3D Perception Model with Persistent State | [Paper](https://arxiv.org/abs/2501.12387) · [Project](https://cut3r.github.io/) · [Code](https://github.com/CUT3R/CUT3R) |
| MoGe: Unlocking Accurate Monocular Geometry Estimation for Open-Domain Images with Optimal Training Supervision | [Paper](https://arxiv.org/abs/2410.19115) · [Project](https://wangrc.site/MoGePage/) · [Code](https://github.com/microsoft/moge) |
| FoundationStereo: Zero-Shot Stereo Matching | [Paper](https://arxiv.org/abs/2501.09898) · [Project](https://nvlabs.github.io/FoundationStereo/) · [Code](https://github.com/NVlabs/FoundationStereo/) |
| Multi-view Reconstruction via SfM-guided Monocular Depth Estimation | [Paper](https://arxiv.org/abs/2503.14483) · [Project](https://zju3dv.github.io/murre/) · [Code](https://github.com/zju3dv/Murre) |
| MV-DUSt3R+: Single-Stage Scene Reconstruction from Sparse Views In 2 Seconds | [Paper](https://arxiv.org/abs/2412.06974) · [Project](https://mv-dust3rp.github.io/) · [Code](https://github.com/facebookresearch/mvdust3r) |
| MegaSaM: Accurate, Fast and Robust Structure and Motion from Casual Dynamic Videos | [Paper](https://arxiv.org/abs/2412.04463) · [Project](https://mega-sam.github.io/) · [Code](https://github.com/mega-sam/mega-sam) |
| Stereo4D: Learning How Things Move in 3D from Internet Stereo Videos | [Paper](https://arxiv.org/abs/2412.09621) · [Project](https://stereo4d.github.io/) · [Code](https://github.com/Stereo4d/stereo4d-code) |
| Zero-Shot Monocular Scene Flow Estimation in the Wild | [Paper](https://arxiv.org/abs/2501.10357) · [Project](https://research.nvidia.com/labs/lpr/zero_msf/) · [Code](https://github.com/NVlabs/zero-msf/) |
| DIFIX3D+: Improving 3D Reconstructions with Single-Step Diffusion Models | [Paper](https://arxiv.org/abs/2503.01774) · [Project](https://research.nvidia.com/labs/toronto-ai/difix3d/) · [Code](https://github.com/nv-tlabs/Difix3D) |
| TacoDepth: Towards Efficient Radar-Camera Depth Estimation with One-stage Fusion | [Paper](https://arxiv.org/abs/2504.11773) · [Code](https://github.com/RaymondWang987/TacoDepth) |
| Convex Relaxation for Robust Vanishing Point Estimation in Manhattan World | [Paper](https://arxiv.org/abs/2505.04788) · [Code](https://github.com/WU-CVGL/GlobustVP) |
| Camera Resection from Known Line Pencils and a Radially Distorted Scanline | [Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Dibene_Camera_Resection_from_Known_Line_Pencils_and_a_Radially_Distorted_CVPR_2025_paper.html) · [Code](https://github.com/jdibenes/pmd) |

<a id="rendering"></a>

## ✨ Rendering & 3D/4D Generation

| Paper | Links |
| --- | --- |
| Neural Inverse Rendering from Propagating Light | [Paper](https://arxiv.org/abs/2506.05347) · [Project](https://anaghmalik.com/InvProp/) · [Code](https://github.com/benattal/neural-radiance-caching) |
| Diffusion Renderer: Neural Inverse and Forward Rendering with Video Diffusion Models | [Paper](https://arxiv.org/abs/2501.18590) · [Project](https://research.nvidia.com/labs/toronto-ai/DiffusionRenderer/) · [Code](https://github.com/nv-tlabs/cosmos1-diffusion-renderer) |
| 3D Student Splatting and Scooping | [Paper](https://arxiv.org/abs/2503.10148) · [Code](https://github.com/realcrane/3D-student-splating-and-scooping) |
| 3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting | [Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_3DGUT_Enabling_Distorted_Cameras_and_Secondary_Rays_in_Gaussian_Splatting_CVPR_2025_paper.html) · [Code](https://github.com/nv-tlabs/3dgrut) |
| Time of the Flight of the Gaussians: Optimizing Depth Indirectly in Dynamic Radiance Fields | [Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Li_Time_of_the_Flight_of_the_Gaussians_Optimizing_Depth_Indirectly_CVPR_2025_paper.html) · [Project](https://visual.cs.brown.edu/projects/gftorf-webpage/) · [Code](https://github.com/brownvc/gftorf) |
| FluidNexus: 3D Fluid Reconstruction and Prediction from a Single Video | [Paper](https://arxiv.org/abs/2503.04720) · [Project](https://yuegao.me/FluidNexus/) · [Code](https://github.com/ueoo/FluidNexus) |
| CraftsMan3D: High-fidelity Mesh Generation with 3D Native Diffusion and Interactive Geometry Refiner | [Paper](https://arxiv.org/abs/2405.14979) · [Project](https://craftsman3d.github.io/) · [Code](https://github.com/wyysf-98/CraftsMan3D) |
| CAT4D: Create Anything in 4D with Multi-View Video Diffusion Models | [Paper](https://arxiv.org/abs/2411.18613) · [Project](https://cat-4d.github.io/) |
| DNF: Unconditional 4D Generation with Dictionary-based Neural Fields | [Paper](https://arxiv.org/abs/2412.05161) · [Project](https://xzhang-t.github.io/project/DNF/) · [Code](https://github.com/xzhang-t/DNF) |
| Birth and Death of a Rose | [Paper](https://arxiv.org/abs/2412.05278) · [Project](https://chen-geng.com/rose4d) |

<a id="humans"></a>

## 🧍 Humans, Avatars & Motion

| Paper | Links |
| --- | --- |
| CAP4D: Creating Animatable 4D Portrait Avatars with Morphable Multi-View Diffusion Models | [Paper](https://arxiv.org/abs/2412.12093) · [Project](https://felixtaubner.github.io/cap4d/) · [Code](https://github.com/felixtaubner/cap4d/) |
| Reconstructing Humans with a Biomechanically Accurate Skeleton | [Paper](https://arxiv.org/abs/2503.21751) · [Project](https://isshikihugh.github.io/HSMR/) · [Code](https://github.com/IsshikiHugh/HSMR) |
| MEGA: Masked Generative Autoencoder for Human Mesh Recovery | [Paper](https://arxiv.org/abs/2405.18839) |
| TokenHSI: Unified Synthesis of Physical Human-Scene Interactions through Task Tokenization | [Paper](https://arxiv.org/abs/2503.19901) · [Project](https://liangpan99.github.io/TokenHSI/) · [Code](https://github.com/liangpan99/TokenHSI) |
| EgoLM: Multi-Modal Language Model of Egocentric Motions | [Paper](https://arxiv.org/abs/2409.18127) · [Project](https://hongfz16.github.io/projects/EgoLM) |

<a id="generation"></a>

## 🎨 Image & Video Generation

| Paper | Links |
| --- | --- |
| Infinity∞: Scaling Bitwise AutoRegressive Modeling for High-Resolution Image Synthesis | [Paper](https://arxiv.org/abs/2412.04431) · [Project](https://foundationvision.github.io/infinity.project/) · [Code](https://github.com/FoundationVision/Infinity) |
| RandAR: Decoder-only Autoregressive Visual Generation in Random Orders | [Paper](https://arxiv.org/abs/2412.01827) · [Project](https://rand-ar.github.io/) · [Code](https://github.com/ziqipang/RandAR) |
| Reconstruction vs. Generation: Taming Optimization Dilemma in Latent Diffusion Models | [Paper](https://arxiv.org/abs/2501.01423) · [Code](https://github.com/hustvl/LightningDiT) |
| Alias-Free Latent Diffusion Models: Improving Fractional Shift Equivariance of Diffusion Latent Space | [Paper](https://arxiv.org/abs/2503.09419) · [Project](https://zhouyifan.net/AF-LDM-Page/) · [Code](https://github.com/SingleZombie/AFLDM) |
| Autoregressive Distillation of Diffusion Transformers | [Paper](https://arxiv.org/abs/2504.11295) · [Code](https://github.com/alsdudrla10/ARD) |
| Language-Guided Image Tokenization for Generation | [Paper](https://arxiv.org/abs/2412.05796) · [Project](https://kaiwenzha.github.io/textok/) |
| AnyEdit: Mastering Unified High-Quality Image Editing for Any Idea | [Paper](https://arxiv.org/abs/2411.15738) · [Project](https://dcd-anyedit.github.io/) · [Code](https://github.com/DCDmllm/AnyEdit) |
| CustAny: Customizing Anything from A Single Example | [Paper](https://arxiv.org/abs/2406.11643v4) · [Project](https://lingjiekong-fdu.github.io/) · [Code](https://github.com/LingjieKong-fdu/CustAny) |
| DreamRelation: Bridging Customization and Relation Generation | [Paper](https://arxiv.org/abs/2410.23280) · [Project](https://shi-qingyu.github.io/DreamRelation.github.io/) · [Code](https://github.com/Shi-qingyu/DreamRelation) |
| Minority-Focused Text-to-Image Generation via Prompt Optimization | [Paper](https://arxiv.org/abs/2410.07838) · [Code](https://github.com/soobin-um/MinorityPrompt) |
| DesignDiffusion: High-Quality Text-to-Design Image Generation with Diffusion Models | [Paper](https://arxiv.org/abs/2503.01645) |
| Motion Prompting: Controlling Video Generation with Motion Trajectories | [Paper](https://arxiv.org/abs/2412.02700) · [Project](https://motion-prompting.github.io/) |
| Go-with-the-Flow: Motion-Controllable Video Diffusion Models Using Real-Time Warped Noise | [Paper](https://arxiv.org/abs/2501.08331) · [Project](https://eyeline-research.github.io/Go-with-the-Flow/) · [Code](https://github.com/Eyeline-Research/Go-with-the-Flow) |
| LookingGlass: Generative Anamorphoses via Laplacian Pyramid Warping | [Paper](https://arxiv.org/abs/2504.08902) · [Project](https://lookingglass-lpw.github.io/) |
| Reanimating Images using Neural Representations of Dynamic Stimuli | [Paper](https://arxiv.org/abs/2406.02659) |

<a id="multimodal"></a>

## 💬 Vision-Language & Video Understanding

| Paper | Links |
| --- | --- |
| Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models | [Paper](https://arxiv.org/abs/2409.17146) · [Blog](https://allenai.org/blog/molmo) · [Code](https://github.com/allenai/molmo) |
| Generative Multimodal Pretraining with Discrete Diffusion Timestep Tokens | [Paper](https://arxiv.org/abs/2504.14666) · [Project](https://ddt-llama.github.io/) · [Code](https://github.com/selftok-team/SelftokTokenizer/) |
| LoRASculpt: Sculpting LoRA for Harmonizing General and Specialized Knowledge in Multimodal Large Language Models | [Paper](https://arxiv.org/abs/2503.16843) |
| **OPA-DPO** — Mitigating Hallucinations in Large Vision-Language Models via DPO: On-Policy Data Hold the Key | [Paper](https://arxiv.org/abs/2501.09695) · [Project](https://opa-dpo.github.io/) · [Code](https://github.com/zhyang2226/OPA-DPO) |
| Seeing Far and Clearly: Mitigating Hallucinations in MLLMs with Attention Causal Decoding | [Paper](https://arxiv.org/abs/2505.16652) · [Project](https://mllms-farsight.github.io/) · [Code](https://github.com/FeilongTangmonash/FarSight) |
| Identifying and Mitigating Position Bias of Multi-image Vision-Language Models | [Paper](https://arxiv.org/abs/2503.13792) |
| Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces | [Paper](https://arxiv.org/abs/2412.14171) · [Project](https://vision-x-nyu.github.io/thinking-in-space.github.io/) · [Code](https://github.com/vision-x-nyu/thinking-in-space) |
| Video-XL: Extra-Long Vision Language Model for Hour-Scale Video Understanding | [Paper](https://arxiv.org/abs/2409.14485) · [Code](https://github.com/VectorSpaceLab/Video-XL) |
| VideoEspresso: A Large-Scale Chain-of-Thought Dataset for Fine-Grained Video Reasoning via Core Frame Selection | [Paper](https://arxiv.org/abs/2411.14794) · [Code](https://github.com/hshjerry/VideoEspresso) |
| OpenING: A Comprehensive Benchmark for Judging Open-ended Interleaved Image-Text Generation | [Paper](https://arxiv.org/abs/2411.18499) · [Project](https://opening-benchmark.github.io/) · [Code](https://github.com/LanceZPF/OpenING) |
| Q-Eval-100K: Evaluating Visual Quality and Alignment Level for Text-to-Vision Content | [Paper](https://arxiv.org/abs/2503.02357) · [Code](https://github.com/zzc-1998/Q-Eval) |
| SEAL: Semantic Attention Learning for Long Video Representation | [Paper](https://arxiv.org/abs/2412.01798) · [Code](https://github.com/SEAL-lvu/SEAL) |
| Learning Audio-guided Video Representation with Gated Attention for Video-Text Retrieval | [Paper](https://arxiv.org/abs/2504.02397) |
| Temporal Alignment-Free Video Matching for Few-shot Action Recognition | [Paper](https://arxiv.org/abs/2504.05956) · [Code](https://github.com/leesb7426/TEAM) |
| Viewpoint Rosetta Stone: Unlocking Unpaired Ego-Exo Videos for View-invariant Representation Learning | [Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Luo_Viewpoint_Rosetta_Stone_Unlocking_Unpaired_Ego-Exo_Videos_for_View-invariant_Representation_CVPR_2025_paper.html) · [Project](https://vision.cs.utexas.edu/projects/ViewpointRosetta/) |
| Temporally Consistent Object-Centric Learning by Contrasting Slots | [Paper](https://arxiv.org/abs/2412.14295) · [Project](https://slotcontrast.github.io/) · [Code](https://github.com/martius-lab/slotcontrast) |
| The PanAf-FGBG Dataset: Understanding the Impact of Backgrounds in Wildlife Behaviour Recognition | [Paper](https://arxiv.org/abs/2502.21201) · [Project](https://obrookes.github.io/panaf-fgbg.github.io/) |

<a id="embodied"></a>

## 🤖 Embodied AI & Autonomous Driving

| Paper | Links |
| --- | --- |
| Navigation World Models | [Paper](https://arxiv.org/abs/2412.03572) · [Project](https://www.amirbar.net/nwm/) · [Code](https://github.com/facebookresearch/nwm/) |
| RoboSpatial: Teaching Spatial Understanding to 2D and 3D Vision-Language Models for Robotics | [Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Song_RoboSpatial_Teaching_Spatial_Understanding_to_2D_and_3D_Vision-Language_Models_CVPR_2025_paper.html) · [Project](https://chanh.ee/RoboSpatial/) · [Code](https://github.com/NVlabs/RoboSpatial) |
| From Multimodal LLMs to Generalist Embodied Agents: Methods and Lessons | [Paper](https://arxiv.org/abs/2412.08442) |
| PDFactor: Learning Tri-Perspective View Policy Diffusion Field for Multi-Task Robotic Manipulation | [Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Tian_PDFactor_Learning_Tri-Perspective_View_Policy_Diffusion_Field_for_Multi-Task_Robotic_CVPR_2025_paper.html) |
| GROVE: A Generalized Reward for Learning Open-Vocabulary Physical Skill | [Paper](https://arxiv.org/abs/2504.04191) · [Project](https://jiemingcui.github.io/grove/) · [Code](https://github.com/jiemingcui/GROVE-pytorch) |
| Closed-Loop Supervised Fine-Tuning of Tokenized Traffic Models | [Paper](https://arxiv.org/abs/2412.05334) · [Project](https://zhejz.github.io/catk/) · [Code](https://github.com/NVlabs/catk) |

<a id="segmentation"></a>

## 🎯 Segmentation & Detection

| Paper | Links |
| --- | --- |
| SegEarth-OV: Towards Training-Free Open-Vocabulary Segmentation for Remote Sensing Images | [Paper](https://arxiv.org/abs/2410.01768) · [Project](https://likyoo.github.io/SegEarth-OV/) · [Code](https://github.com/likyoo/SegEarth-OV) |
| Effective SAM Combination for Open-Vocabulary Semantic Segmentation | [Paper](https://arxiv.org/abs/2411.14723) |
| Keep the Balance: A Parameter-Efficient Symmetrical Framework for RGB+X Semantic Segmentation | [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Cai_Keep_the_Balance_A_Parameter-Efficient_Symmetrical_Framework_for_RGBX_Semantic_CVPR_2025_paper.pdf) |
| Towards Explicit Geometry-Reflectance Collaboration for Generalized LiDAR Segmentation in Adverse Weather | [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_Towards_Explicit_Geometry-Reflectance_Collaboration_for_Generalized_LiDAR_Segmentation_in_Adverse_CVPR_2025_paper.pdf) |
| Efficient Test-time Adaptive Object Detection via Sensitivity-Guided Pruning | [Paper](https://xueyangfu.github.io/paper/2025/SGP_CVPR_25.pdf) |

<a id="imaging"></a>

## 📷 Computational Imaging & Restoration

| Paper | Links |
| --- | --- |
| Learned Binocular-Encoding Optics for RGBD Imaging Using Joint Stereo and Focus Cues | [Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Learned_Binocular-Encoding_Optics_for_RGBD_Imaging_Using_Joint_Stereo_and_CVPR_2025_paper.html) · [Project](https://liangxunou.github.io/25liulearned/) · [Code](https://github.com/Lorena-Y-Liu/Learned-Binocular-Optics) |
| Opportunistic Single-Photon Time of Flight | [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Nousias_Opportunistic_Single-Photon_Time_of_Flight_CVPR_2025_paper.pdf) |
| Descriptor-In-Pixel : Point-Feature Tracking For Pixel Processor Arrays | [Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Bose_Descriptor-In-Pixel__Point-Feature_Tracking_For_Pixel_Processor_Arrays_CVPR_2025_paper.html) · [Project](https://lauriebose.github.io/DIP/) |
| Removing Reflections from RAW Photos | [Paper](https://arxiv.org/abs/2404.14414) · [Project](https://erickee.com/reflections/cvpr2025.html) |
| DORNet: A Degradation Oriented and Regularized Network for Blind Depth Super-Resolution | [Paper](https://arxiv.org/abs/2410.11666) · [Code](https://github.com/yanzq95/DORNet) |
| Improving Diffusion Inverse Problem Solving with Decoupled Noise Annealing | [Paper](https://arxiv.org/abs/2407.01521) · [Project](https://daps-inverse-problem.github.io/) · [Code](https://github.com/zhangbingliang2019/DAPS) |
| DiffFNO: Diffusion Fourier Neural Operator | [Paper](https://arxiv.org/abs/2411.09911) · [Project](https://jasonliu2024.github.io/difffno-diffusion-fourier-neural-operator/) |
| Semi-Supervised State-Space Model with Dynamic Stacking Filter for Real-World Video Deraining | [Paper](https://arxiv.org/abs/2505.16811) |

<a id="learning"></a>

## 🧠 Learning, Efficiency & Trustworthiness

| Paper | Links |
| --- | --- |
| OverLoCK: An Overview-first-Look-Closely-next ConvNet with Context-Mixing Dynamic Kernels | [Paper](https://arxiv.org/abs/2502.20087) · [Code](https://github.com/LMMMEng/OverLoCK) |
| CleanDIFT: Diffusion Features without Noise | [Paper](https://arxiv.org/abs/2412.03439) · [Project](https://compvis.github.io/cleandift/) · [Code](https://github.com/CompVis/cleandift) |
| LibraGrad: Balancing Gradient Flow for Universally Better Vision Transformer Attributions | [Paper](https://www.arxiv.org/abs/2411.16760) · [Code](https://github.com/NightMachinery/LibraGrad) |
| Do We Always Need the Simplicity Bias? Looking for Optimal Inductive Biases in the Wild | [Paper](https://arxiv.org/abs/2503.10065) |
| Rethinking Spiking Self-Attention Mechanism: Implementing α-XNOR Similarity Calculation in Spiking Transformers | [Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Xiao_Rethinking_Spiking_Self-Attention_Mechanism_Implementing_a-XNOR_Similarity_Calculation_in_Spiking_CVPR_2025_paper.html) |
| Towards Universal Dataset Distillation via Task-Driven Diffusion | [Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Qi_Towards_Universal_Dataset_Distillation_via_Task-Driven_Diffusion_CVPR_2025_paper.html) |
| Gromov–Wasserstein Problem with Cyclic Symmetry | [Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Takeda_Gromov-Wasserstein_Problem_with_Cyclic_Symmetry_CVPR_2025_paper.html) |
| UniAP: Unifying Inter- and Intra-Layer Automatic Parallelism by Mixed Integer Quadratic Programming | [Paper](https://arxiv.org/abs/2307.16375) |
| Enhancing Diversity for Data-free Quantization | [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhao_Enhancing_Diversity_for_Data-free_Quantization_CVPR_2025_paper.pdf) |
| Geometric Knowledge-Guided Localized Global Distribution Alignment for Federated Learning | [Paper](https://arxiv.org/abs/2503.06457) · [Code](https://github.com/WeiDai-David/2025CVPR_GGEUR) |
| Black-Box Forgery Attacks on Semantic Watermarks for Diffusion Models | [Paper](https://arxiv.org/abs/2412.03283) · [Code](https://github.com/and-mill/semantic-forgery) |
| Adv-CPG: A Customized Portrait Generation Framework with Facial Adversarial Attacks | [Paper](https://arxiv.org/abs/2503.08269) · [Code](https://github.com/April-yy/Adv-CPG) |
| Rethinking Vision-Language Model in Face Forensics: Multi-Modal Interpretable Forged Face Detector | [Paper](https://arxiv.org/abs/2503.20188) · [Code](https://github.com/CHELSEA234/M2F2_Det) |

<a id="applications"></a>

## 🔬 Medical & Scientific Vision

| Paper | Links |
| --- | --- |
| TopoCellGen: Generating Histopathology Cell Topology with a Diffusion Model | [Paper](https://arxiv.org/abs/2412.06011) · [Code](https://github.com/Melon-Xu/TopoCellGen) |
| Enhancing SAM with Efficient Prompting and Preference Optimization for Semi-supervised Medical Image Segmentation | [Paper](https://arxiv.org/abs/2503.04639) |
| IceDiff: High Resolution and High-Quality Arctic Sea Ice Forecasting with Generative Diffusion Prior | [Paper](https://arxiv.org/abs/2410.09111) |

## 📎 Related Reading

<details>
<summary>4 additional papers from the original collection</summary>

These entries are retained for reference and are not counted among the 95 papers in the official oral program.

| Paper | Links |
| --- | --- |
| Exploring CLIP's Dense Knowledge for Weakly Supervised Semantic Segmentation | [Paper](https://arxiv.org/abs/2503.20826) · [Code](https://github.com/zwyang6/ExCEL) |
| FedSPA: Generalizable Federated Graph Learning under Homophily Heterogeneity | [Paper](https://www.cs.emory.edu/~jyang71/files/fedspa.pdf) |
| One Category One Prompt: Dataset Distillation using Diffusion Models | [Paper](https://arxiv.org/abs/2403.07142) · [Code](https://github.com/mint-vu/D3M) |
| Video-XL-Pro: Reconstructive Token Compression for Extremely Long Video Understanding | [Paper](https://arxiv.org/abs/2503.18478) · [Code](https://github.com/VectorSpaceLab/Video-XL) |

</details>

## 🌷 Contributing & Acknowledgments

Missing a link or spotted a mistake? [Open an issue](https://github.com/yejun688/CVPR_2025_Oral_Paper_List/issues) or submit a pull request with the paper title and updated links.

Thanks to the paper authors for sharing their work, and to [cvpr25_oral_gpu_info](https://github.com/kxhit/cvpr25_oral_gpu_info) for the original collection reference.
