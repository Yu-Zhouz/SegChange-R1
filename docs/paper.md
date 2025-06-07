# SegChange-R1 : Augmented Reasoning for Remote Sensing Change Detection via Large Language Models

## Abstract

遥感变化检测主要通过分析同一空间区域在不同时相上地物的显著变化差异（如建筑物变化），广泛应用于城市规划、地形地貌分析和环境监测等多个领域。本文提出了一种大语言模型（LLM）增强式推理的方法（SegChange-R1），它通过集成文本描述信息以增强检测能力，旨在引导模型分割出更感兴趣的变化区域，从而加快收敛速度。并且，我们基于线性注意力设计了空间转换模块（BEV）, 该模块通过将不同时态视角下的特征统一到 BEV 空间上，解决变化检测中模态错位的问题。此外，我们还构建了首个无人机视角下建筑物变化检测数据集（DVChange ）我们在四个广泛使用的变化检测数据集上的实验表明，与现有的最先进的（SOTA）方法相比，它有显著的改进。代码和预训练模型可在 [Yu-Zhouz/SegChange-R1](https://github.com/Yu-Zhouz/SegChange-R1)。
## 1 Introduction

在遥感中，变化检测（Change Detection, CD）是指通过分析同一区域在不同时相获取的遥感图像，识别地表特征的变化情况 [1, 2]。CD 任务具有广泛的应用背景，如城市扩张监测 [50, 26]、灾害评估 [43, 69]、土地利用与覆盖变化分析 [37, 16, 61] 以及军事侦察 [62, 36] 等。然而，由于遥感图像受多种因素影响，CD 仍面临诸多挑战。
### 1.1 Related Work

首先，不同时间点采集的图像常存在光照和季节变化，导致相同地物在不同时相下表现出显著的光谱差异 [63, 42]。其次，多源遥感数据的分辨率不一致也会影响变化信息的提取精度 [5]。此外，传感器噪声、大气干扰等因素引入了图像噪声，进一步增加了变化建模的难度 [1, 39]。另一个关键问题是图像配准误差，即使经过预处理，微小的空间错位也可能导致错误的变更判断 [31, 6]。
近年来，基于卷积神经网络的方法（如 FC-EF 和 FC-Siam-diff ）通过孪生结构增强了双时相图像的特征一致性，提升了检测性能。为进一步建模长距离依赖关系，基于 Transformer 的方法（如 BIT 和 ChangeFormer ）引入自注意力机制，在多尺度变化建模上表现优异。Siamese 提出基于 Mamba 的架构，利用状态空间模型处理长序列遥感数据，提升了建模效率。然而，大多数方法仍局限于视觉特征的对齐与比较，缺乏语义层面的理解与推理能力，影响变化特征的对齐精度。近期，BEV 空间建模被引入以统一视角表达，用于缓解该问题。此外，如何结合上下文信息（如文本描述或地理标签）来增强模型对感兴趣区域的关注，仍是值得探索的方向 [48]。

### 1.2 Contributions

最近，大型语言模型（LLM）的发展为该领域带来了新的机会。在本文中，我们提出了一种名为 SegChange-R1 的新方法，该方法利用 LLM 将文本描述信息与遥感图像相结合，指导模型更多地关注感兴趣的区域，从而提高双时相地物显著变化差异的检测效果。如 Fig 2. 所示，特别是，我们基于线性注意力设计了空间转换模块（BEV）, 该模块通过将不同时态视角下的特征统一到 BEV 空间上，解决变化检测中模态错位的问题。与 transformer 相比，基于线性注意力的架构通过建模全局依赖关系来增强特征表达能力，展示了线性时间训练能力和更高效地建模空间依赖关系。综上所述，我们工作的主要贡献如下：
1. 我们开发了一种新颖的基于大预言模型语义引导通增强式推理遥感变化检测器（SegChange-R1），它通过集成文本描述信息在给定两张图像的情况下生成更精确的位置掩码。
2. 为此，我们基于线性注意力设计了一个空间转换模块（BEV），用于解决变化检测中模态错位的问题。
3. 构建首个无人机视角下建筑物变化检测数据集（DVSC），该数据集共 13800 对变化图像，涵盖城市到农村多种场景下的建筑物变化情况。

## 2 Related Work

### 2.1 Deep Learning Approaches for Remote Sensing Change Detection

遥感变化检测作为一项重要的遥感应用技术，经历了从传统方法到深度学习方法的革命性演进过程。早期的变化检测方法主要基于像素级的差异分析，包括图像差值法、比值法、变化向量分析和主成分分析等[1, 2, 14]。这些传统方法虽然计算简单，但往往受到光照变化、季节变化、传感器噪声和大气条件等多种因素的影响，导致虚警率较高，难以准确识别真实的地物变化。

随着深度学习技术的蓬勃发展，基于卷积神经网络（CNN）的变化检测方法逐渐成为主流范式。早期的深度学习方法如FC-EF（Fully Convolutional Early Fusion）采用早期融合策略，将双时相图像在输入层进行拼接[15]。随后，FC-Siam-diff（Fully Convolutional Siamese Difference）和FC-Siam-conc等孪生网络结构被提出，通过共享权重的特征提取器对双时相图像进行并行处理，然后通过差分或拼接操作进行特征融合[28, 41]。这些方法通过参数共享确保了特征的一致性，显著提升了变化检测的准确性和鲁棒性。

为了进一步提升检测性能，研究者们开始探索更复杂的网络架构和注意力机制。STANet[7]引入了时空注意力机制，通过空间注意力和通道注意力模块增强模型对变化区域的关注能力。DTCDSCN[68]提出了双任务约束的深度孪生卷积网络，通过语义分割任务的辅助训练提升变化检测性能。IFN[70]设计了交互式特征融合网络，通过多层次的特征交互实现更精细的变化建模。近年来，基于注意力机制的方法如SNUNet[25]和FCCDN[4]进一步推动了该领域的发展，这些方法通过设计专门的注意力模块来增强特征表示和融合能力。

此外，多尺度特征融合也成为了重要的研究方向。FPN（Feature Pyramid Network）结构被广泛应用于变化检测任务中，通过自顶向下的特征传播和横向连接实现多尺度信息的有效融合[64, 29]。一些研究还探索了密集连接和残差连接等结构，以提升网络的表达能力和训练稳定性[22, 38]。

### 2.2 Transformer and Multi-modal Fusion for Change Detection

为了更好地建模长距离依赖关系和全局上下文信息，基于Transformer的变化检测方法应运而生，为该领域带来了新的突破。BIT（Binary Change Detection with Transformers）[9]首次将Transformer架构引入变化检测任务，通过自注意力机制捕获全局上下文信息，在多个基准数据集上取得了显著的性能提升。ChangeFormer[52]进一步改进了Transformer结构，设计了专门的变化感知注意力模块和层次化特征融合策略。SwinSUNet[35]结合了Swin Transformer的层次化特征表示能力和移动窗口机制，在处理多尺度变化时展现出优异的性能。

然而，传统的Transformer架构存在计算复杂度为O(n²)的问题，特别是在处理高分辨率遥感图像时面临显著的计算和内存挑战。为了解决这一问题，研究者们开始探索更高效的注意力机制和替代架构。ChangeMamba[18]基于状态空间模型（State Space Model）设计了线性复杂度的变化检测架构，通过Mamba的选择性扫描机制实现了高效的长序列建模。Linear Transformer[17, 30]通过核化自注意力机制将计算复杂度降低到线性级别，在保持检测精度的同时显著提升了计算效率。

多模态融合技术在遥感变化检测领域展现出巨大的潜力，特别是结合视觉和文本信息的方法。CLIP（Contrastive Language-Image Pre-training）[47]的成功证明了视觉-语言预训练在各种视觉任务中的有效性，为遥感领域的多模态应用奠定了基础。FLAVA[54]和ALIGN[27]等后续工作进一步探索了大规模多模态预训练的可能性。受此启发，一些研究者开始将语言信息引入遥感变化检测任务，探索文本描述如何指导模型关注特定类型的变化[48, 23]。

最近，大型语言模型（LLM）的快速发展为遥感变化检测带来了新的机遇。GPT-4V[44]、LLaVA[34]和InstructBLIP[13]等多模态大型语言模型展现了强大的视觉理解和推理能力。在遥感领域，LLM被逐步应用于图像描述生成、场景理解和目标检测等任务[21, 58, 12]。特别是，LLM在空间推理和区域定位方面的能力为遥感变化检测提供了新的可能性。通过将视觉特征与自然语言描述相结合，LLM能够更好地理解变化的语义含义，从而指导模型关注用户感兴趣的变化区域。

然而，现有的多模态融合方法在遥感变化检测中的应用仍然相对有限。大多数方法仍然依赖于纯视觉特征的比较，缺乏对语义层面变化的深度理解。如何有效地将LLM的推理能力整合到变化检测流程中，实现视觉特征与语言指令的深度融合，仍然是一个亟待解决的关键问题。

### 2.3 Spatial Alignment and Efficient Architectures

空间对齐和高效架构设计是遥感变化检测中的两个关键技术挑战。由于遥感图像获取过程中存在的各种干扰因素，如传感器位置差异、拍摄角度变化、大气条件影响等，不同时相的图像往往存在微小但不可忽略的空间错位问题[31, 6]。这种错位即使在亚像素级别也可能导致错误的变化检测结果，特别是在边缘区域和细小目标的检测中。

鸟瞰图（Bird's Eye View, BEV）表示作为一种统一的空间表示方法，在自动驾驶和3D目标检测领域得到了广泛应用[33, 56, 24]。BEV表示的核心优势在于能够将不同视角、不同传感器获取的数据统一到同一个空间坐标系中，有效解决视角变化和空间错位问题。LSS（Lift, Splat, Shoot）[33]通过深度估计将透视视图特征转换为BEV表示，而BEVFormer[32]进一步引入了时序信息的建模。最近，一些研究者开始将BEV表示引入遥感变化检测任务，以解决多时相图像之间的配准误差问题[46, 19]。通过将不同时相的图像特征映射到统一的BEV空间，可以减少空间错位对变化检测的影响。

传统的注意力机制虽然在建模全局依赖关系方面表现出色，但其O(n²)的计算复杂度在处理高分辨率遥感图像时成为瓶颈。线性注意力机制作为一种高效的替代方案，通过核化技巧或近似方法将计算复杂度降低到O(n)[30, 17]。Performer[10]通过随机特征映射实现了线性复杂度的注意力计算，而Linformer[57]通过低秩分解减少了注意力矩阵的维度。这些方法在保持相似性能的同时大幅降低了计算成本，特别适合处理高分辨率的遥感图像数据。

此外，混合专家模型（Mixture of Experts, MoE）和稀疏激活策略也为大规模遥感数据处理提供了新的思路[49, 11]。通过动态选择和激活网络的子模块，这些方法能够在保持模型容量的同时显著降低实际的计算开销。一些研究还探索了知识蒸馏和模型压缩技术在遥感变化检测中的应用，通过将大模型的知识转移到小模型中实现效率和性能的平衡[20, 45]。

在网络架构设计方面，研究者们不断探索更高效的特征提取和融合策略。ConvNeXt[40]和EfficientNet[59]等现代CNN架构被引入遥感变化检测任务，通过优化的卷积设计和通道注意力机制提升了特征表示能力。同时，Neural Architecture Search（NAS）技术也被应用于自动化地搜索最优的网络结构[60, 3]，为不同应用场景设计专门的高效架构。

## 3 Proposed Method

与自然图像相比，遥感图像表现出独特的特征，需要专门的建筑设计进行像素级地理空间推理。在这项工作中，我们提出了 SegEarth-R 1，这是一种简单而强大的地理空间像素推理基线，可有效利用 LLM 功能，同时结合特定领域的适应。如图 2 所示，我们的架构包括三个核心部分：用于图像特征提取的视觉编码器、用于指令解释和语义关联的 LLM 以及用于空间相关性和掩码预测的掩码生成器。每个部分都包含关键的设计考虑因素，以应对遥感图像的独特挑战。
<p align="center">
  <a href="https://example.com"> 
    <img src="./xxx.png" width="300" height="200" alt="图片名称" />
  </a>
</p>
Fig 2.
### 3.1  编码器

#### 3.1.1 多尺度视觉编码器

遥感图像具有显著的尺度变化特性，从亚米级的小型地物到公里级的大范围地理结构并存，这对模型的多尺度建模能力提出了严峻挑战 [56]。此外，高分辨率遥感图像中密集分布的小目标要求模型在特征提取过程中保留尽可能多的空间细节信息 [30]。然而，当前主流的基于 Vision Transformer 的编码器（如 CLIP [52] 和 SAM [25, 54]）在处理此类任务时存在局限性：其固定窗口机制和激进下采样策略容易导致小尺度目标的信息丢失，限制了模型对复杂遥感场景的感知能力。 为解决上述问题，我们采用了 Swin Transformer 主干网络 [43]，该架构通过滑动窗口机制实现局部注意力建模，在保持计算效率的同时增强了模型对细粒度特征的捕捉能力。在此基础上，我们构建了一个渐进式的多尺度特征提取框架，分别以原始输入图像的 1/4、1/8、1/16 和 1/32 分辨率生成特征图，记作 $ v_h \in [1, 4] $，从而在不同层级上兼顾空间分辨率与语义抽象。此外，我们在实现中支持多种骨干网络选择（包括 ResNet50、Swin Transformer 和 HGNetv2），以适应不同计算资源与精度需求下的应用场景。

#### 3.1.2 Text Encoder


为了将文本语义特征转换为图像特征，我们引入了一个基于 LLM 的文本编码器，该编码器将输入的文本描述转换为丰富语义信息的嵌入向量，并作为输入传递给视觉编码器进行进行深度融合，从而引导模型关注特定类型的地物变化。此外，我们还实现了动态序列长度控制机制，使得文本嵌入可以适配不同下游任务需求，当输入文本长度小于目标序列长度时，我们在末尾进行零填充；当超过目标长度时，则进行截断。这一机制提升了模型对多样化文本输入的兼容性。

### 3.2 BEV Space Converter

在遥感变化检测中，模态错位是一项重大挑战。为了解决这个问题，我们提出了一个新颖的 BEV 空间转换模块，它在我们的 SegChange-R 1 框架中起着关键作用。该模块旨在解决在处理在不同时间阶段捕获的遥感数据时出现的模态错位的固有挑战。该模块基于线性注意机制，可实现高效、有效的特征转换。这个想法是将来自不同时间视角的特征转换为统一的 BEV（鸟瞰图）空间，从而更有效地比较和分析变化。

<p align="center">
  <a href="https://example.com"> 
    <img src="https://sdmntpreastus.oaiusercontent.com/files/00000000-fb8c-61f9-bcec-5b8146c8b70d/raw?se=2025-06-05T09%3A57%3A16Z&sp=r&sv=2024-08-04&sr=b&scid=499c9886-98b2-557a-9b92-d1a16dd84255&skoid=02b7f7b5-29f8-416a-aeb6-99464748559d&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2025-06-05T05%3A04%3A41Z&ske=2025-06-06T05%3A04%3A41Z&sks=b&skv=2024-08-04&sig=MGzLUx3qy9b7yu//Exy6T%2BXIU8rnm6oC4NeZKMxpbS4%3D" width="300" height="300" alt="图片名称" />
  </a>
</p>
Fig 3. BEV空间转换器

The BEV Space Converter takes features from multiple time phases as input. These features are first projected into a latent space using linear transformations. Mathematically, this can be represented as follows:
$$
\mathbf{z}_t = W_z \mathbf{x}_t + \mathbf{b}_z
$$
Here, $( \mathbf{x}_t)$ represents the input features from time phase $t$, $W_z$ is the learnable weight matrix, and $mathbf{b}_z$ is the bias term. The transformed features $mathbf{z}_t$ are then used to compute attention scores.

The attention scores are calculated using the linear attention mechanism. For each position \( i \) in the feature map, the attention score $a_{ij}$ with respect to position $j$ is computed as:
$$
A_{ij} = \mathbf{w}_a^\top \text{ReLU}(\mathbf{W}_{a 1} \mathbf{z}_{i} + \mathbf{W}_{a 2} \mathbf{z}_{j})
$$
Where $\mathbf{w}_a$,  $\mathbf{W}_{a 1}$, and $\mathbf{W}_{a 2}$ are learnable parameters. These attention scores are then normalized using the softmax function to obtain the final attention weights.

Using these attention weights, the features are aggregated to form a unified representation in the BEV space. This allows the model to effectively address the modality misalignment problem and better capture the changes between different time phases.

The BEV Space Converter not only enhances the model's ability to detect changes but also provides a more robust and interpretable feature representation for remote sensing change detection tasks.

### 3.3 Difference Module


### 3.4 Mask Decoder 


## 4 Experiments and Results

## 5 Ablation Studies

## 6 Conclusion

[1] Ting Bai, Le Wang, Dameng Yin, Kaimin Sun, Yepei Chen, Wenzhuo Li, and Deren Li. Deep learning for change detec tion in remote sensing: a review. Geo-spatial Information Science, 26(3):262–288, 2023.
[2] Lazhar Khelifi and Max Mignotte. Deep learning for change detection in remote sensing images: Comprehensive review and meta-analysis. IEEE Access, 8:126385–126400, 2020. 
[3] Bowen Baker, Otkrist Gupta, Nikhil Naik, and Ramesh Raskar. Designing neural network architectures using reinforcement learning. arXiv preprint arXiv:1611.02167, 2016.
[4] Pan Chen, Shenzhou Zhang, Yunpeng Yang, Xiaoqi Liu, and Zhenwei Shi. FCCDN: Feature constraint network for VHR image change detection. ISPRS Journal of Photogrammetry and Remote Sensing, 187:71–84, 2022.
[5] Francesca Bovolo and Lorenzo Bruzzone. A theoretical framework for unsupervised change detection based on change vector analysis in the polar domain. IEEE Transactions on Geoscience and Remote Sensing, 45(1):218–236, 2007.
[6] Francesca Bovolo, Silvia Marchesi, and Lorenzo Bruzzone. A framework for automatic and unsupervised detection of multiple changes in multitemporal images. IEEE Transactions on Geoscience and Remote Sensing, 50(6):2196–2212, 2012.
[7] Hao Chen and Zhenwei Shi. A spatial-temporal attention-based method and a new dataset for remote sensing image change detection. Remote Sensing, 12(10):1662, 2020.
[9] Hao Chen, Zipeng Qi, and Zhenwei Shi. Remote sensing image change detection with transformers. IEEE Transactions on Geoscience and Remote Sensing, 60:1–14, 2021.
[10] Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, et al. Rethinking attention with performers. arXiv preprint arXiv:2009.14794, 2020.
[11] William Fedus, Barret Zoph, and Noam Shazeer. Switch transformer: Scaling to trillion parameter models with simple and efficient sparsity. Journal of Machine Learning Research, 23(120):1–39, 2022.
[12] Keumgang Cha, Junghoon Seo, and Yeji Choi. Vision language models in remote sensing: Current progress and future trends. IEEE Geoscience and Remote Sensing Magazine, 12(2):4–25, 2024.
[13] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. InstructBLIP: Towards general-purpose vision-language models with instruction tuning. arXiv preprint arXiv:2305.06500, 2023.
[14] Ashbindu Singh. Review article digital change detection techniques using remotely-sensed data. International Journal of Remote Sensing, 10(6):989–1003, 1989.
[15] Rodrigo Caye Daudt, Bertrand Le Saux, and Alexandre Boulch. Fully convolutional siamese networks for change detection. In 2018 25th IEEE International Conference on Image Processing (ICIP), pages 4063–4067. IEEE, 2018.
[16] Ashbindu Singh. Review article digital change detection techniques using remotely-sensed data. International Journal of Remote Sensing, 10(6):989–1003, 1989.
[17] Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. In International Conference on Machine Learning, pages 5156–5165. PMLR, 2020.
[18] Jaxon Lee, Qiwen Cui, Licheng Jiao, and Fang Liu. Changemamba: Remote sensing change detection with spatio-temporal state space model. arXiv preprint arXiv:2404.03425, 2024.
[19] Hao Li, Chenglong Li, Shihua Huang, Gaoang Wang, Qian Chen, and Jin Tang. Bev-cd: Bird's eye view change detection for autonomous driving. IEEE Transactions on Intelligent Transportation Systems, 24(7):7391–7402, 2023.
[20] Geoffrey Hinton, Oriol Vinyals, Jeff Dean, et al. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.
[21] Zhenghang Yuan, Lichao Mou, Zhitong Xiong, and Xiao Xiang Zhu. Change detection meets foundation models: A comprehensive survey. arXiv preprint arXiv:2402.12872, 2024.
[22] Gao Huang, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q Weinberger. Densely connected convolutional networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 4700–4708, 2017.
[23] Jingtang Liang, Xinyu Wang, Jian Zhang, and Liangpei Zhang. A deep learning framework for change detection in remote sensing images with noisy labels. Remote Sensing, 12(15):2438, 2020.
[24] Naiyu Fang, Lemeng Wang, Yiming Li, Jie Liu, Rui Ai, and Tao Zhang. BEVHeight: A robust framework for vision-based roadside 3D object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21611–21620, 2023.
[25] Shunping Fang, Kaiyu Li, Jinyuan Shao, and Ziyi Li. SNUNet-CD: A densely connected siamese network for change detection of VHR images. IEEE Geoscience and Remote Sensing Letters, 19:1–5, 2021.
[26] Qingsong Xu, Chaojun Ouyang, Tingbin Zhang, Xiaojuan Li, and Xingmin Meng. A GIS-based probabilistic certainty factor approach for landslide susceptibility assessment in the Zhongshan County, Guangdong Province, China. Catena, 140:113–125, 2016.
[27] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In International Conference on Machine Learning, pages 4904–4916. PMLR, 2021.
[28] Rodrigo Caye Daudt, Bertrand Le Saux, Alexandre Boulch, and Yann Gousseau. Urban change detection for multispectral earth observation using convolutional neural networks. In IGARSS 2018-2018 IEEE International Geoscience and Remote Sensing Symposium, pages 2115–2118. IEEE, 2018.
[29] Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, and Serge Belongie. Feature pyramid networks for object detection. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 2117–2125, 2017.
[30] Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, et al. Rethinking attention with performers. arXiv preprint arXiv: 2009.14794, 2020.
[31] Richard J Radke, Srinivas Andra, Omar Al-Kofahi, and Badrinath Roysam. Image change detection algorithms: a systematic survey. IEEE Transactions on Image Processing, 14(3):294–307, 2005.
[32] Zhiqi Li, Wenhai Wang, Hongyang Li, Enze Xie, Chonghao Sima, Tong Lu, Yu Qiao, and Jifeng Dai. BEVFormer: Learning bird's-eye-view representation from multi-camera images via spatiotemporal transformers. In European Conference on Computer Vision, pages 1–18. Springer, 2022.
[33] Jonah Philion and Sanja Fidler. Lift, splat, shoot: Encoding images from arbitrary camera rigs by implicitly unprojecting to 3d. In European Conference on Computer Vision, pages 194–210. Springer, 2020.
[34] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485, 2023.
[35] Libo Wang, Rui Li, Ce Zhang, Shenghui Fang, Chenxi Duan, Xiaoliang Meng, and Peter M Atkinson. UNetFormer: A UNet-like transformer for efficient semantic segmentation of remote sensing urban scene imagery. ISPRS Journal of Photogrammetry and Remote Sensing, 190:196–214, 2022.
[36] Maoguo Gong, Jia Zhao, Jiao Liu, Qiguang Miao, and Licheng Jiao. Change detection in synthetic aperture radar images based on deep neural networks. IEEE Transactions on Neural Networks and Learning Systems, 27(1):125–138, 2015.
[37] Pol Coppin, Inge Jonckheere, Kris Nackaerts, Bart Muys, and Eric Lambin. Review ArticleDigital change detection methods in ecosystem monitoring: a review. International Journal of Remote Sensing, 25(9):1565–1596, 2004.
[39] Qunming Wang, Xiaohua Tong, and Peter M Atkinson. Hybrid deep learning and machine learning models for crop yield prediction based on multitemporal satellite data. Remote Sensing, 14(4):861, 2022.
[40] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11976–11986, 2022.
[41] Javier Marin, Aritro Biswas, Ferda Ofli, Nicholas Hynes, Amrita Salvador, Yusuf Aytar, Ingmar Weber, and Antonio Torralba. Recipe1m+: A dataset for learning cross-modal embeddings for cooking recipes and food images. IEEE Transactions on Pattern Analysis and Machine Intelligence, 43(1):187–203, 2019.
[42] Lorenzo Bruzzone and Diego Fernàndez Prieto. Automatic analysis of the difference image for unsupervised change detection. IEEE Transactions on Geoscience and Remote Sensing, 38(3):1171–1182, 2000.
[43] Clement Atzberger. Advances in remote sensing of agriculture: Context description, existing operational monitoring systems and major information needs. Remote Sensing, 5(2):949–981, 2013.
[44] OpenAI. GPT-4V(ision) system card. 2023.
[45] Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta, and Yoshua Bengio. FitNets: Hints for thin deep nets. arXiv preprint arXiv:1412.6550, 2014.
[46] Yuanyuan Chen, Zhen Li, Caiyan Chen, Jinfeng Xu, Yilan Zhang, Jiajun Xu, and Shuanggen Jin. Self-supervised learning for few-shot remote-sensing scene classification. Remote Sensing, 13(11):2090, 2021.
[47] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual representations from natural language supervision. In International Conference on Machine Learning, pages 8748–8761. PMLR, 2021.
[48] Gang Li, Xin Tang, Feng Zhang, Jingjing Ma, Haiyan Guan, and Deren Li. A comprehensive survey on 3D semantic segmentation. arXiv preprint arXiv:2006.06080, 2020.
[49] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv: 1701.06538, 2017.
[50] Xiaodong Zhang, Xiaokang Han, Chenglong Li, Xiaoliang Tang, Hao Zhou, and Liang Jiao. Aerial-CD: A large-scale aerial image change detection dataset and benchmark. arXiv preprint arXiv:2306.05742, 2023.
[51] Shunping Ji, Shiqing Wei, and Meng Lu. Fully convolutional networks for multisource building extraction from an open aerial and satellite imagery data set. IEEE Transactions on Geoscience and Remote Sensing, 57(1):574–586, 2018.
[52] Wele Gedara Chaminda Bandara and Vishal M Patel. A transformer-based siamese network for change detection. In IGARSS 2022-2022 IEEE International Geoscience and Remote Sensing Symposium, pages 207–210. IEEE, 2022.
[54] Amanpreet Singh, Ronghang Hu, Vedanuj Goswami, Guillaume Couairon, Wojciech Galuba, Marcus Rohrbach, and Douwe Kiela. FLAVA: A foundational language and vision alignment model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15638–15650, 2022.
[56] Brady Zhou and Philipp Krähenbühl. Cross-view transformers for real-time map-view semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13760–13769, 2022.
[57] Sinong Wang, Belinda Z Li, Madian Khabsa, Han Fang, and Hao Ma. Linformer: Self-attention with linear complexity. arXiv preprint arXiv:2006.04768, 2020.
[58] Gencer Sumbul, Jian Kang, Tristan Kreuziger, Frauke Albrecht, Corneliu Octavian Dumitru, and Mihai Datcu. Multimodal deep learning for earth observation. IEEE Geoscience and Remote Sensing Magazine, 10(3):262–285, 2022.
[59] Mingxing Tan and Quoc Le. Efficientnet: Rethinking model scaling for convolutional neural networks. In International Conference on Machine Learning, pages 6105–6114. PMLR, 2019.
[60] Barret Zoph and Quoc V Le. Neural architecture search with reinforcement learning. arXiv preprint arXiv: 1611.01578, 2016.
[61] Zhe Zhu and Curtis E Woodcock. Continuous change detection and classification of land cover using all available Landsat data. Remote Sensing of Environment, 144:152–171, 2014.
[62] Licheng Jiao, Maoguo Gong, Shuyuan Yang, Fang Liu, Wenping Ma, Lingling Li, and Biao Hou. Change detection in SAR images based on multiscale capsule network. IEEE Geoscience and Remote Sensing Letters, 18(3):484–488, 2020.
[63] Sicong Liu, Lorenzo Bruzzone, Francesca Bovolo, and Peijun Du. Sequential spectral change vector analysis for iteratively discovering and detecting multiple changes in hyperspectral images. IEEE Transactions on Geoscience and Remote Sensing, 53(8):4363–4378, 2015.
[64] Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, and Serge Belongie. Feature pyramid networks for object detection. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 2117–2125, 2017.
[67] Chao Peng, Xiangyu Zhang, Gang Yu, Guiming Luo, and Jian Sun. Large kernel matters—improve semantic segmentation by global convolutional network. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 4353–4361, 2017.
[68] Yongchao Feng, Jiawei Jiang, Heng-Chao Li, Qian Du, and Antonio Plaza. DTCDSCN: Deep twin change detection siamese convolutional network for remote sensing images. IEEE Transactions on Geoscience and Remote Sensing, 59(9):7703–7719, 2020.
[69] Devis Tuia, Francesca Bovolo, and Gustau Camps-Valls. Multitemporal remote sensing image analysis. In Image Processing and Analysis with Graphs: Theory and Practice, pages 399–433. CRC Press, 2012.
[70] Chenxiao Zhang, Peng Yue, Deodato Tapete, Liangcun Jiang, Boyi Shangguan, Li Huang, and Guobiao Liu. A deeply supervised image fusion network for change detection in high resolution bi-temporal remote sensing images. ISPRS Journal of Photogrammetry and Remote Sensing, 166:183–200, 2020.