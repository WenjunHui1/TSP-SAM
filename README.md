# Endow SAM with Keen Eyes: Temporal-spatial Prompt Learning for Video Camouflaged Object Detection (CVPR 2024)

Wenjun Hui, Zhenfeng Zhu, Shuai Zheng, Yao Zhao

![fig1_illustration_v3](https://github.com/WenjunHui1/TSP-SAM/assets/103172926/1bad61b9-7eaa-4600-b82e-363fab20a5de)

The Segment Anything Model (SAM), a prompt-driven foundational model, has demonstrated remarkable performance in natural image segmentation. However, its application in video camouflaged object detection (VCOD) encounters challenges, chiefly stemming from the overlooked temporal-spatial associations and the unreliability of user-provided prompts for camouflaged objects that are difficult to discern with the naked eye. To tackle the above issues, we endow SAM with keen eyes and propose the Temporal-spatial Prompt SAM (TSP-SAM), a novel approach tailored for VCOD via an ingenious prompted learning scheme.
Firstly, motion-driven self-prompt learning is employed to capture the camouflaged object, thereby bypassing the need for user-provided prompts. With the detected subtle motion cues across consecutive video frames, the overall movement of the camouflaged object is captured for more precise spatial localization.
Subsequently, to eliminate the prompt bias resulting from inter-frame discontinuities, the long-range consistency within the video sequences is taken into account to promote the robustness of the self-prompts. 
It is also injected into the encoder of SAM to enhance the representational capabilities. Extensive experimental results on two benchmarks demonstrate that the proposed TSP-SAM achieves a significant improvement over the state-of-the-art methods. With the mIoU metric increasing by 7.8% and 9.6%, TSP-SAM emerges as a groundbreaking step forward in the field of VCOD.

# Method

![fig2_framework-v2](https://github.com/WenjunHui1/TSP-SAM/assets/103172926/13409a82-dfac-4855-b133-074a02e64b9a)

(a) Motion-driven self-prompt learning. The motion-driven self-prompt learning is to use the implicit inter-frame motion in the frequency domain to facilitate the spatial identification of the camouflaged object, thereby learning the self-prompts for SAM.

(b) Robust prompt learning based on long-range consistency. To eliminate the prompt bias stemming from underlying inter-frame discontinuities, the long-range temporal-spatial consistency Xc(t) within the video sequences is modeled to promote the robustness of the self-prompts.

(c) Temporal-spatial injection for representation enhancement. To enhance the representational capabilities of SAM, the long-range temporal-spatial consistency Xc(t) is injected into the image embedding Xe(t) of SAM, contributing to more precise detection.

# Results
![image](https://github.com/WenjunHui1/TSP-SAM/assets/103172926/340bd300-8a79-452b-8c31-568a25f64a36)

![image](https://github.com/WenjunHui1/TSP-SAM/assets/103172926/8cd26469-b7d7-453f-8b5b-498e697d4b78)


https://github.com/WenjunHui1/TSP-SAM/assets/103172926/3d47857e-c93f-414b-9857-c06de3fb5f2e


This is the official PyTorch implementation of our work "Endow SAM with Keen Eyes: Temporal-spatial Prompt Learning for Video Camouflaged Object Detection" accepted at CVPR 2024.

_Code will be available soon._
