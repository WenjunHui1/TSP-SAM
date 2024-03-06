# TSP-SAM

# Endow SAM with Keen Eyes: Temporal-spatial Prompt Learning for Video Camouflaged Object Detection (CVPR 2024)

Wenjun Hui, Zhenfeng Zhu, Shuai Zheng, Yao Zhao

[fig1_illustration.pdf](https://github.com/WenjunHui1/TSP-SAM/files/14503988/fig1_illustration.pdf)

The Segment Anything Model (SAM), a prompt-driven foundational model, has demonstrated remarkable performance in natural image segmentation. However, its application in video camouflaged object detection (VCOD) encounters challenges, chiefly stemming from the overlooked temporal-spatial associations and the unreliability of user-provided prompts for camouflaged objects that are difficult to discern with the naked eye. To tackle the above issues, we endow SAM with keen eyes and propose the \underline{T}emporal-\underline{s}patial \underline{P}rompt SAM (TSP-SAM), a novel approach tailored for VCOD via an ingenious prompted learning scheme.
Firstly, motion-driven self-prompt learning is employed to capture the camouflaged object, thereby bypassing the need for user-provided prompts. With the detected subtle motion cues across consecutive video frames, the overall movement of the camouflaged object is captured for more precise spatial localization.
Subsequently, to eliminate the prompt bias resulting from inter-frame discontinuities, the long-range consistency within the video sequences is taken into account to promote the robustness of the self-prompts. 
It is also injected into the encoder of SAM to enhance the representational capabilities. Extensive experimental results on two benchmarks demonstrate that the proposed TSP-SAM achieves a significant improvement over the state-of-the-art methods. With the mIoU metric increasing by \textbf{7.8\%} and \textbf{9.6\%}, TSP-SAM emerges as a groundbreaking step forward in the field of VCOD.

# Method

[fig2_framework.pdf](https://github.com/WenjunHui1/TSP-SAM/files/14503991/fig2_framework.pdf)

This is the official PyTorch implementation of our work "Endow SAM with Keen Eyes: Temporal-spatial Prompt Learning for Video Camouflaged Object Detection" accepted at CVPR 2024.

_Code will be available soon._
