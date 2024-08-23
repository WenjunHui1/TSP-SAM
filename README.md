# Endow SAM with Keen Eyes: Temporal-spatial Prompt Learning for Video Camouflaged Object Detection (CVPR 2024)

Wenjun Hui, Zhenfeng Zhu, Shuai Zheng, Yao Zhao

```
@InProceedings{hui2024endow,
    author    = {Wenjun Hui, Zhenfeng Zhu, Shuai Zheng, Yao Zhao},
    title     = {Endow SAM with Keen Eyes: Temporal-spatial Prompt Learning for Video Camouflaged Object Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
}
```

# Method

![fig2_framework-v2](https://github.com/WenjunHui1/TSP-SAM/assets/103172926/13409a82-dfac-4855-b133-074a02e64b9a)

## Weights and results
The best checkpoint:

The predicted masks: https://drive.google.com/drive/folders/1nmPpToxGJMfNLB6A6FlsDnIBvNQBZaqV.

## Envs


## Evaluation

```shell
./env/bin/python main.py
```


# Results
![image](https://github.com/WenjunHui1/TSP-SAM/assets/103172926/340bd300-8a79-452b-8c31-568a25f64a36)
Table evaluates TSP-SAM and the baselines on MoCA-Mask and CAD2016 datasets. The conclusion is:
(i) compared to the combination of mask and point prompts, the combination of mask and box prompts is more reliable for SAM. This is attributed to the limitations of the point prompt in conveying the information about the weak boundaries. In contrast, bounding box excels in indicating the boundaries of the camouflaged object, even in the presence of bias. 

(ii) the strong contrast in the performance between TSP-SAM and single-image camouflaged object detection methods points to the importance of temporal-spatial relationships in breaking camouflage. 

(iii) compared with video object segmentation methods, TSP-SAM achieves the best performance. Notably, TSP-SAM exhibits a 1.3% - 8.7% improvement over [5] on MoCA-Mask dataset, meaning that the temporal-spatial exploration in TSP-SAM is more effective compared to that in SLTNet.

## Presentation
https://github.com/WenjunHui1/TSP-SAM/assets/103172926/b3908367-96bf-47c6-9ca5-ea0b67ca827c
