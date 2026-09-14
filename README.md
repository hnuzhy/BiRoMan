# BiRoMan
The codebase for developing modular Bimanual Robotic Manipulation using dual-arm platforms. These platforms have been utilized in our bimanual manipulation projects including [VLBiMan](https://hnuzhy.github.io/projects/VLBiMan), [BiDemoSyn](https://hnuzhy.github.io/projects/BiDemoSyn), [VLBiMan++](https://hnuzhy.github.io/projects/VLBiManPlus).

<table>
  <tr>
    <td align="center" width=58.5%><img src="./DualArmAubo/dual-arm-aubo.jpg" width=100%></td>
    <td align="center" width=41.5%><img src="./DualArmRokae/dual-arm-rokae.jpg" width=100%></td>
  </tr>
</table>

## ● Platform 1: [DualArmAubo](./DualArmAubo)

This is a platform with a contralateral dual-arm manipulator, a Kingfisher R-6000 binocular camera and two DH PGI-80-80 parallel grippers.


## ● Platform 2: [DualArmRokae](./DualArmRokae)

This is a platform with a semi-humanoid dual-arm manipulator, a Kingfisher R-6000 binocular camera and two Jodell Robotics RG75-30 parallel grippers.

## ● Acknowledgement
We acknowledge the providers of various hardware used in this project, including the [Aubo-i5 robotic arm](https://www.aubo-cobot.com/public/i5product3), [Rokae xMate CR7 robotic arm](https://www.rokae.com/en/product/show/545/xMateCR.html), [DH gripper PGI-80-80](https://en.dh-robotics.com/product/pgi), [Jodell Robotics RG75-300](https://www.jodell-robotics.com/product-detail?id=5), and [kingfisher binocular camera](https://docs.dexforce.com/en/PickWiz/V1.8.3/w138l9kf/). 

## ● Citation
If you use our code in your research, please cite with:
```
% VLBiMan++ (arxiv2026.09)
@article{zhou2026vlbiman++,
  title={VLBiMan++: Expanding the Generalization Boundary of Vision-Language Anchored One-Shot Bimanual Manipulation},
  author={Zhou, Huayi and Gao, Wei and Han, Yiyang and Jia, Kui and Huang, Hui},
  journal={arXiv preprint arXiv:2609.xxxxx},
  year={2026}
}

% VLBiMan (ICLR2026)(arxiv2025.09)
@inproceedings{zhou2026vlbiman,
  title={VLBiMan: Vision-Language Anchored One-Shot Demonstration Enables Generalizable Bimanual Robotic Manipulation},
  author={Zhou, Huayi and Jia, Kui},
  booktitle={International Conference on Learning Representations (ICLR)},
  volume={2026},
  pages={112330--112360},
  year={2026}
}

% BiDemoSyn (RSS2026)(arxiv2025.12)
@inproceedings{zhou2026one,
  title={One-Shot Real-World Demonstration Synthesis for Scalable Bimanual Manipulation},
  author={Zhou, Huayi and Jia, Kui},
  booktitle={Proceedings of Robotics: Science and Systems (RSS)},
  year={2026},
  address={Sydney, Australia}, 
  month={July}, 
  doi={10.15607/RSS.2026.XXII.001} 
}
```
