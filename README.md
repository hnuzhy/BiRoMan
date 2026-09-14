# BiRoMan
The codebase for developing modular Bimanual Robotic Manipulation using dual-arm platforms. These platforms have been utilized in our bimanual manipulation projects including [VLBiMan](https://hnuzhy.github.io/projects/VLBiMan), [BiDemoSyn](https://hnuzhy.github.io/projects/BiDemoSyn), and [VLBiMan++](https://hnuzhy.github.io/projects/VLBiManPlus).

⚠︎ *Please note that these modular solutions are tightly coupled with the specific hardware (e.g., robotic arms, parallel grippers and visual cameras). You have to build your own robot manipulation system from scratch.*

<table>
  <tr>
    <td align="center" width=58.5%> Dual-Arm Aubo-i5 </td>
    <td align="center" width=41.5%> Dual-Arm Rokae xMate CR7 </td>
  </tr>
  <tr>
    <td align="center" width=58.5%><img src="./DualArmAubo/dual-arm-aubo.jpg" width=100%></td>
    <td align="center" width=41.5%><img src="./DualArmRokae/dual-arm-rokae.jpg" width=100%></td>
  </tr>
</table>


## ● Platform 1: [DualArmAubo](./DualArmAubo)

**This is a platform with a contralateral fixed-base dual-arm manipulator, a Kingfisher R-6000 binocular camera and two DH-Robotics PGI-80-80 parallel grippers.** Specifically, the platform comprises a rectangular workspace (110cm×70cm) with two 6-DoF Aubo-i5 collaborative arms (880mm reach) mounted on opposite short edges of the table. This opposing-arm setting maximizes shared workspace while minimizing self-collision risks, albeit differing from anthropomorphic designs. Each arm is equipped with a gripper (80mm max opening, 50mm effective length), controlled in binary states (open/closed). Tool length compensation accounts for 160mm absolute length of the gripper. The scene perception is provided by a binocular stereo camera (960×540 RGB resolution), mounted 100cm above the table long edge to capture a third-person view. The calibrated stereo setup reconstructs high-fidelity 3D point clouds using [IGEV](https://github.com/gangweix/igev) or [FoundationStereo](https://github.com/NVlabs/FoundationStereo), eliminating the need for wrist-mounted cameras while ensuring full task visibility.Consequently, we do not employ eye-in-hand cameras at the robot end-effectors.


## ● Platform 2: [DualArmRokae](./DualArmRokae)

**This is a platform with a semi-humanoid fixed-base dual-arm manipulator, a Kingfisher R-6000 binocular camera and two Jodell Robotics RG75-300 parallel grippers.** This dual-arm robotic platform configured in a popular humanoid style. It consists of two Rokae xMate CR73 6-DoF collaborative arms (988mm reach), each equipped with a gripper with a max opening of 75 mm and absolute length of 210mm. A binocular camera is mounted centrally at the head position. An anthropomorphic dual-arm configuration is clearly better aligned with human operational habits, making teaching easier and movements more human-like. However, its drawbacks include an increased overlapping workspace between the arms and restricted reachability for each individual arm. As for the handling of the grippers, the configuration of the binocular stereo camera, and the point cloud reconstruction methods, they are all similar or identical to those used with the dual-arm Aubo system.


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
