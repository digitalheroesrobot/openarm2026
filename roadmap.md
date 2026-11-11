          META QUEST 3
                │
                ▼
           DORA WebXR
                │
                ▼
       DORA IK (V1 adaptation)
                │
                ▼
        OpenArm V1 MuJoCo
                │
                ▼
       Physical Cup Grasping
                │
                ▼
     OpenArmDataset Recorder
                │
                ▼
          LeRobot V3.0
                │
                ▼
            ACT Training


            | Phase | Development | Expected result |
|---|---|---|
| 0 | Architecture and repository audit | Confirm software compatibility |
| 1 | OpenArm V1 MuJoCo environment | Stable simulation with cup |
| 2 | Scripted cup pickup | Robot physically grasps and lifts cup |
| 3 | DORA WebXR setup | Quest 3 communicates with Ubuntu |
| 4 | DORA IK adaptation | VR controller moves OpenArm V1 |
| 5 | VR cup pickup | Complete pickup using Quest 3 |
| 6 | OpenArmDataset recording | Collect demonstrations |
| 7 | LeRobot conversion | Prepare training dataset |
| 8 | ACT training | Train autonomous cup-picking policy |