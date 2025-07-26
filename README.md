# Fire_Survey
# Vision-Based Fire Management System Using Autonomous Unmanned Aerial Vehicles: A Comprehensive Survey
This repository systematically surveys fire management studies, covering various tasks such as classification, detection, and segmentation. It serves as a platform to record and track recent research work in fire management.
# 1 Paper Link
The complete paper and data will be available after publication.
# Fire behaviors, factors, impacts, and solutions using UAVs
| **Behaviours**       | **Factor**         | **Challenges**                         | **Impacts**                       | **UAV-based Solution**         | **Advantages**                  |
|----------------------|--------------------|----------------------------------------|----------------------------------|-------------------------------|----------------------------------|
| Combustion           | Fuel               | Rapid Fire Spread                      | Air Quality Degradation          | Aerial Surveillance           | Rapid Deployment                |
| Combustion           | Fuel               | Rapid Fire Spread                      | Air Quality Degradation          | Aerial Surveillance           | Rapid Deployment                |
| Release Heat         | Oxygen             | Limited Resources                      | Environmental Destruction        | Early Detection               | Enhanced Safety                 |
| Flame Formation      | Heat               | Terrain and Accessibility              | Soil Degradation                 | Firefighting Support          | Cost-Effectiveness              |
| Spread               | Ignition Source    | Urban Interface                        | Water Quality Impacts            | Environmental Monitoring      | Flexibility and Versatility     |
| Smoke                | Weather Conditions | Evacuations and Sheltering             | Carbon Emissions                 | Safety and Risk Assessment    | High Accuracy                   |
| Size and Intensity   | Topography         | Health and Safety Risks                | Economic Losses                  | Communication and Coordination | Access to Remote Areas          |
| ~                    | Human Activities   | Climate Change                         | Displacement of Communities      | Search and Rescue             | Aerial Surveillance             |
| ~                    | ~                  | Public Awareness and Education         | Human Health Impacts             | ~                             | Minimize Human Risk             |
| ~                    | ~                  | Post-Fire Recovery and Rehabilitation  | Psychological and Social Impacts | ~                             | Environmental Monitoring        |
| ~                    | ~                  | ~                                      | Long-term Recovery Challenges    | ~                             | High-Resolution Imaging         |



# Comparative Analysis of UAV‑Based Fire Management Survey Papers: Key Strengths and Limitations.
| **Reference** | **Strength** | **Limitation** |
|--------------|--------------|----------------|
| [41] | • Aboard sensor tool, fire perception methods, and coordination approaches tailored to specific applications.<br>• Introduced recent frameworks for UGVs. | • Shortage of detailed facts about the DL techniques for fire detection.<br>• Lack of dataset information and focus only on forest fires. |
| [42] | • Discusses UAV obstacle detection, search and rescue, and types of UAVs.<br>• Discusses TML methods and general DL and AI-based detection of fire. | • Absence of detailed data about the DL techniques for fire detection.<br>• Few fire datasets and classification techniques mentioned. |
| [44] | • Covers flame and smoke detection using optical remote sensing.<br>• Discusses TML and DL-based detection. | • Focuses only on detection methods; lacks segmentation, classification, and advanced DL techniques.<br>• Lacks UAVs and dataset information. |
| [45] | • Focused on forest fire detection, prevention, and firefighting. | • Missing DL-based methods, datasets, and segmentation techniques. |
| [46] | • Classification of UAVs, cameras, models, weights.<br>• Datasets, frameworks, AI-based hardware and software techniques. | • Basic overview only; lacks full dataset/image references.<br>• Brief mention of AI-based methods. |
| [47] | • UAV-based detection, monitoring, fire combat.<br>• Vision-based fire detection. | • Lacks modern DL/RL methods and updated techniques.<br>• Lacks technical details. |
| [48] | • UAV technologies, sensors, image segmentation and ML methods. | • No UAV model/classification info.<br>• Omits recent DL-based computer vision techniques. |
| [49] | • Drone tech for detection, monitoring, firefighting.<br>• Fire behavior and diagnosis. | • Lacks recent DL updates and datasets.<br>• Focused only on forest fires. |
| [50] | • Remote sensing, deep learning models.<br>• Dataset descriptions and evaluation metrics. | • Excludes UAV model characteristics.<br>• Few datasets; lacks RL methods. |
| [51] | • Fire and victim localization, health assessment, fire behavior. | • Does not explore DL for classification/segmentation.<br>• Few datasets.<br>• Focused on forest fires. |
| **Our Paper** | • UAV types and their role in fire management.<br>• Latest DL/computer vision techniques.<br>• Dataset types and multimodal data (RGB, thermal, IR).<br>• Challenges and case studies. | • Limited to vision-based methods.<br>• Lacks battery/bandwidth analysis and deployment costs.<br>• ML techniques not covered. |


# Existing survey paper summary of targeted area and application domain. 
| **Ref**              | **Det** | **Cls** | **Seg** | **Forest** | **Urban** | **Rural** | **Vehicle** | **CNN** | **Attn** | **ViT** | **RNN** | **GAN** | **YOLO** | **G.DL** | **ML** |
|----------------------|--------|--------|--------|----------|--------|--------|---------|------|--------|------|------|------|-------|--------|------|
| [41]        | ✅     | ❌     | ✅     | ✅       | ❌     | ❌     | ❌      | ✅   | ❌     | ❌   | ❌   | ❌   | ❌    | ✅     | ❌   |
| [42]      | ✅     | ✅     | ❌     | ✅       | ❌     | ❌     | ❌      | ✅   | ❌     | ❌   | ❌   | ❌   | ❌    | ✅     | ✅   |
| [44]     | ✅     | ❌     | ❌     | ✅       | ❌     | ❌     | ❌      | ✅   | ✅     | ❌   | ❌   | ✅   | ✅    | ❌     | ✅   |
| [45]          | ✅     | ❌     | ❌     | ✅       | ❌     | ❌     | ❌      | ❌   | ❌     | ❌   | ❌   | ❌   | ❌    | ✅     | ❌   |
| [46]     | ✅     | ❌     | ❌     | ✅       | ❌     | ❌     | ❌      | ✅   | ❌     | ❌   | ❌   | ✅   | ✅    | ✅     | ❌   |
| [47]           | ✅     | ✅     | ✅     | ✅       | ❌     | ❌     | ❌      | ❌   | ❌     | ❌   | ❌   | ❌   | ❌    | ❌     | ✅   |
| [48]              | ✅     | ✅     | ✅     | ✅       | ❌     | ❌     | ❌      | ❌   | ❌     | ❌   | ❌   | ❌   | ❌    | ❌     | ✅   |
| [49]         | ✅     | ❌     | ❌     | ✅       | ❌     | ❌     | ❌      | ✅   | ❌     | ❌   | ❌   | ❌   | ❌    | ✅     | ✅   |
| [50]     | ✅     | ✅     | ✅     | ✅       | ❌     | ❌     | ❌      | ✅   | ❌     | ❌   | ❌   | ❌   | ✅    | ❌     | ✅   |
| [51]           | ✅     | ❌     | ❌     | ✅       | ❌     | ❌     | ❌      | ✅   | ❌     | ❌   | ✅   | ❌   | ❌    | ✅     | ✅   |
| **Our Paper**        | ✅     | ✅     | ✅     | ✅       | ✅     | ✅     | ✅      | ✅   | ✅     | ✅   | ✅   | ✅   | ✅    | ✅     | ❌   |

**Note:**  
Det = *Detection*, Cls = *Classification*, Seg = *Segmentation*, CNN = *Convolutional Neural Network*, RNN = *Recurrent Neural Network*, YOLO = *You Only Look Once*, ViT = *Vision Transformer*, GAN = *Generative Adversarial Network*, G.DL = *General Deep Learning*, ML = *Machine Learning*.


# Existing survey paper summary of dataset types.
| **Reference**            | **RGB** | **Thermal** | **IR**  |
|--------------------------|--------|------------|--------|
| [41]            | ✅     | ❌         | ✅     |
| [42]          | ✅     | ❌         | ✅     |
| [44]          | ✅     | ❌         | ✅     |
| [45]             | ❌     | ❌         | ✅     |
| [46]         | ✅     | ❌         | ❌     |
| [47]                | ✅     | ❌         | ✅     |
| [48]                 | ❌     | ❌         | ❌     |
| [49]             | ❌     | ❌         | ❌     |
| [50]         | ✅     | ❌         | ❌     |
| [51]              | ✅     | ❌         | ❌     |
| **Our Paper**            | ✅     | ✅         | ✅     |



# Overview of Deep Learning Models for UAV-Based Fire Detection, Classification, and Segmentation
| Ref  | Method                                    | Categories     | Ref  | Method                               | Categories        |
|-------|-------------------------------------------|----------------|-------|-------------------------------------|-------------------|
| [57]  | VGG16, VGG19, InceptionV3                 | CNN            | [68]  | Intermediate Fusion VGG16 ECP-LEACH| CNN               |
| [58]  | FFireNet                                  | CNN            | [60]  | EfficientNetB7-ACNet                | CNN               |
| [59]  | Reduce-VGGNet                             | CNN            | [61]  | DCNN FireNet                       | CNN               |
| [62]  | X-MobileNet                              | CNN            | [74]  | UAV-Net                           | CNN               |
| [63]  | LwF-Inception-V3                         | CNN            | [54]  | CNN                              | CNN               |
| [64]  | BCN-MobileNet-V2                         | CNN            | [65]  | FireXnet                         | CNN               |
| [66]  | RBFN-AISR                               | CNN            | [67]  | SegNet                          | CNN               |
| [80]  | FNU-LSTM                                | RNN            | [81]  | RLSTM-NN                       | RNN               |
| [82]  | RNN                                     | RNN            | [83]  | CNN-RCNN                       | RNN               |
| [89]  | CNN, RNN                               | RNN            | [84]  | ABi-LSTM                      | RNN               |
| [85]  | SA-EX-LSTM                            | RNN            | [165] | RNN-WO                         | RNN               |
| [79]  | DIFFDC-MDL                          | RNN            | [166] | FSA                           | RNN               |
| [93]  | ADE-Net                                | Attention      | [60]  | ACNet                         | Attention         |
| [94]  | DMFA-Fire                             | Attention      | [95]  | Lightweight Model Attention Base CNN | Attention         |
| [96]  | UAV-FDN                              | Attention      | [97]  | P-DenseNet-A-TL               | Attention         |
| [99]  | FuF-Det                              | Attention      | [100] | BranTNet                     | Attention         |
| [98]  | YOLOV5, Spatial attention, GTP      | Attention YOLO | [91]  | LUFFD-YOLO                   | Attention, YOLO   |
| [106] | TransUNet-R50-ViT                   | ViT            | [110] | FWSRNet                      | ViT               |
| [112] | ViTM                               | ViT            | [113] | TransUNet, MedT              | ViT               |
| [114] | FireViTNet                        | ViT, CNN       | [115] | CT-Fire                      | ViT, CNN          |
| [119] | FireFormer                      | ViT, CNN       | [121] | Swin Transformer             | ViT               |
| [116] | STPMSAHI                        | ViT, CNN       | [120] | Deeplabv3                   | ViT, CNN          |
| [111] | FFS-UNet                         | ViT            | [117] | Lightweight ViT, CNN         | ViT, CNN          |
| [119] | FireFormer                      | ViT, CNN       | [118] | Modified VIT                 | ViT               |
| [141] | FL-YOLOv7                       | YOLO           | [142] | YOLO                        | YOLO              |
| [167] | YOLO-CSQ                       | YOLO           | [128] | YOLOv8, CNN-RCNN            | YOLO              |
| [140] | YOLOv5                        | YOLO           | [147] | FSDF                        | YOLO              |
| [130] | YOLOV8, LSTM                  | YOLO, RNN      | [131] | YOLO3                       | YOLO              |
| [132] | YOLOV8                       | YOLO           | [133] | FFYOLO                      | YOLO              |
| [134] | YOLOv8s                      | YOLO           | [135] | Mask R-CNN and YOLO V5,7,8  | YOLO              |
| [136] | Yolov5                       | YOLO           | [139] | YOLOV4, YOLOV5, YOLOV7, YOLOV8, and Faster RCNN | YOLO |
| [143] | YOLO                         | YOLO           | [137] | Yolo-Edge                   | YOLO              |
| [138] | YOLOv3                       | YOLO           | [152] | FireDM                      | GAN               |
| [153] | Generative AI                | GAN            | [154] | FIRe-GAN                    | GAN               |
| [159] | GAN                         | GAN            | [149] | CycleGAN                    | GAN               |
| [162] | MGANs                       | GAN            | [155] | ACGAN                       | GAN               |
| [150] | GAN                         | GAN            | [156] | IC-GAN                      | GAN               |
| [157] | FGL-GAN                     | GAN            | [158] | GAN                         | GAN               |
| [160] | AttentionGAN                | GAN            | [168] | NDGANs                      | GAN               |


# Detection, Classification and Segmentation Datasets

# Detection, Classification and Segmentation Datasets

| Ref   | Dataset Name                             | Dataset Types       | Total  |
|-------|----------------------------------------|---------------------|--------|
| [185] | [Burned Area UAV dataset](https://zenodo.org/records/7944963#.ZGYP6nbMIQ8)                    | RGB                 | 22500  |
| [179] | [FLAME](https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs)                         | RGB & IR            | 47992  |
| [180] | [FLAME2](https://ieee-dataport.org/open-access/flame-2-fire-detection-and-modeling-aerial-multi-spectral-image-dataset)                       | RGB & IR            | 59451  |
| [181] | [FLAME3](https://ieee-dataport.org/open-access/flame-2-fire-detection-and-modeling-aerial-multi-spectral-image-dataset)                       | RGB & IR            | 739    |
| [186] | [AIDER](https://www.kaggle.com/datasets/maryamsana/yolov5emergencyresponse)                                            | RGB                 | 1000   |
| [187] | [DC Lab fire dataset](https://www.kaggle.com/datasets/dataclusterlabs/fire-and-smoke-dataset)                              | RGB                 | 10210  |
| [188] | [Wildfire dataset](https://data.mendeley.com/datasets/gjmr63rz2r/1)                                                  | RGB                 | 1900   |
| [184] | [FireNet](https://github.com/OlafenwaMoses/FireNET?tab=readme-ov-file)                                                  | RGB                 | 502    |
| [189] | [Fire dataset](https://github.com/jackfrost1411/fire-detection)                                                   | RGB                 | 3225   |
| [190] | [Fire detection dataset](https://www.kaggle.com/datasets/ritupande/fire-detection-from-cctv)                                | RGB                 | 864    |
| [191] | [Furg fire](https://github.com/steffensbola/furg-fire-dataset)                                                    | RGB                 | 36702  |
| [183] | [Mivia’s fire dataset](https://mivia.unisa.it/datasets/)                                                       | RGB                 | 62690  |
| [192] | [Firefly](https://github.com/ERGOWHO/Firefly2.0?tab=readme-ov-file)                                                 | RGB                 | 19273  |
| [193] | [Domestic fire dataset](https://github.com/datacluster-labs/Domestic-Fire-and-Smoke-Dataset/tree/main)                          | RGB                 | 5000   |
| [194] | [Fire dataset](https://www.kaggle.com/datasets/phylake1337/fire-dataset)                                         | RGB                 | 999    |
| [195] | [DFire dataset](https://github.com/gaiasd/DFireDataset)                                                    | RGB                 | 21000  |
| [196] | [Fire Smoke and Human Detector Dataset](https://universe.roboflow.com/spyrobot/fire-smoke-and-human-detector)                | RGB                 | 9749   |
| [197] | [Fire-Detection-Polygon](https://universe.roboflow.com/fire-detection-polygon/fire-detection-polygon/model/17)                  | RGB                 | 3010   |
| [198] | [Activate Fire](https://github.com/pereira-gha/activefire)                                                   | Thermal             | 146,214|
| [199] | [UAV Thermal Imaginary](https://www.kaggle.com/datasets/adiyeceran/uav-thermal-imaginary-fire-dataset/data)                   | Thermal             | 3980   |
| [200] | [UAVs-FFDB](https://data.mendeley.com/datasets/5m98kvdkyt/2)                                                  | RGB                 | 15560  |
| [201] | [UAVs-Fire dataset](https://github.com/LeadingIndiaAI/Forest-Fire-Detection-through-UAV-imagery-using-CNNs/tree/master/data)              | RGB                 | 2096   |
| [182] | [D-Fire](https://github.com/gaiasd/DFireDataset?tab=readme-ov-file)                                          | RGB                 | 21,000 |
| [140] | [FASDD UAV](https://essd.copernicus.org/preprints/essd-2022-394/)                                         | RGB                 | 53530  |
| [202] | [Corsican dataset](http://cfdb.univ-corse.fr/index.php?menu=1)                                           | RGB                 | -      |
| [91]  | [M4SFWD](https://github.com/Philharmy-Wang/M4SFWD) (RGB, [GAN](https://ieee-dataport.org/documents/multiple-scenarios-multiple-weather-conditions-multiple-lighting-conditions-and-multiple)) | RGB, GAN             | 35,526 |
| [148] | [MTBS data](https://ieee-dataport.org/documents/large-scale-burn-severity-mapping-multispectral-imagery-using-deep-semantic-segmentation) | RGB-IR              | 6000   |



