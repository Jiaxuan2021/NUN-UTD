# Detecting Nearshore Underwater Targets with Hyperspectral Nonlinear Unmixing Autoencoder
-----------
> *Jiaxuan Liu, Jiahao Qi, Dehui Zhu, Hao Wen, Hejun Jiang, and Ping Zhong. TGRS.2024.*

Hyperspectral underwater target detection is a promising and challenging task in remote sensing image processing. Existing methods face significant challenges when adapting to real nearshore environments, where cluttered backgrounds hinder the extraction of target signatures and exacerbate signal distortion.Hyperspectral unmixing demonstrates potential effectiveness for nearshore underwater target detection by simultaneously extracting water background endmembers and separating target signals. To this end, this paper investigates a novel nonlinear unmixing network for hyperspectral underwater target detection, denoted as NUN-UTD.

![Framework](/pics/fig_framework.png "NUN-UTD"){width=900px style="display: block; margin: 0 auto;"}

***
### Dataset
Due to the difficulty of deploying underwater targets and the high cost of data collection, research in this area has predominantly relied on simulated data. To advance the study of underwater target detection in real-world scenarios, we collected a dataset of real underwater scenes and conducted experiments on this data. The deployed underwater target is an iron plate, and the target's prior spectral data were collected onshore.

![Framework](/pics/ref.png "reference"){width=400px style="display: block; margin: 0 auto;"}

> The River Scene data sets was captured by Headwall Nano-Hyperspec imaging sensor equipped on DJI Matrice 300 RTK unmanned aerial vehicle, and it was collected at the Qianlu Lake Reservoir in Liuyang (28◦18′40.29′′ N, 113◦21′16.23′′ E), Hunan Province, China on July 31, 2021.

- Dataset format: mat

![Framework](/pics/datasets.png "reference"){width=800px style="display: block; margin: 0 auto;"}

- River Scene1
242×341 pixels with 270 spectral bands

- River Scene2
255 × 261 pixels with 270 spectral bands

- River Scene3
137 × 178 pixels with 270 spectral bands

- Simulated Data
The data set has a spatial resolution of 200 × 200 pixels, with wavelength coverage from 400 to 700 nm at 150 spectral bands.

Keys: 
- data: The hyperspectral imagery contains underwater targets
- target: The target prior spectrum collected on land
- gt: The ground truth of underwater target distribution

----

### Training and Testing

1. Modify `config.py`
2. Run ` python main.py `

> You can use `demo_for_reproducibility.py` to reproduce the results, model weights are in folder *result*.

Training for new dataset need to generate NDWI mask. 
> NDWI Water Mask (require gdal):
> - water -- 0
> - land -- 255
> - selected bands get from envi
> - GREEN.tif: green band 549.1280 nm
> - NIR.tif: near-infrared band 941.3450 nm

Then the code will automatically generate VCA initial weight. 



*There has been limited research in this field, and many challenges remain in applying these methods to real-world scenarios. We sincerely hope that this work contributes positively to the field, despite the theoretical and practical limitations that still exist. If you have any concerns, please do not hesitate to contact liu_jiaxuan2021@163.com.*

