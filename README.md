# WhiteMatterHyperintensities

White matter hyperintensity segmentation using a 2D U-Net. Keras with Tensorflow backend


## Brain extraction:

Using betl


Solarized dark             |  Solarized Ocean          |  Solarized Ocean
:-------------------------:|:-------------------------:|:-------------------------:
![](https://...Ocean.png)  |  ![](https://...Dark.png) |  ![](https://...Dark.png)



## Normalization:

All slices where shaped to (240, 240)





## Data augmentation:

| Shifts                                       | Rotations   |
| -------------------------------------------- | ----------- |
| X axis: [-0.3, 0.3]<br />Y axis: [-0.3, 0.3] | [-15º, 15º] |



## T1, FLAIR and FLAIR's top-hat as a 3 channel image input for the network

| T1                        | FLAIR                     | Top-hat                   | Ground truth              |
| ------------------------- | ------------------------- | ------------------------- | ------------------------- |
| ![](https://...Ocean.png) | ![](https://...Ocean.png) | ![](https://...Ocean.png) | ![](https://...Ocean.png) |



## Training:


### Weight initialization

Based on a Gaussian kernel of $ stdv = \frac{2}{\sqrt{N}} $


### Loss function

Dice Similarity Coefficient Loss over N slices where $ N = batch\_size  $

$$ DSCLoss = - \frac{\sum_{n = 1}^{N} | g_{n} \circ \ p_{n} |}{\sum_{n = 1}^{N} |g_{n}| + |p_{n}|} $$




### Optimizer: Adam

### Hyperparameters:

- Batch size: 30
- Learning rate: 0.000001


## Results

Solarized dark             |  Solarized Ocean          |  Solarized Ocean
:-------------------------:|:-------------------------:|:-------------------------:
![](https://...Ocean.png)  |  ![](https://...Dark.png) |  ![](https://...Dark.png)

Solarized dark             |  Solarized Ocean          |  Solarized Ocean
:-------------------------:|:-------------------------:|:-------------------------:
![](https://...Ocean.png)  |  ![](https://...Dark.png) |  ![](https://...Dark.png)
