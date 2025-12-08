# Reccomending Visually Similar Professionally Designed Rooms
![intro_photo](https://github.com/user-attachments/assets/a2de8d7d-bd44-4653-bb7c-b4fbcd50d239)

## Overview
1. Introduction
2. Methodology & Data
3. Findings
4. Discussion
5. Conclusion

## Introduction

Have you ever struggled to decorate an apartment, dorm, or bedroom? Interior decoration requires expertise in layout and color harmony, and whether you don’t have an eye for design, or you’re dealing with a particularly unflattering dorm room setup, it can be a challenge for many. Yet the benefits of decorating your personal space well are clear: research has shown that well-decorated spaces benefit mental and physical well-being by reducing stress and improving mood and even have the potential to boost productivity and creativity. Currently there are few freely available AI room design tools, and of these, I have yet to find one which is open source. For my final project, I aim to fill this gap.

This project investigates whether neural networks can learn the visual structure and stylistic attributes of interior spaces and use that understanding to recommend visually similar professionally designed rooms. By training models to recognize room type and décor style and by generating robust image embeddings, this project aims to create an open-source tool that provides personalized design inspiration based on a user’s photograph, providing inspiration for budget-friendly designs one can achieve with items similar to those they already own and within their preferred decor style.


## Methodology & Data

Convolutional networks are a type of neural network that use convolution in place of general matrix multiplication in at least one of their layers. This makes them good for working with data that has a grid-like topology, such as images, since they preserve spatial structure and allow for a much smaller parameter count.

My project uses two CNNS. The first is a CNN trained on the Houzz interior design dataset, containing over 18,000 professional room photos labeled by design style (e.g., modern, rustic, industrial, etc.). Each image is resized to 224×224 pixels and its pixel values normalized to the range [0, 1]. During training, I apply data augmentation, including random rotations (±20°), horizontal flips, zooms, translations, and brightness adjustments, to increase robustness and prevent overfitting. To reduce confusion between visually similar design styles, I merged the 19 original style classes into 6.

The model uses MobileNetV2 as a base architecture (pretrained on ImageNet) with the last 20 layers unfrozen for fine-tuning. On top of the base model, I add global average pooling, dropout (0.3) for regularization, and a dense softmax output layer equal to the number of style classes. The model is trained for 15 epochs using the Adam optimizer (learning rate = 1e-4) and categorical cross-entropy loss, with both training and validation performance monitored. Because the dataset is moderately imbalanced across styles, I compute class weights proportional to inverse class frequency to reduce bias toward dominant categories. After training, the Style CNN achieved a 49.32% validation accuracy and 1.42 validation loss. Class-wise performance showed strongest F1 scores for modern_contemporary (0.59) and eclectic_other (0.46), and confusion was most common between visually similar categories such as modern_contemporary and industrial.

<img width="989" height="490" alt="room_style_classification" src="https://github.com/user-attachments/assets/122178bb-d430-4228-ab1f-d24ad2532b17" />



The second network is a pretrained ResNet50-Places365 model, trained on over 8 million scene images across 365 categories types of indoor and outdoor spaces. Although I initially considered training a smaller MobileNetV2-based CNN for room type classification, the pretrained ResNet50-Places365 model proved substantially more effective. When evaluated on my six-category validation subset (bedroom, kitchen, bathroom, living room, dining room, and home office) created by auto-labelling the Houzz data using CLIP, the model achieved a 80.61% overall accuracy and a validation loss of 0.58.


To identify the most visually similar professionally designed rooms, the system first runs the pretrained room-type CNN on all images in the Houzz dataset and stores their predicted categories. The Style CNN is then modified by removing its final softmax classification layer, so instead of a classification label it outputs a feature vector. When a user provides a query image, it is first classified by the room-type CNN to ensure comparisons are made only within the same room category. The query image and all matching dataset images are then passed through the Style CNN to generate their embeddings. Finally, the system computes cosine similarity between the query’s style embedding and each candidate’s embedding, ranking results by similarity and returning the top five most visually similar professionally designed interiors.

<img width="960" height="720" alt="query_photo_flow" src="https://github.com/user-attachments/assets/8d496db9-78ea-4916-9201-255c418b3e7a" />





## Findings

To determine how well my model truly captures stylistic similarities between rooms, I created a color histogram similarity baseline. Each image is represented by a 3D RGB histogram with parameters bins=(8, 8, 8), which divides each color channel (Red, Green, Blue) into 8 ranges, resulting in 512 total features (8×8×8). The histogram is normalized so that images of different sizes remain comparable. I extract color histograms for every image in the Houzz dataset and compare the cosine similarity of each histogram against the query image, returning the top 5. This model outputs the most visually similar professionally designed rooms based on color alone.




**Bedroom from testing data**

Room Type Filering + Style Cosine Similarity:
<img width="1182" height="208" alt="model_output_4" src="https://github.com/user-attachments/assets/cb706b57-809f-4284-80d1-cab8005793c2" />

Color histogram:
<img width="1182" height="208" alt="model_output_8" src="https://github.com/user-attachments/assets/a2c28875-aca1-4098-8334-4bd0a6ab169b" />



**Bathroom from testing data**

Room Type Filering + Style Cosine Similarity:
<img width="1182" height="208" alt="model_output_3" src="https://github.com/user-attachments/assets/f8e8e46a-f902-488c-99d4-d8d8b1024582" />

Color histogram:
<img width="1182" height="208" alt="model_output_7" src="https://github.com/user-attachments/assets/48e8c5af-2598-4a5d-9fbc-faa365e14dab" />



**My bedroom**

Room Type Filering + Style Cosine Similarity:
<img width="1182" height="208" alt="model_output_1" src="https://github.com/user-attachments/assets/b7a75035-480d-4216-b8cd-47978ae4da7f" />

Color histogram:
<img width="1182" height="208" alt="model_output_5" src="https://github.com/user-attachments/assets/f77f3938-d200-48a0-9e3a-668ecb558c12" />


**My bathroom**

Room Type Filering + Style Cosine Similarity:
<img width="1182" height="263" alt="model_output_2" src="https://github.com/user-attachments/assets/7d8baac5-99f3-4fd6-af79-c57dbecdb1c8" />

Color histogram:
<img width="1182" height="263" alt="model_output_6" src="https://github.com/user-attachments/assets/9842d31d-02e8-4f1f-918d-3d66ea79df27" />



## Discussion
Overall, my models gave reasonable predictions that could be helpful to someone looking to design a space. They were more relevant than the outputs given by the color histogram. However, one of the main aims of my tool was to give budget friendly design inspiration, but many visually similar images have different flooring, tiling, paint color, or other architectural differences that would be costly to modify. Also, room type miscalculation by the ResNet50-Places365 model creates poor results.

Additionally, I ran into significant difficulties building my style model. My first iteration of the model with 19 classes had a maximum of 27% across different epochs, training rates, and input shapes. The model is also very computationally expensive and takes roughly 30 minutes to run on my macbook. With the merged class data, model performance still remains below what would be expected for simple non-subjective tasks, but results still reflect a meaningful stylistic understanding.

## Conclusion
This project demonstrates the potential of convolutional neural networks to understand and compare interior design images based on both room type and visual style. By combining a pretrained ResNet50-Places365 model for room classification with a custom-trained Style CNN for aesthetic similarity, the system produces design recommendations that are more relevant and visually coherent than a simple color-based method. While overall accuracy of the Style CNN remains modest due to the subjectivity of style and limited training data, the model successfully aligns with human perception of design categories. Future improvements could include expanding the dataset, fine-tuning the pretrained network, and incorporating material segmentation to focus on modifiable design elements, making the tool more practical for budget-conscious home design inspiration.

## Houzz Interior Design Dataset
https://www.kaggle.com/datasets/stepanyarullin/interior-design-styles
