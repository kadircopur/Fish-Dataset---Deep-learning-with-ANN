# Fish Image Classification Project

This project focuses on classifying fish species from images using deep learning techniques. The project includes image preprocessing, model design, and optimization, along with specific parameters and configurations used to enhance the classification performance.

## Preprocessing

Several steps were applied to prepare the image data before feeding it into the model, ensuring efficient and accurate processing:

### 1. Image Resizing
All images were resized to **192x192** pixels. This was done primarily for performance reasons, as smaller image sizes reduce computational load while still maintaining the necessary features for classification.

### 2. Adaptive Thresholding
To enhance the visibility of key features in the images, **adaptive thresholding** was applied. This process helped emphasize edges and important patterns in the fish images, improving the model's ability to learn distinctive features.

### 3. Normalization
After experimenting with various normalization ranges, the pixel values were normalized to the range **-0.32 to 0.32**. This range was chosen as it provided the best separation of image features, leading to better model performance during training.

### 4. Threading for Preprocessing
Since preprocessing (such as resizing and thresholding) is computationally expensive, **threading** was implemented to speed up these tasks. Without threading, preprocessing would have taken significantly more time, which is a critical consideration when dealing with large datasets.

### 5. Label Encoding
The fish species labels were numerically encoded into classes (e.g., 1, 2, 3) to prepare them for the model's output layer. This numeric format is essential for efficient training and prediction.

### 6. Train-Test Split
To ensure sufficient data for training, the dataset was split into **85% training** and **15% test** data. This larger training set helped improve the model’s performance by exposing it to more examples during the learning phase.

## Model Architecture

The neural network was designed with the following components:

### 1. Input Layer
- Takes in **192x192 pixel** images with three color channels (RGB).

### 2. Hidden Layers
- The model consists of **3 intermediate layers**, which progressively extract and downsample features from the images. These layers help the model learn from the input data and recognize patterns unique to each fish species.

### 3. Output Layer
- The output layer classifies the fish into different species using a **softmax activation function**, which provides a probability distribution over all possible classes.

## Model Parameters

The following parameters were used during the model training:

### 1. Optimizer
The **Adam optimizer** was used to train the model. Adam is widely used in deep learning because it combines the benefits of two other optimizers—AdaGrad and RMSProp—and provides adaptive learning rates for each parameter during training.

- **Optimizer**: `Adam`
- **Learning Rate**: `0.001`

### 2. Loss Function
The **categorical cross-entropy** loss function was used since this is a multi-class classification problem. Categorical cross-entropy calculates the difference between the true label and predicted probabilities and is commonly used in classification tasks.

- **Loss Function**: `Categorical Cross-Entropy`

### 3. Batch Size and Epochs
A batch size of **32** was used, which balances memory consumption and model training time. The model was trained for **50 epochs**. Early stopping was implemented to halt the training process if the validation loss stopped improving after several epochs.

- **Batch Size**: `32`
- **Epochs**: `50`
- **Early Stopping**: Enabled (with patience set to 5 epochs)

### 4. Learning Rate Scheduler
A **learning rate scheduler** was used to adjust the learning rate during training. As the model approaches a solution, the learning rate was decreased to allow for finer weight updates, helping the model converge more effectively.

---

## Training Strategy

### 1. Early Stopping
To prevent overfitting, **early stopping** was implemented during training. The training process was monitored, and it was halted if the validation loss stopped improving after a set number of epochs (patience of 5 epochs). This ensures that the model does not overfit the training data and maintains better generalization to unseen data.

### 2. Model Regularization
To further mitigate overfitting, regularization techniques such as dropout were considered. Although not applied in this version, future improvements can integrate such techniques.

---

## Key Technical Details

1. **Image Resizing**: 192x192 pixels for performance optimization.
2. **Adaptive Threshold**: Applied to enhance important features in the image.
3. **Normalization**: Pixel values normalized to the range [-0.32, 0.32] after testing different ranges.
4. **Threading**: Used to speed up preprocessing tasks and reduce processing time.
5. **Train-Test Split**: 85% training, 15% testing to improve generalization.
6. **Model Layers**: 3 intermediate layers to extract features.
7. **Optimizer**: Adam with learning rate of 0.001.
8. **Loss Function**: Categorical Cross-Entropy.
9. **Batch Size**: 32.
10. **Overfitting Prevention**: Early stopping enabled.

---

## How to Run the Project

1. Clone the repository and navigate to the project directory:
   ```bash
   git clone <repository-url>
   cd fish-image-classification
   ```

2. Run the Jupyter Notebook:
   ```bash
   jupyter notebook fish_classification.ipynb
   ```

Follow the steps in the notebook to preprocess the data, train the model, and evaluate the results.

---

## Conclusion

This project demonstrates the use of deep learning to classify fish species from images. Key techniques such as adaptive thresholding, image normalization, and threading were used to preprocess the data, while parameters such as the Adam optimizer and early stopping were applied to optimize the training process.

By focusing on optimizing preprocessing and model design, the classification task was efficiently handled, resulting in a model capable of distinguishing between different fish species.

---
