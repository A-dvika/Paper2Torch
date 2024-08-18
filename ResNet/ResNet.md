# Understanding the Problem Before ResNet

## Introduction

![alt text](image.png)

Before ResNet (Residual Networks) was introduced, training deep neural networks was a significant challenge. As researchers tried to make networks deeper to improve their performance, they encountered a major problem known as the **"degradation problem."**

## The Degradation Problem

### What Is It?

The degradation problem occurs when increasing the depth of a neural network results in **higher training error**. This is surprising because, in theory, a deeper network should perform at least as well as a shallower one.

### Why Does This Happen?

The issue isn't about **overfitting** (where a model performs well on training data but poorly on new data). Instead, it's about the difficulty in **optimizing** very deep networks. Here's what happens:

- **Vanishing/Exploding Gradients:** As networks get deeper, the gradients (used to update the network during training) can become very small (vanishing) or very large (exploding). This makes it hard for the network to learn effectively.

- **Training Error Increases:** Even with techniques like special initialization and normalization to help with the vanishing/exploding gradient problem, adding more layers still caused **higher training error**. This means the network struggled to learn as it got deeper.

## Why Was This a Problem?

If deeper networks lead to worse performance, it limits how much we can improve a model by adding more layers. This was a big obstacle because researchers knew that deeper networks should, in theory, be able to capture more complex patterns and perform better.

## How Did ResNet Solve This?

ResNet introduced a new way of thinking about layers in a deep network. Instead of just stacking layers and hoping the network learns well, ResNet used **residual learning**:

- **Residual Learning:** The network learns to predict the difference (or "residual") between the output it wants and the output it currently has. This makes it easier for the network to learn even when it's very deep.

By using residual learning, ResNet made it possible to train networks with **hundreds** of layers, leading to significant improvements in performance on tasks like image classification.

## Conclusion

Before ResNet, the degradation problem made it hard to train very deep neural networks. ResNet's innovation of residual learning solved this problem, allowing for the creation of much deeper and more powerful networks.

# Understanding Skip Connections and Residual Blocks in ResNets

## Skip Connections
![alt text](image-1.png)
### What Are Skip Connections?

In deep neural networks, **skip connections** (also known as **shortcut connections**) are special pathways that bypass one or more layers. They allow the network to add the input of a block directly to its output.

### Why Are Skip Connections Important?

Skip connections help to solve a major problem in deep networks called the **vanishing gradient problem**. When a network is very deep, the gradients used to update the weights can become very small, making it hard for the network to learn. By using skip connections, the network can bypass some layers and ensure that the gradients have a more direct path, which helps in better training.

## Residual Blocks

### What Is a Residual Block?

A **residual block** is a fundamental building block of ResNets. Here's how it works:

1. **Two Weight Layers:** Each residual block contains two layers of weights (or neurons).
2. **ReLU Activation:** Each weight layer is followed by a ReLU activation function, which helps in introducing non-linearity.
3. **Shortcut Connection:** The input to the block is added to the output of the second weight layer. This is the **shortcut connection**.

### How Does a Residual Block Work?

1. **Input and Output:** Let's denote the input to a residual block as \( x \). The block applies two weight layers to \( x \), producing an output.
2. **Adding Input:** The output of the second weight layer is then added to the original input \( x \). This sum is the final output of the residual block.

   Mathematically, if \( F(x) \) is the function learned by the two weight layers, the output is \( F(x) + x \). This means the block learns the residual (or difference) between the desired output and the input \( x \), and then adds the input back.

### Why Use Residual Blocks?

- **Easier Learning:** By learning the residual (or difference) rather than the original function, it becomes easier for the network to learn. If the optimal function is close to an identity mapping (i.e., the output should be similar to the input), it's easier to adjust the residual to be close to zero.
- **Improved Training:** Residual blocks help prevent the vanishing gradient problem and allow very deep networks to be trained more effectively.

To understand why ResNet (Residual Networks) works effectively, it’s helpful to break down the concept of residual learning and provide a numerical example to illustrate its benefits.

## Why ResNet Works

### Key Concepts

1. **Residual Learning:** Instead of learning the desired mapping \( H(x) \) directly, ResNet learns the residual \( F(x) = H(x) - x \), where \( x \) is the input. The network then learns to predict \( F(x) \) and adds \( x \) back to get the final output: \( H(x) = F(x) + x \).

2. **Shortcut Connections:** These are direct paths that skip one or more layers and add the input to the output of a block. This helps gradients flow through the network more effectively, reducing the vanishing gradient problem and making it easier to train very deep networks.

### Why It Helps

1. **Easier Optimization:** Learning \( F(x) \) (the residual) is often easier than learning \( H(x) \) (the original mapping) directly. If the residual \( F(x) \) is close to zero, the network can easily learn the identity function.

2. **Gradient Flow:** Shortcut connections allow gradients to flow directly through the network, which helps in training deeper networks by mitigating the vanishing gradient problem.

## Numerical Example

Let's use a simplified numerical example to illustrate how residual learning works.

### Example Setup

- **Input (x):** \( [2, 3] \)
- **Desired Output (H(x)):** \( [5, 6] \)

We want to train a network to approximate \( H(x) \). Instead of learning \( H(x) \) directly, ResNet learns the residual \( F(x) \) where:

\[ F(x) = H(x) - x \]

### Calculate Residual

1. **Compute Residual \( F(x) \):**

   \[
   F(x) = H(x) - x
   \]

   \[
   F(x) = [5, 6] - [2, 3] = [3, 3]
   \]

2. **Train the Network to Learn \( F(x) \):**

   The network learns to output \( [3, 3] \) given the input \( [2, 3] \).

3. **Add Input \( x \) to Residual \( F(x) \):**

   After training, the network's output is:

   \[
   H(x) = F(x) + x
   \]

   \[
   H(x) = [3, 3] + [2, 3] = [5, 6]
   \]

### Comparison with Direct Learning

If we were to learn \( H(x) \) directly:

- **Desired Output (H(x)):** \( [5, 6] \)
- **Learning Task:** The network needs to learn \( [5, 6] \) directly from input \( [2, 3] \).

In a deep network, learning such a mapping can be challenging due to the complexity of the function and potential vanishing gradients.

### Why Residual Learning Is Better

By learning the residual \( [3, 3] \) instead of \( [5, 6] \):

1. **Simpler Task:** Learning \( [3, 3] \) is often simpler than learning \( [5, 6] \) directly.
2. **Identity Mapping:** If the residual \( F(x) \) were zero, the network would just need to output \( x \), making the learning task even simpler.


ResNet works effectively because it transforms the learning problem into a simpler one by focusing on learning residuals rather than direct mappings. This approach helps in training deeper networks by facilitating gradient flow and reducing the difficulty of optimization tasks.

# architecture of ResNet models

![alt text](image-2.png)

This image depicts the architecture of ResNet models of varying depths: 18-layer, 34-layer, 50-layer, 101-layer, and 152-layer.

Let's dig deeper into anyone of them, say ResNet50...

# Architecture of ResNet-50

## Overview

**ResNet-50** is a type of Residual Network (ResNet) that is known for its deep architecture, specifically having 50 layers. It is designed to address issues in training very deep neural networks by using residual learning with shortcut connections.

## Key Components of ResNet-50

1. **Input Layer**
   - **Size:** 224x224x3 (Height x Width x Channels)
   - **Purpose:** The input layer receives the raw image data, which is then processed through the network.

2. **Initial Convolution Layer**
   - **Layer:** Convolutional layer with a kernel size of 7x7
   - **Stride:** 2
   - **Purpose:** This layer performs initial feature extraction from the input image.

3. **Max Pooling Layer**
   - **Kernel Size:** 3x3
   - **Stride:** 2
   - **Purpose:** Reduces the spatial dimensions of the feature maps, making the network more computationally efficient.

4. **Residual Blocks**

   ResNet-50 consists of a series of residual blocks, each designed to learn residual mappings. The architecture uses **bottleneck blocks** which are more efficient compared to standard residual blocks.

   - **Bottleneck Block Structure:**
     1. **1x1 Convolution:** Reduces the number of channels, acting as a compression layer.
     2. **3x3 Convolution:** Performs the main convolution operation.
     3. **1x1 Convolution:** Expands the number of channels back to a higher dimension.

   - **Shortcut Connections:** Add the input of the block directly to its output, helping with gradient flow and training.

5. **Residual Block Groups**

   ResNet-50 is divided into four groups of residual blocks:

   - **Group 1:** 
     - **Layers:** 3 Bottleneck Blocks
     - **Output Channels:** 256
     - **Stride:** 1 (for all blocks in this group)

   - **Group 2:**
     - **Layers:** 4 Bottleneck Blocks
     - **Output Channels:** 512
     - **Stride:** 2 (for the first block in this group)

   - **Group 3:**
     - **Layers:** 6 Bottleneck Blocks
     - **Output Channels:** 1024
     - **Stride:** 2 (for the first block in this group)

   - **Group 4:**
     - **Layers:** 3 Bottleneck Blocks
     - **Output Channels:** 2048
     - **Stride:** 2 (for the first block in this group)

6. **Global Average Pooling Layer**
   - **Purpose:** Reduces each feature map to a single value by averaging over the entire spatial dimensions.
   - **Output Size:** 1x1x2048

7. **Fully Connected Layer**
   - **Purpose:** Converts the feature vector to class scores for classification.
   - **Output Size:** Number of classes (e.g., 1000 for ImageNet classification)

8. **Softmax Activation**
   - **Purpose:** Converts the output scores into probabilities for classification tasks.


In ResNet architectures, both **Basic Blocks** and **Bottleneck Blocks** are used, each serving a specific purpose to address the challenges of training deep networks. Here’s a detailed comparison between them:

## Basic Block
![alt text](image-3.png)
### Structure

- **Convolutional Layers:** Two 3x3 convolutional layers.
- **Batch Normalization:** Applied after each convolutional layer.
- **Activation Function:** ReLU (Rectified Linear Unit) applied after each convolution.
- **Shortcut Connection:** The input is added directly to the output of the block.



### Characteristics

- **Parameters:** Relatively more parameters due to two 3x3 convolutions.
- **Computational Complexity:** Higher compared to Bottleneck Blocks, especially in very deep networks.
- **Depth:** Used in shallower ResNet versions like ResNet-18 and ResNet-34.

### Use Case

Basic Blocks are simpler and are effective for relatively shallower networks. They are straightforward but can become inefficient as the network depth increases due to the higher computational cost.

## Bottleneck Block

### Structure

- **1x1 Convolutional Layer:** Reduces the number of channels (compression layer).
- **3x3 Convolutional Layer:** Performs the main convolution operation.
- **1x1 Convolutional Layer:** Expands the number of channels back to a higher dimension.
- **Batch Normalization:** Applied after each convolutional layer.
- **Activation Function:** ReLU applied after the 1x1 and 3x3 convolutions.
- **Shortcut Connection:** The input is added to the output of the block, similar to the Basic Block.



### Characteristics

- **Parameters:** Fewer parameters compared to Basic Blocks due to the use of 1x1 convolutions.
- **Computational Complexity:** More efficient, especially for very deep networks.
- **Depth:** Used in deeper ResNet versions like ResNet-50, ResNet-101, and ResNet-152.

### Use Case

Bottleneck Blocks are designed for deeper networks. They use fewer parameters and computational resources while maintaining the network’s capacity to learn complex representations. This makes them more suitable for very deep ResNet variants.


