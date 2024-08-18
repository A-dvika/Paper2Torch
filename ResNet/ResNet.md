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


