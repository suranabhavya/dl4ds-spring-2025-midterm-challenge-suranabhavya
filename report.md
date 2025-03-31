# DS542 Midterm Challenge Report

## 1. Comprehensive AI Disclosure Statement

In preparing this project, I consulted various resources—including ChatGPT—to clarify concepts and explore ideas for advanced techniques such as transfer learning, mixup augmentation, and learning rate scheduling. Although I received some initial guidance and ideas for model structures and experiment tracking strategies, every line of code was written, modified, and thoroughly tested by me. The AI assistance was used sparingly and only to verify best practices while I implemented the project independently.

## 2. Completeness and Clarity of the Report

This report provides an overview of the strategies and methodologies used to tackle the midterm challenge. It outlines:
- The rationale behind using different model architectures (a simple CNN baseline, a manually implemented ResNet, and a transfer learning approach with a pretrained ResNet50).
- Detailed descriptions of the data augmentation techniques (including Cutout, Mixup, Random Erasing, and Auto-Augment strategies) used to enhance generalization.
- A comprehensive explanation of the training setup, including hyperparameters, experiment tracking with Weights & Biases (WandB), and learning rate scheduling.
- An analysis of results on both the clean CIFAR-100 test set and out-of-distribution (OOD) samples.
- A brief ablation study discussing the impact of key augmentations and regularization techniques.

## 3. Thoroughness of Experiment Tracking

All experiments were meticulously tracked using Weights & Biases (WandB). For each part of the challenge, key metrics such as training and validation losses, accuracies, learning rates, and hyperparameter settings were logged. This allowed for:
- Easy comparison between the baseline, ResNet-based, and transfer learning approaches.
- Continuous monitoring of overfitting and performance during training.
- Effective model selection based on the best validation accuracy.
- Iterative tuning of parameters such as learning rate, weight decay, mixup strength, and dropout rates.


## 4. Justification of Design Choices

### Model Architectures:
- **Part 1 (Custom CNN):**  
  A custom CNN was developed as an initial baseline to verify the end-to-end pipeline and establish a performance reference.

- **Part 2 (ResNet-based Model):**  
  A manually implemented ResNet18 was used to explore the benefits of residual learning in deeper networks.

- **Part 3 (Pretrained Model with Transfer Learning):**  
  The final and best-performing approach uses a pretrained ResNet50 model. This choice leverages robust feature representations learned from the extensive ImageNet dataset, which are then fine-tuned on CIFAR-100. By replacing the final fully connected layer with one tailored for CIFAR-100 and adjusting learning rates (using a lower learning rate for the pretrained base and a higher one for the classifier), the model benefits from both strong general-purpose features and dataset-specific fine-tuning. This approach significantly improved performance on both clean and OOD test sets.

### Data Augmentation:
- Advanced augmentations such as Random Crop, Horizontal Flip, Color Jitter, and Random Erasing were applied to enrich the training data.
- Techniques like Mixup and Cutout were integrated to further regularize the model and mitigate overfitting.
  
### Training Strategies:
- **Learning Rate Scheduling:**  
  Schedulers like Cosine Annealing and OneCycleLR were experimented with to ensure smooth and effective convergence.
- **Regularization:**  
  Methods including label smoothing and dropout were incorporated to mitigate overfitting.

## 5. Analysis of Results

- **Clean CIFAR-100 Test Set:**
  - The best-performing models across the different parts achieved significant improvements over the baseline. Detailed performance metrics (accuracy, loss) were logged via WandB, enabling a clear view of the model’s progression.
  
- **OOD Evaluation:**
  - OOD testing was performed to ensure robustness under distributional shifts. The performance on these images was competitive and indicated that the advanced augmentation strategies and fine-tuning of pretrained models contributed positively to generalization.
  
- **Overall Performance:**
  - The transfer learning approach (Part 3) with ResNet50 fine-tuning produced the best results, surpassing the benchmark accuracy of 39.7% on the Kaggle leaderboard. This validates the effectiveness of leveraging pretrained models and carefully designed augmentation and regularization strategies.

## 6. Ablation Study

A focused ablation study was performed to assess the impact of key augmentation and regularization strategies:
- **Mixup Augmentation:**  
  Removing mixup resulted in higher training accuracy but increased overfitting, as evidenced by a drop in validation performance.
- **Cutout and Random Erasing:**  
  These augmentations improved robustness, particularly evident in the OOD evaluations.
- **Label Smoothing:**  
  Without label smoothing, the model produced overconfident predictions that degraded generalization performance.
- **Learning Rate Schedulers:**  
  While OneCycleLR accelerated early convergence, both OneCycleLR and Cosine Annealing led to similar final performance, underscoring the importance of a well-planned learning rate schedule.

## 7. Conclusion

This project demonstrates a systematic progression from a custom CNN baseline to a deep ResNet architecture and finally to a transfer learning model using a pretrained ResNet50. Experiment tracking with WandB enabled detailed performance monitoring and rapid iteration. The integration of advanced augmentation techniques and regularization strategies played a crucial role in improving model generalization, particularly in challenging OOD scenarios. Overall, the pretrained ResNet50 model provided the best results by leveraging strong, transferable features from ImageNet and fine-tuning them effectively for CIFAR-100, thereby surpassing the benchmark and achieving robust performance.

---