---
hide:
  - navigation
---
  

## Language Bias

As we mentioned in our paper, the language output of the model will to some extent affect our evaluation results. For the [longformer model](https://huggingface.co/LibrAI/longformer-harmful-ro), its performance in Chinese is relatively poor. Therefore, when calculating RtA, we consider responses with a Chinese character ratio greater than $\alpha$ as invalid sample (the default setting of $\alpha$ is 0.3)
