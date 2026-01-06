
## Intro

[My](https://rickyding1997.github.io) approach to `brain simulation`

1. Identify a list of computational modules in the brain
2. For each module
    1. Understand what it computes: input -> output
    2. Get training data (synthetic or real-world)
    3. Train artificial neural networks to replicate its functionality
3. Combine modules

Why?

- Biological plausibility is a **trap**, simulating spikes and neurotransmitters does not help us understand how brain generates intelligence
- Analogy: considering transistor physics is irrelevant to understanding how a computer computes `a + b` -- they are on different **isolated** levels of abstractions, they do not depend on one another to work

- Current works in AI are mainly focused on solving daily-life tasks (text, image/video, game playing) -- I want to use these technologies to understand the brain
- It is well proven that ANNs can produce intelligence (LLMs, RL agents) -- making them qualified to model modules in the brain

## List of `implemented` modules

```bash
pip install brain-modules
```

1. [Hippocampal Place Cells (2014 Nobel Prize)](./docs/hippocampus/1_PlaceCells.md)
