
# A Functional, AI-Based Way to Understand Brain Computations

* Personal research plan / exploration
* My idea, but polished by ChatGPT

## 0. Why I’m interested in this

I’m interested in understanding **how the brain produces intelligence**.

Not how ion channels work.
Not how neurotransmitters diffuse.
But **what computations are being done**, and **how they combine into intelligent behavior**.

I don’t believe simulating more biological detail automatically brings more understanding.
At the same time, I don’t think current AI work (text, images, games) is aimed at understanding the brain either.

So my idea is simple:

> **Use successful AI methods, but aim them at understanding brain functions, not at daily-life tasks.**

---

## 1. What I think are problems with current approaches

### 1.1 Computational neuroscience (in general)

Many comp-neuro models focus on:

* spiking neurons
* synapses
* biophysical realism

But often:

* there is **no clear function being implemented**
* models are randomly connected
* people observe simple statistics (oscillations, firing rates)
* intelligence does not emerge

At some point, this feels like:

> “We simulated more details, but we still don’t know what the system is *doing*.”

If biological realism alone were the key, we could simulate molecules, DNA, or even atoms — but that clearly wouldn’t help us understand intelligence.

---

### 1.2 Modern AI (deep learning, RL)

AI avoids the “bio-plausibility trap” and that’s good.

It:

* uses objectives
* trains systems to *do something*
* produces planning, reasoning, abstraction

But most AI work:

* focuses on practical tasks (language, images, games)
* is **not designed to map to brain functions**
* doesn’t ask “what computation does the brain use here?”

So AI is powerful, but often **not aimed at brain understanding**.

---

## 2. Basic idea (very simple)

My approach is:

1. **Identify computations that the brain seems to perform**
2. **Reproduce those computations using ANN**
3. **Start simple, then gradually combine modules**

That’s it.

No attempt to simulate spikes.
No attempt to copy biology unless it helps.
No attempt to build a “full brain”.

Just:

> *What computation is needed here? Can an ANN implement it?*

---

## 3. Inspiration: levels of explanation (Marr, etc.)

This idea is not new.

[David Marr](https://en.wikipedia.org/wiki/David_Marr_(scientist)) suggested that systems should be understood at different levels:

* **What problem is being solved?**
* **What algorithm solves it?**
* **How is it physically implemented?**

The mistake many people make is jumping straight to the last level.

My focus is mainly on:

* *what* computation
* *how* it could work algorithmically

Biology is useful **later**, as a constraint or comparison — not as a starting point.

Many people in neuroscience, cognitive science, and NeuroAI already argue for this direction.

---

## 4. Methodology (step by step, realistic)

### Step 1: Pick **simple but meaningful brain functions**

Not “language” or “general intelligence”.

Examples of **small but non-trivial functions**:

* working memory (holding information across time)
* state estimation (hidden variables)
* credit assignment across time
* planning with an internal model
* memory indexing / retrieval
* attention / routing information

These are:

* well discussed in neuroscience
* clearly useful
* not just toy problems

---

### Step 2: Turn each function into a **clean task**

For each function:

* design a **minimal task** that *requires* that computation
* avoid shortcuts
* keep the environment simple and interpretable

For example:

* tasks that *cannot* be solved without memory
* tasks that *require* planning several steps ahead

This idea is common in cognitive modeling and RL research.

---

### Step 3: Use ANN to solve the task

Use whatever works:

* RNNs
* transformers
* simple feedforward models
* RL agents

The goal is not performance records, but:

* **does the model really implement the intended computation?**
* how does it represent information?
* how does learning shape the solution?

---

### Step 4: Analyze, not just train

After training:

* inspect internal representations
* test generalization
* remove parts and see what breaks

This is where understanding comes from.

---

### Step 5: Slowly combine modules

Only after understanding individual pieces:

* combine memory + planning
* combine perception + memory
* see where things interfere or help each other

This should be done **slowly**, not by scaling up blindly.

---

## 5. Important existing work along this path (I’m not alone)

Many people have already done **important pieces** of this idea:

### 5.1 Normative and task-based models in neuroscience

* Models that ask: *what computation would be optimal here?*
* Widely used for perception, decision making, and learning

### 5.2 RL as a model of dopamine and learning

* Temporal-difference learning linked to dopamine signals
* Very successful example of functional modeling

### 5.3 Memory models inspired by hippocampus

* Successor representations
* Episodic memory models
* Navigation and planning frameworks

### 5.4 AI models compared to brain data

* Deep networks compared with visual cortex
* Language models compared with brain responses
* Even when architectures differ, functions sometimes match well

### 5.5 NeuroAI and cognitive modeling communities

* Explicitly trying to connect AI models with brain computation
* Often emphasize function over biological detail

So this is **not a new idea** — it’s more like continuing and organizing an existing direction.

---

## 6. What I *don’t* expect from this approach

* I don’t expect to fully explain the brain
* I don’t expect perfect biological realism
* I don’t expect one clean modular decomposition

Brains are messy.

But even partial success — understanding *some* core computations — would already be valuable.

---

## 7. Why this feels like a reasonable path for an independent learner

* It’s **conceptual**, not hardware-heavy
* It doesn’t require massive datasets
* It builds understanding step by step
* It connects neuroscience, AI, and cognition in a concrete way

Most importantly:

> Every model must *do something meaningful*, or it’s discarded.

---

## 8. Short summary

In one paragraph:

> Instead of simulating more biological detail or chasing real-world AI benchmarks, I want to understand the brain by identifying core computations it performs and reproducing them using artificial neural networks. The plan is to start from simple but meaningful functions (like memory or planning), study them carefully, and then gradually combine them. This direction already exists in many forms across neuroscience and AI, and I see this as a personal attempt to learn from and contribute to that ongoing effort.
