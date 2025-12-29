# Notes form Ultrascale Playbook

This tutorial contains my distilled notes from Hugging Face’s [Ultrascale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook). While I will try to capture the core technical logic of the book, I still highly recommend reading the original book as there are many things that I have not mentioned here. 

In these notes we will first start with important refreshers, then we dive into single GPU optimization, and only after that we will move on to multi GPU training and parallelization methods. There are 5 main parallelization methods that we will discuss here: Data Parallelism (ZeRO-1, ZeRO-2, ZeRO-3), Tensor Parallelism, Context Parallelism, Pipeline Parallelism and finally Expert Parallelism.

**Now, let’s begin!**

# Numerical precision

In a GPU (and computer hardware in general), we don't store numbers like “12.5” as a single value. We store it as a binary formula composed of three parts.

$$
\text{Value} = (-1)^{\text{Sign}} \times (1 + \text{Mantissa}) \times 2^{(\text{Exponent})}
$$

- **Sign:** determines if the value is positive or negative (always 1 bit).
- **Mantissa:** determines the precision. More bits here mean more decimal places and higher accuracy.
- **Exponent:** determines the range. A larger exponent allows the model to represent massive or tiny numbers without hitting "overflow" or "underflow."

Choosing a format is a trade-off between speed, memory, and numerical stability. Here are some standard formats used in LLM training:

| **Format** | **Mantissa** | **Exponent** | **Max Value** | **Typical Use Case** |
| --- | --- | --- | --- | --- |
| **FP32** | 23 bits | 8 bits | $3.4 \times 10^{38}$ | Master Weights, Optimizer States |
| **BF16** | 7 bits | 8 bits | $3.4 \times 10^{38}$ | **LLM Training Standard** |
| **FP16** | 10 bits | 5 bits | $65,504$ | Inference, Older Training |

# The Memory Hierarchy

To understand why GPU training is often limited by data movement rather than raw math speed, we must look at the GPU memory hierarchy.

GPUs utilize several different memory types, each with a specific trade-off between capacity and speed. It is helpful to visualize this as a pyramid: the closer you get to the "brain" (the tensor cores) where all the calculations happen, the faster the memory becomes, but the less space you have.

- **VRAM (High Bandwidth Memory - HBM):** This is the "main warehouse." It is large (80GB on an H100) but relatively slow to access. This is where our model weights, gradients, and optimizer states live.
- **L2 Cache:** A middle-ground storage area that is much faster than VRAM but much smaller. It acts as a staging area to reduce the number of trips the GPU has to make to the VRAM.
- **SRAM (Shared Memory / L1 Cache):** This is incredibly fast, on-chip memory. However, it is tiny (measured in kilobytes/megabytes). Data must be here for the GPU to work on it efficiently.
- **Registers:** The "desktop." This is where the actual numbers sit while the tensor cores perform the matrix multiplication.

During training, data is streamed through this hierarchy. The efficiency of our training often depends on how effectively we can keep the tensor cores fed without waiting for data to arrive from the slower VRAM:

$$
\text{VRAM} \rightarrow \text{L2 Cache} \rightarrow \text{SRAM} \rightarrow \text{Registers} \rightarrow \text{Tensor Cores}
$$

![image.png](Notes%20form%20Ultrascale%20Playbook/56e434e4-da9e-430b-babe-d5e8dced490c.png)

# The Memory Math

When training a neural network model, we store several items in memory:

- Model Weights: Stored in BF16 (2 bytes per parameter).
- Gradients: Also stored in BF16 (2 bytes per parameter).
- Optimizer States (Adam): This is the heaviest part. It stores a master copy of the weights in FP32 (4 bytes), plus the momentum (4 bytes) and variance (4 bytes) matrices. Total: 12 bytes per parameter.
- Activations: Output of each layers after forward pass (stored in BF16)

Here is the formula to calculate memory usage to store a model inside GPU memory:

$$
m_{static} = \underbrace{(2 \times N)}_{\text{Weights}} + \underbrace{(2 \times N)}_{\text{Gradients}} + \underbrace{(12 \times N)}_{\text{Optimizer}} = 16 \times N \\ m_{activations} = L \cdot seq \cdot bs \cdot h \cdot \left( 34 + \frac{5 \cdot nheads \cdot seq}{h} \right) \\m_{training} = m_{static} + m_{activations}
$$

Where:

- $N$ : Total number of model parameters
- $L$ : Number of layers
- $h$ : Hidden dimension size
- $nheads$: Number of attention heads

Let’s get a general sense of how much memory we need for a model (with full and mixed precision giving the same overall values):

| **Model parameters** | **FP32** | **BF16**  |
| --- | --- | --- |
| 1B | 16 GB | 20 GB |
| 7B | 112 GB | 140 GB |
| 70B | 1120 GB | 1400 GB |
| 405B | 6480 GB | 8100 GB |

![Screenshot 2025-12-22 at 18.24.18.png](Notes%20form%20Ultrascale%20Playbook/Screenshot_2025-12-22_at_18.24.18.png)

# Single GPU training

Now that we understand the memory constraints, let’s look at the actual execution. A single training iteration is composed of three distinct phases:

1. **The Forward Pass:** Data flows from the input through each layer. We calculate the output and store the activations in VRAM (because we’ll need them later for gradient computation).
2. **The Backward Pass:** We calculate the gradients. This is the most computationally expensive part.
3. **The Optimizer Update:** The GPU uses the gradients and optimizer states to adjust the weights.

But here emerges our first problem. Let’ stake a look at the graph below indicating memory usage for different sequence lengths:

![Graph of memory usage for different sequence lengths. Memory usage scales linearly with the batch size and quadratically with the sequence length.](Notes%20form%20Ultrascale%20Playbook/Screenshot_2025-12-22_at_18.44.07.png)

Graph of memory usage for different sequence lengths. Memory usage scales linearly with the batch size and quadratically with the sequence length.

These graphs tell a striking story: for short sequences (or small batch sizes), memory usage for activations is almost negligible, but from around 2-4k tokens they start to take up a significant amount of memory, while usage for parameters, gradients, and optimizer states (as we’ll discuss later) is roughly independent of the sequence length and batch size.

For large numbers of input tokens (i.e., large batch sizes/sequences), activations become by far the largest memory burden. To cap the activation memory footprint we will use a technique called “activation recomputation”.

## **Activation recomputation (Gradient Checkpointing)**

In standard training, you store every single activation from the forward pass in VRAM because you need them later to calculate gradients during the backward pass.

The Core Mechanism of activation recomputation:

Instead of keeping everything in VRAM, the GPU follows a **Checkpointing** strategy:

1. **Forward Pass (Sparse Saving):** The GPU calculates all activations but **discards** most of them. It only saves "checkpoints" at specific milestones (e.g., the input to every Transformer block).
2. **Backward Pass (On-Demand Recovery):** When the backward pass reaches a layer whose activations were discarded, the GPU pauses. It goes back to the last "checkpoint" and **re-runs the forward math** for that small section to regenerate the needed activations.
3. **Calculation:** Once the activations are back, it finishes the gradient calculation and immediately deletes them again.

![On the left we have memory graph of 8B model with no recomputation, and on the right is the same model with selective recomputation.](Notes%20form%20Ultrascale%20Playbook/image.png)

On the left we have memory graph of 8B model with no recomputation, and on the right is the same model with selective recomputation.

Unfortunately, activations still have a linear dependence on the batch size, and all our profiles in the bar plots above were using batch_size = 1, so as we move to larger batch sizes this might become an issue again. Fortunately, we have a second tool in our box - **gradient accumulation** to the rescue!

## Gradient accumulation

Batch sizes can get huge, so we divide batches into micro batches, compute forward and backward for each micro batch and the either sum or mean the grads.

1. **Forward/Backward 1:** Process Micro-batch #1 → Compute gradients → **Store them** in the `.grad` buffer.
2. **Forward/Backward 2:** Process Micro-batch #2 → Compute gradients → **Add them** to the existing gradients in the buffer.
3. **Repeat:** Do this for $N$ steps.
4. **The Update:** Only after $N$ steps do you call `optimizer.step()` and `optimizer.zero_grad()`.

Gradient accumulation is the number of forward-backwards passes done with micro batches to get one 1 batch size.

With gradient accumulation, the global batch size can be computed as follows:

$$
bs=gbs=mbs×grad_{acc}
$$

Gradient accumulation allows us to effectively increase our batch size up to infinity while the memory footprint stays constant.

# Multi GPU Training

Now let’s get a larger workstation with a couple of GPUs and start investigating our first scaling technique, called ***data parallelism*** - which, as we'll see, is just a parallel version of gradient accumulation. 

Important knowledge about communication methods to know before we start:

| **Collective Primitive** | **Operation (Logic)** | **Resulting State** | **Key Use Case in Ultrascale** |
| --- | --- | --- | --- |
| **All-Gather** | Each GPU starts with a unique shard and distributes it to all other GPUs in the group. | Every GPU possesses the **complete, concatenated tensor**. | **ZeRO-3 Forward/Backward:** Materializing full weights from shards. |
| **Reduce-Scatter** | Performs an element-wise reduction (usually a sum) across GPUs and "scatters" the result. | Each GPU possesses only a **unique shard of the summed total**. | **ZeRO-2/3 Backward:** Summing gradients while keeping them sharded. |
| **All-Reduce** | A global sum followed by a broadcast (often implemented as Reduce-Scatter + All-Gather). | Every GPU possesses the **identical, full-summed tensor**. | **Standard DDP:** Synchronizing gradients before the optimizer step. |

## Data Parallelism (DP)

The idea of data parallelism is to clone model weights in each gpu and run forward and backward passes on different micro batches of data in parallel on each GPU and then gather all gradients for the optimizer.

![image.png](Notes%20form%20Ultrascale%20Playbook/image%201.png)

Let’s say we have global batch size of 64, and 4 GPUs. DP for each gpu is 16 and we have micro batch size of 2. Here is what happens:

$$
\begin{aligned} 
\left.
\begin{aligned} 
&\text{GPU}_1:(2 \text{ samples} \rightarrow \text{fwd/bwd} \rightarrow \text{grad*}) \times 8 \text{ times}\rightarrow \text{Grad}_1 \\
&\text{GPU}_2: (2 \text{ samples} \rightarrow \text{fwd/bwd} \rightarrow \text{grad*}) \times 8 \text{ times} \rightarrow \text{Grad}_2 \\
&\text{GPU}_3: (2 \text{ samples} \rightarrow \text{fwd/bwd} \rightarrow \text{grad*}) \times 8 \text{ times} \rightarrow \text{Grad}_3 \\
&\text{GPU}_4: (2 \text{ samples} \rightarrow \text{fwd/bwd} \rightarrow \text{grad*}) \times 8 \text{ times} \rightarrow \text{Grad}_4
\end{aligned}
\right\} 
\xrightarrow{\substack{\text{All} \\ \text{Reduce}}} 
\text{Global Grads}
\end{aligned}
$$

* - Micro batch gradients at each pass are saved in grad buffer and summed together.

All reduce is a circular process where each gpu sends a fraction of its gradients to the next one until each GPU holds the global gradients. The complexity of all reduce is O(N).

However, there are some downsides of DP:

1. All reduce is bottleneck because if we have 1024 GPUs, it will take 1023 operations for all devices to obtain the full gradients, while stalling. Also with bigger DP our computation-communication ratio gets smaller and smaller which is undesirable.
2. Every GPU holds a copy of the model. This is fine for smaller models, but larger models is hard to fit into one GPU.

---

## **Zero Redundancy Optimizer (ZeRO)**

While data parallelism is an efficient way to scale training, the naive replication of optimizer states, gradients, and parameters across each DP rank introduces significant memory redundancy. ZeRO eliminates this by partitioning the optimizer states, gradients, and parameters across the data parallel dimension, while still allowing computation with the full set of parameters. This sometimes requires more communications between DP ranks, which may or may not be fully overlapped, as we’ll see next!

This approach is organized into three possible optimization stages:

- ZeRO-1: optimizer state partitioning
- ZeRO-2: optimizer state + gradient partitioning
- ZeRO-3: optimizer state + gradient + parameter partitioning

![Here, Ψ denotes the number of parameters, *k* denotes the memory multiplier of optimizer states (k=12 for Adam, as we've just seen), and N_d denotes DP degree.](Notes%20form%20Ultrascale%20Playbook/image%202.png)

Here, Ψ denotes the number of parameters, *k* denotes the memory multiplier of optimizer states (k=12 for Adam, as we've just seen), and N_d denotes DP degree.

---

### Zero - 1: optimizer state partitioning

In ZeRO-1, optimizer states are partitioned into **$N_d$** equal shards, where **$N_d$** is the data-parallel (DP) degree. As a result, each DP rank stores only **$1/N_d$** of the optimizer states. During the optimization step, each rank updates only the corresponding **$1/N_d$** shard of the **FP32** master ****weights.

Each GPU holds replicas of model parameters and gradients while sharding the optimizer states.

1. Each GPU performs forward pass for its micro batch and get activations
2. Each GPU then performs backward pass on activations to get local gradients
3. Perform all-reduce to get global gradients from all GPUs
4. Local optimizer updates its local states in each GPU
5. Perform all gather to get fully updated model parameters in BF16 and store in each GPU
    
    ![dp_zero1.gif](Notes%20form%20Ultrascale%20Playbook/dp_zero1.gif)
    

---

### Zero - 2: optimizer state + gradient partitioning

Since on each replica we only need to have the gradient shard corresponding to its optimizer state shard, it makes sense to shard gradients as well, similarly to the optimizer states. Then, during the backward pass, instead of performing an all-reduce over the gradients, we only perform a reduce-scatter operation!

Each GPU holds replicas of model parameters while sharding the gradients optimizer states.

1. Each GPU performs forward pass for its micro batch and get activations.
2. Each GPU performs backward pass to get its local gradients.
3. Perform reduce-scatter to receive the part of gradients that it needs to update in the optimizer ($1/N_d$).
4. Local optimizer updates its local states in each GPU.
5.  Perform all gather to get fully updated model parameters in BF16 and store in each GPU.
    
    ![dp_zero2.gif](Notes%20form%20Ultrascale%20Playbook/dp_zero2.gif)
    

### Zero - 3: optimizer state + gradient + parameter partitioning

The "magic" of **ZeRO-3** is that it collects and discard parameters **layer-by-layer** just in time for the math.

1. **The Trigger:** The code starts the forward pass and reaches **Layer 1**.
2. **All-Gather (Layer 1):** Each GPU shouts to the others to get the missing shards for *only* Layer 1. Now, every GPU has the full parameters for Layer 1.
3. **Compute:** Each GPU runs its **Micro-batch** through Layer 1.
4. **Discard (Release):** As soon as the math for Layer 1 is done, the GPUs **delete** the full parameters they just collected, keeping only their original small shard.
5. **Repeat:** This repeats for Layer 2, Layer 3, and so on.
6. After that each GPU will hold its local activation. 
7. Next we do backward pass (exactly the same way as forward pass just inverted in flow) to compute local gradients.
8. Perform reduce-scatter to receive the part of gradients that it needs to update in the optimizer ($1/N_d$)
9. Local optimizer updates its local states in each GPU
10. Optimizer uses local gradients to update local model parameters

### Summary of ZeRO

With ZeRO, we can train even models that would ordinarily not fit into a single GPU by sharding the parameters, gradients, and optimizer states across DP replicas, while incurring a small communication cost.

![Screenshot 2025-12-24 at 13.43.07.png](Notes%20form%20Ultrascale%20Playbook/Screenshot_2025-12-24_at_13.43.07.png)

However, there are some limits here: DP only works if a layer of the model fits in a single GPU, and ZeRO can only partition the parameters, gradients, and optimizer states, not the activation memory! As you can remember, the memory scales with sequence length and batch size. We could just limit those, but in practice we don’t want to be limited by hardware to train with only a short sequence length.

To overcome this issue, it's time to examine a new, orthogonal axis of parallelism - ***tensor parallelism (TP)***. Unlike ZeRO-3, which relies on heavy parameter communication, TP proposes to shard parameters, gradients, optimizer states, AND activations across devices without requiring any communication of model parameters between GPUs.

---

## Tensor & Sequence Parallelism (TP-SP)

TP-SP represents the most granular form of model parallelization. Unlike standard Data Parallelism, where GPUs act as independent workers processing different sentences, TP-SP forces multiple GPUs to cooperate as a single "Super GPU" on a single sample. This strategy is essential for training ultra-large models (like Llama-3 70B) because it shards the two biggest memory consumers: Weights and Activations.

- **Tensor Parallelism (TP):** Shards the weight matrices of the Linear and Attention layers. By using Column and Row Parallelism, the model's parameters are split across GPUs, meaning no single chip needs to hold the full 140GB+ of weights for a 70B model.
- **Sequence Parallelism (SP):** Extends sharding to the **Activations**. Instead of every GPU redundantly storing the same sequence of tokens for LayerNorm, Dropout, and Residual connections, SP slices the sequence along the token dimension. If a sequence has 1,024 tokens and you have 4 GPUs, each GPU only stores the activations for 256 tokens.

Because TP-SP requires GPUs to communicate multiple times **inside every single transformer layer**, the "Network Tax" is extremely high. 

![Untitled design.png](Notes%20form%20Ultrascale%20Playbook/Untitled_design.png)

**More about TP:**

There are 2 ways we can perform TP: “column” and “row” parallelism. Column and Row parallelism work together in a specific sequence during TP parallelism.

In the first half of the "sandwich," we use Column Parallelism for the expansion layer. Each GPU receives the full input but only holds a vertical slice of the weight matrix. Because each GPU is working on a different set of output neurons, they can all compute their partial results simultaneously without needing to talk to one another. This is highly efficient for memory because the massive weight matrix is split, but it traditionally requires every GPU to have a redundant copy of the input data.

![Column parallelism](Notes%20form%20Ultrascale%20Playbook/image%203.png)

Column parallelism

The second half of the operation is Row Parallelism, which handles the contraction back to the original dimension. This is where the magic of the "sandwich" happens: since the previous layer produced sharded outputs, those outputs can stay right where they are and serve as the sharded inputs for the Row-wise layer. Each GPU multiplies its local input shard by its local row shard of the weights. To get the final, mathematically correct result, the GPUs perform a single All-Reduce to sum their partial results together. This design is elegant because it allows the GPUs to stay "siloed" and productive through most of the block, only paying the network tax once at the very end.

![Row parallelism](Notes%20form%20Ultrascale%20Playbook/image%204.png)

Row parallelism

In practice, we combine these. We use **Column Parallelism** to expand the hidden state and **Row Parallelism** to bring it back. This 'Sandwich' is clever because it allows the GPUs to work independently for most of the calculation, only requiring a single **All-Reduce** at the very end of the block to synchronize their results.

---

## Context Parallelism (CP)

**Context Parallelism** is the specialized strategy used to handle **huge** context windows (e.g., 128k to 1M+ tokens). When an input sequence (like an entire book) is too large for the VRAM of a single GPU or even a single server node, CP shards that sequence across the data center.

> The Core Logic: While Sequence Parallelism (SP) shards tokens across GPUs inside a node via NVLink, Context Parallelism (CP) shards tokens across nodes using the cluster network (InfiniBand/Ethernet).
> 

**The "Ring Attention" Example:**

To understand how GPUs "talk" across shards, let’s use an **8-word sentence** processed with **CP=4**.

The Sentence: `"The quick brown fox jumps over the dog."`

**Step 1: Distribution (The Sharding)**

The sequence is split into "chapters." Each GPU is now the "owner" of the Query ($Q$) for its specific words.

- **GPU 1:** `[The, quick]`
- **GPU 2:** `[brown, fox]`
- **GPU 3:** `[jumps, over]`
- **GPU 4:** `[the, dog]`

**Step 2: The Attention Problem**

In a Transformer, every word must "attend" to every other word. **GPU 4** (holding "the dog") must look at the "fox" on **GPU 2** to calculate its attention score. However, GPU 4 physically does not have the "fox" in its memory.

**Step 3: The Ring Relay (The "Magic")**

Instead of gathering the whole sentence (which would cause an OOM crash), the GPUs pass their **Keys** $K$ and **Values** $V$ around in a circle.

| **Rotation** | **GPU 1 (The, quick) looks at...** | **GPU 4 (the, dog) looks at...** |
| --- | --- | --- |
| **Start** | Its own KV: `[The, quick]` | Its own KV: `[the, dog]` |
| **Pass 1** | Receives KV from GPU 4: `[the, dog]` | Receives KV from GPU 3: `[jumps, over]` |
| **Pass 2** | Receives KV from GPU 3: `[jumps, over]` | Receives KV from GPU 2: `[brown, fox]` |
| **Pass 3** | Receives KV from GPU 2: `[brown, fox]` | Receives KV from GPU 1: `[The, quick]` |

**The Result**

By the end of the 3rd pass, **GPU 4** has successfully calculated attention scores for its words against every other word in the sentence.

**Key Achievement:** At no point did any single GPU have to store more than **4 words** of $K$ and $V$ data at once, yet they mathematically achieved the same result as if they had processed the full 8-word sentence on one machine.

- **SP (Collective):** Uses **All-Gather**. Every GPU talks to every other GPU simultaneously to reconstruct the full sequence for the linear layers. This is a "heavy" burst of traffic that only works on NVLink.
- **CP (Point-to-Point):** Uses a **Ring**. GPU 1 *only* talks to GPU 2. It passes a small chunk of memory (the KV cache), calculates, and passes it again. This "Relay Race" is much easier on the network between servers because it doesn't try to flood the entire cluster at once.

There is a slight problem. In decoder-only models, tokens can only look at the past. This means GPU 1 (holding the start of the sentence) has almost no work to do, while GPU 4 (holding the end) must calculate attention for the entire sequence. This inequality leaves your most expensive hardware sitting idle. To fix this, we use **Zig-Zag Ring Attention**.

![image.png](Notes%20form%20Ultrascale%20Playbook/image%205.png)

### **Zig-Zag Ring Attention**

We need a better way to distribute the input sequences. This can be achieved by not assigning the tokens to the GPUs in a purely sequential manner, but instead mixing up the ordering a bit such that we have a good mix of early and late tokens on each GPU. 

![image.png](Notes%20form%20Ultrascale%20Playbook/image%206.png)

By pairing early and late tokens on the same GPU, we balance the computational load across the entire ring, ensuring that every GPU in the cluster stays fully utilized.

## Pipeline parallelism (PP)

In pipeline parallelism we split model layers between GPUs which looks similar to ZeRO-3. However, there is a key difference between these 2 approaches. I will use kitchen example to explain that:

**The "Kitchen" Analogy**

Imagine we are baking a 100-layer cake (a 100-layer model) with 4 chefs (GPUs).

- **ZeRO-3 (Collaborative Baking):** Every chef stands in front of the *same* oven. They split the recipe into 4 pieces. When it's time to bake Layer 1, they all shout and gather the instructions, bake it together, then throw the instructions away and move to Layer 2.
    - *Key:* Every chef works on **every layer**, but they only "memorize" a piece of the recipe.
- **Pipeline Parallelism (Assembly Line):** You set up 4 different stations. Chef 1 is *only* responsible for Layers 1–25. Chef 2 handles 26–50, and so on. Chef 1 bakes his part, then slides the cake to Chef 2.
    - *Key:* Each chef **owns** a specific set of layers. Chef 1 never even sees the instructions for Layer 50.

**The PP Advantage:** In PP, you only send the **Activations** (the "intermediate result") between GPUs. Activations are often much smaller than the full model weights. This allows PP to scale across racks where the network is slower.

### Naive PP

The biggest downside of PP is that it's hard to keep everyone busy. If Chef 1 is working on Layer 1, Chef 2 is sitting idle waiting for the cake to be passed. This waiting time is called a **Bubble**. Here's how our GPU utilization looks when doing a naive and simple forward and backward pass through the model (here, the numbers indicate the model layers):

![image.png](Notes%20form%20Ultrascale%20Playbook/image%207.png)

### ***All forward, all backward*** PP ***(AFAB)***

To fix bubble of naive PP, we use **Micro-batches** (passing many small cakes instead of one big one). Now, when the second GPU is busy processing micro-batch 1, the first GPU can already start processing micro-batch 2. Here is a schedule using eight micro-batches:

![Before, the numbers in the diagram indicated the layers, but in all pipeline parallel plots from here on they indicate micro-batches. You can think of each square here as containing several layers, as seen in the previous figure.](Notes%20form%20Ultrascale%20Playbook/image%208.png)

Before, the numbers in the diagram indicated the layers, but in all pipeline parallel plots from here on they indicate micro-batches. You can think of each square here as containing several layers, as seen in the previous figure.

However, just as annoying as the bubble is the memory required for storing all the activations. We need to keep all of the activations in memory until we reach the backward stage, which quickly leads to a memory explosion in these implementations of PP. Can we do better and avoid this issue?

### **One forward, one backward PP**

This schedule is called ***one forward, one backward (1F1B)*** because the middle/steady state involves alternately performing one forward and one backward pass. The general idea is to start performing the backward pass as soon as possible. The schedule looks like this:

![Screenshot 2025-12-25 at 20.52.32.png](Notes%20form%20Ultrascale%20Playbook/Screenshot_2025-12-25_at_20.52.32.png)

While 1F1B significantly reduces our activation memory footprint, the pipeline bubble still remains a major efficiency bottleneck.

### Interleaved Pipeline Parallelism

The fundamental problem with standard 1F1B is that the "bubble" size is proportional to the number of pipeline stages. To fix this, we use **Interleaved Pipeline Parallelism**. Instead of one GPU owning a single contiguous block of layers (e.g., Layers 1–8), we split the model into more granular pieces.

In an interleaved setup, GPU 1 might own Layers 1–2 and Layers 9–10, while GPU 2 owns Layers 3–4 and 11–12. By breaking the model into these smaller chunks, the micro-batches can "loop" through the GPUs more frequently. This significantly reduces the size of the initial "warm-up" and final "cool-down" bubbles, leading to much higher GPU utilization. The trade-off is a slight increase in communication frequency, as the GPUs must pass activations back and forth more often than in a simple linear pipeline. 

Let's take a look at how this works:

![Screenshot 2025-12-25 at 20.52.12.png](Notes%20form%20Ultrascale%20Playbook/Screenshot_2025-12-25_at_20.52.12.png)

### Zero Bubble

While interleaving shrinks the bubble, **Zero Bubble** is a more recent scheduling innovation that aims to eliminate it entirely.5 The insight behind Zero Bubble is that the backward pass is actually composed of two different types of math:

1. $\text{grad\_input}$ **(B):** Calculating the gradients for the previous layer so the pipeline can keep moving backward.
2. 2. $\text{grad\_weight}$ **(W):** Calculating the actual updates for the weights of the current layer.

In standard 1F1B, the GPU waits until it has finished both B and W before moving on. Zero Bubble schedules these operations asynchronously.6 It prioritizes the "B" pass (to keep the pipeline flowing) and "hides" the "W" pass (which is computationally heavy but not required for the pipeline to continue) inside the idle gaps.7 By intelligently reordering these sub-steps, Zero Bubble can theoretically achieve near-100% efficiency, effectively "filling" the bubble with weight gradient calculations.

![image.png](Notes%20form%20Ultrascale%20Playbook/image%209.png)

---

## Expert parallelism (EP)

As we move toward Mixture-of-Experts (MoE) architectures, like those used in DeepSeek or Mixtral, we encounter a unique type of layer. In an MoE model, only a fraction of the model is activated for any given token. This creates a massive memory footprint (because you have many experts) but a low computational cost per token. **Expert Parallelism** is designed specifically to shard these independent expert layers across GPUs.
In Expert Parallelism, we shard the **Experts**. If a layer has 8 experts and we have 8 GPUs, each GPU homes exactly one expert.

Let’s say we have 8 GPUs and 8 experts.

1. **Parallel Processing:** All 8 GPUs begin the forward pass independently. They process their respective micro-batches
2. **The Routing Decision:** When the micro-batches reach the MoE layer, the Router analyzes every token in every micro-batch. It decides which of the 8 experts is best suited to handle each specific token.
3. **The Token Exchange (All-to-All):** Since "Expert 1" lives on GPU 1 and "Expert 8" lives on GPU 8, the GPUs must swap tokens. If GPU 1 is holding a token that needs Expert 8, it "dispatches" that token across the network.
4. **Specialized Computation:** Once the exchange is complete, each GPU finds itself holding a new batch of tokens that all specifically required its local expert. The GPU performs the FFN calculation and then "returns" the tokens to their original owners.

# Summary

We have now explored the five core parallelism strategies required to scale model training across thousands of GPUs. Each strategy targets a specific bottleneck, whether it is memory storage, network bandwidth, or the quadratic cost of long sequences.

| **Strategy** | **Sharding Axis** | **Primary Goal** |
| --- | --- | --- |
| **Data Parallelism (DP/ZeRO)** | **Batch Dimension** | Replicates the model to increase throughput. **ZeRO** 1, 2, and 3 further eliminate redundancy by sharding optimizer states, gradients, and parameters. |
| **Tensor Parallelism (TP)** | **Hidden Dimension** | Slices individual weight matrices (Width) across GPUs to fit ultra-wide layers that exceed single-GPU VRAM. |
| **Sequence & Context (SP/CP)** | **Sequence Dimension** | Shards the input tokens. **SP** handles intra-node activations via NVLink, while **CP** enables million-token windows using inter-node Ring Attention. |
| **Pipeline Parallelism (PP)** | **Layer Dimension** | Slices the model by depth (Layers). This "Assembly Line" approach is essential for models too tall to fit on a single server node. |
| **Expert Parallelism (EP)** | **Expert Dimension** | Shards independent "Experts" in MoE architectures. It uses **All-to-All** exchanges to route tokens to specialized GPUs. |

In practice, these methods are rarely used in isolation. To train the world’s largest models, engineers stack these dimensions to achieve maximum efficiency. Here are some combos:

1. ZeRO-1 / ZeRO-2 + PP (DeepSeek-v3)
2. TP + PP / ZeRO-3
- **ZeRO-1/2 + PP:** Used in training models like **DeepSeek-V3**. This combines the low-overhead weight sharding of ZeRO with the multi-node scalability of Pipeline Parallelism.
- **TP + PP or ZeRO-3:** The "Maximum Memory" configuration. This is used for models so massive (like Llama-3 405B) that every single component (weights, activations, and optimizer states) must be sharded in every possible direction.
- **DP + EP:** The standard for **Mixture-of-Experts**. This allows researchers to scale the "knowledge" of the model (experts) without increasing the computational cost per token.