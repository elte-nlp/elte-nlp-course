---
title: "Large World Models: Takeaways & Review"
subtitle: "GPUDay 2024, Budapest, Hungary"
author: Natabara Máté Gyöngyössy
institute: "Eötvös Loránd University, Department of Artificial Intelligence"
date: 2024
theme: Marburg
colortheme: orchid
fontsize: 17pt
linkcolor: blue
header-includes: |
  \let\emphasized\emph
  \let\strong\textbf
  \renewcommand{\textbf}[1]{\textcolor{blue}{\strong{#1}}}
  `\setbeamertemplate{navigation symbols}{}`{=latex}
  `\setbeamertemplate{footline}[page number]`{=latex}
link-citations: true
aspectratio: 1610
output: 
  beamer_presentation:
    slide_level: 2
bibliography: extra_gpuday_lam.bib
---

# How to model the world?

## Language

Natural language...

- is a symbolic sequence.
- is context-aware.
- could be ambiguous.
- follows a given structure and conveys meaning.

# Language, but Grounded

## Meanings as Mental Images

::: columns

:::: {.column width="20%"}

![](./figures_gpu/gpuday_arrow_1.png){height=70%}

::::

:::: {.column width="70%"}

\centering
![](./figures_gpu/gpuday_head_1.png){height=70%}

::::

:::
\small
Perception of the language creates a mental image similar to percieving or recalling the object [@deacon1997symbolic].

## Meanings as Associative Mappings

::: columns

:::: {.column width="20%"}

![](./figures_gpu/gpuday_arrow_2.png){height=68%}

::::

:::: {.column width="70%"}

\centering
![](./figures_gpu/gpuday_head_2.png){height=68%}

::::

:::

\small
Language is a learned by internalizing distributed probabilistic connections of word-word and word-object structures [@deacon1997symbolic; @skinner1957verbal].

## Innate Universal Grammar

::: columns

:::: {.column width="20%"}

![](./figures_gpu/gpuday_arrow_3.png){height=68%}

::::

:::: {.column width="70%"}

\centering
![](./figures_gpu/gpuday_head_3.png){height=68%}

::::

:::
\small
By learning a language we learn the language's connection to the innate Universal Grammar over which we perform inference. [@deacon1997symbolic; @cook2014chomsky].

## Innate Mental Language

::: columns

:::: {.column width="20%"}

![](./figures_gpu/gpuday_arrow_4.png){height=70%}

::::

:::: {.column width="70%"}

\centering
![](./figures_gpu/gpuday_head_4.png){height=70%}

::::

:::
\small
Learning a language is a translation task from and to an inner mental language [@deacon1997symbolic; @pinker2003language].

\normalsize
## The AI Perspective

- Associative mappings are the closest to how LLMs are trained.
- Transformer circuits try to discover the "mental images" of trained models.
- Ongoing research is eager to incorporate exact "as Universal as possible" grammars into LLMs.
- Translation to and from an LLM's "mental" language is the hottest solution for modality extensions.

# LLMs: The Backbone

## Characteristics of LLMs

- Context-awareness: Attention mechanism, or similar techniques. Few-shot learning possible.
- Self-supervised learning: Using vast amounts of "unlabeled" data.
- Autoregressive generation: Modeling continuation probabilities over a sequence of symbols (tokens).
- Large-scale: $1-2000$B parameters.

## The Role of Dynamic Selection 

Attention learns a **dynamic** (based on $\mathbf{x}^*$) selection **mechanism** that is used to process each element of the input sequence $\mathbf{x}$. The dynamic selection works by calculating a vector dim. scaled dot-product relevance score between the input and the query after learnable linear projections ($\mathcal K$, $\mathcal Q$, $\mathcal V$) [@vaswani2017attention].

$$s(\mathbf{x}_i, \mathbf{x}^*) = \frac{\mathcal K (\mathbf{x}_i)\cdot \mathcal Q (\mathbf{x}^*)}{\sqrt{d}}$$ 
$$\mathop{\mathrm{softmax}}(\langle s(\mathbf{x}_1,\mathbf{x}^*),\dots,s(\mathbf{x}_n,\mathbf{x}^*) \rangle)\cdot \mathcal V(\langle \mathbf{x_1},\dots,\mathbf{x}_n)\rangle$$

## Self-supervision

The distributional semantic approach [@lenci2023distributional] assumes:

- Words that occur in similar contexts are semantically similar.
- The meaning of a word could be inferred from the context it appears in.

This context could be bidirectional (fill-mask style) or causal (autoregressive, predict the next style).

## The Language Recipe
\centering
![](./figures_gpu/gpuday_text_framework.png){height=80%}

## Specifics of a GPT-like Model

::: columns

:::: {.column width="70%"}

- Using Causal Multi-Head Attention to mix tokens.
- Feed-forward layers used to mix channels.
- Subword tokenization with Byte-Pair Encoding.
- Autoregressive with $k$-th order Markov assumption.
- @radford2019language

$p(x_1, ..., x_n) = \displaystyle \prod_{i=1}^n p(x_i|x_{i-k}, ..., x_{i-1})$

::::

:::: {.column width="30%"}

\centering
![](./figures_gpu/gpuday_gpt.png){height=90%}

::::

:::

## Alternatives

Selective (input-dependent $\mathbf{B}$, $\mathbf{C}$ and $\Delta$) State-Space Models [@gu2023mamba]

![S4 block with SRAM state caching.](./figures_gpu/mamba.png){height=80%}

## Alternatives

Retention with preset decay to construct dual-form (parallel, serial) networks [@sun2023retentive].

![Retention for training (left) and inference (right).](./figures_gpu/retnet.png){height=80%}

# How DL Research Benefits from LLMs?

## How to Handle a Giant?

![From @lifeofbrian1979](./figures_gpu/romans.jpg){height=75%}

## Preference Alignment

Alternatives are not learned due to:

- Data Sparsity (training on all 100K words long sequences is impossible).
- Teacher Forcing (the model is not incentivized to explore alternatives).

But we can do it in a second phase using sequence-level preference training based on a small dataset of human preference data.
This instruction fine-tuning produced ChatGPT as well.

## Instruction Fine-tuning

PPO-based RL with (reward, reference and policy LLM models) was the first breakthrough in human preference alignment [@ouyang2022training]. 

$\max_{\pi_\theta}\mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(y\mid{}x)} \left[ r_{\phi}(x, y) \right]-\beta\mathbb{D}_{\text{KL}} \left[ \pi_\theta(y\mid{}x) \parallel \pi_{\text{ref}}(y\mid{}x) \right]$

Later Direct Preference Optimization (DPO) was introduced that uses maximum likelihood-based training without a reward model [@rafailov2023direct].

$\max_{\pi_\theta}\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$

## Instruction Fine-tuning

Lately, even the reference model could be omitted by using Odds Ratio Preference Optimization (ORPO) [@hong2024orpo].

$\text{odds}_\theta(y \mid x) = \frac{1 - \pi_\theta(y \mid x)}{\pi_\theta(y \mid x)}$

$\max_{\pi_\theta}\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\log \sigma \left( \log \left( \frac{\text{odds}_\theta (y_w \mid x)}{\text{odds}_\theta (y_l \mid x)} \right) \right)$

## Instruction Fine-tuning

![PPO, DPO and ORPO compared in terms of the model versions used during the steps of alignment tuning [@hong2024orpo]](./figures_gpu/orpo.png){width=90%}


## Flash Attention

The HBM GPU memory's access is slow, use the SRAM cache instead [@dao2022flashattention]!

- Iterative processing of the QK product
- Parallelized softmax calculation
- Recompute intermediate values during backward pass
- In Flash Attention 2 [@dao2023flashattention2] GPU process scheduling is also optimized.

```python
torch.backends.cuda.enable_flash_sdp()
```

## Flash Attention

![Hierarchy of GPU memory and the benefits of an iterative fused kernel to reduce HBM access. From [@dao2022flashattention]](figures/flash_attention.png){width=100%}

## Adapters

:::: columns

::: column

- Full fine-tuning of a GPT-3.5 level model needs 520 GB of memory@fp16.
- Tuning the top layers is inefficient.
- Adapter methods add small trainable parameter sets to all layers of the model.

:::

::: column

\centering
![Parallel (mergable) low-rank adaptation (LoRA) method. LoRA's are portable. [@hu2022lora]](figures/lora_decomposition.png){height=50%}

:::

::::

## Ensembling

By combining models on the module level, ensembles, such as Mixtures of Experts (MoE) enable large, sparse models with data-specific experts [@chen2022understanding; @fedus2022switch].

\centering
![](./figures/moe_motivation.png){height=50%}

## Ensembling

![Switch Transformer from @fedus2022switch](./figures/switch.png){height=70%}

## Speculative Decoding

- Autoregressive predictions are guided by a smaller model (or medusa heads) [@xia2023speculative; @leviathan2023fast; @chen2023accelerating; @gante2023assisted; @cai2024medusa].
- Validation is done by the original model. 
- 2-8x speedup with effectively no loss in quality.

## Speculative Decoding

![Medusa head $k$ predicts the $1+k$-th token. Candidates are validated by the main LLM head in the next pass while generating the new candidates as well. [@cai2024medusa]](./figures_gpu/medusa.png){height=70%}

## In-context Learning

- Context information is used to adapt the model's behavior on the fly enabling zero-shot and few-shot learning.
- This opens up the possibility of input-tuning and answer-engineering (as a ML task).
- The context could be accessed from an external data source as well.
- Reasoning and planning (agents) are also possible.

## Agent-loops

![A ReAct-style agent observes the current state, reasons about it, generates a candidate action and reflectively improves it before execution [@yao2023react].](figures/agent_loop.png){height=70%}

# World Models and the Future

## Modality Extension

\centering
![](./figures_gpu/modalities.png){height=90%}

## Emerging Modality Connections

:::: columns

::: column
Aligning modality pairs $\mathcal{M}_i$ and $\mathcal{M}_j$ along a spanning tree of all modalities we get weakly aligned modalities for each $\mathcal{M}_i$ and $\mathcal{M}_{k\neq{}j}$ as well. 

Language is a good candidate for a modality that can form pairs with most other modalities.

:::

::: column

\centering
![Modality pairs with training data (solid) and without training data (dotted) from [@girdhar2023imagebind]](./figures/imagebind_pentagram.png){height=100%}

:::

::::

## Language as a Transporter of Meaning

![ImageBind retrievals of non-trivial modality pairs (with object detection in the visual modality) [@girdhar2023imagebind]](figures/imagebind_crossmod_2.png){width=90% align=center}


## The Large World Model Template

\centering
![](./figures_gpu/multimodal_arch.png){height=90%}

## LLava = LLama + Vision

- LLaVa uses an LLM + a CLIP-like vision encoder.
- It prepends a single image prefix to the text input and generates text.
- GPT-4V used a similar approach early 2023.

![LLaVA architecture from @liu2023visual.](figures/llava.png){width=80%}

## Interleaved Input & Proper Decoding

![By applying the corresponding encoders and decoders @tang2023codi2 train an any-to-any model.](./figures_gpu/codi.jpg){height=75%}

## LWMs in Action

LWMs are capable of **summarizing** lectures, **generating** toned audio responses, performing **speech recognition** at SOTA levels.

OpenAI [@OpenAI2024] and Google [@team2023gemini] each provide LWM services for development **beating single-modality models** in many tasks. Input and output streaming is also possible to reduce latency (taking timing information into account).

## LWMs in Action

![Video-based Q&A by @liu2024world](./figures_gpu/lwm_ex.png){height=100%}

## LWMs in Action

![Multimodal generation based on interleaved input sequences by @tang2023codi2](./figures_gpu/codi_ex.png){height=100%}


## And many more...

- Robot control [@embodimentcollaboration2024open]
- Action spaces & environment modeling [@bruce2024genie]
- Modelling priors for image generation [@ramesh2022hierarchical]
- Time Series [@das2024decoderonly]
- Motion [@jiang2023motiongpt]
- 2D-to-3D object generation [@xu2024instantmesh]

## What we lack

- Stronger Reasoning (avoiding hallucinations)
- Continual Learning (personalization, adaptation)
- Symbolic Logical Inference (e.g. for theorem proving)
- Massively Multimodal Models (for dozens of modalities)

Strong AI?

``` {=latex}
\end{frame}

\begin{frame}[noframenumbering]{}
\centering
\vspace{2cm}

\Large{\textbf{Thank you for your attention!}}

\vspace{2cm}

\includegraphics[width=\textwidth]{./figures_gpu/elte_logo.png}
\end{frame}
```

# References {.allowframebreaks}
\footnotesize