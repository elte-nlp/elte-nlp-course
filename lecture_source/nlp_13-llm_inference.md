---
title: "Natural Language Processing"
subtitle: "Lecture 13: LLM Inference"
author: Natabara Gyöngyössy
institute: "Eötvös University, Department of Artificial Intelligence"
date: 2024
theme: Marburg
colortheme: orchid
fontsize: 14pt
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
bibliography: nlp_13-llm_inference.bib
---

# Introduction

As language models become larger and larger, questions arise about how to use them effectively, how to make their answers more varied ("creative"), and how to prevent them from generating harmful content. In this lecture, we will discuss the following topics:

- Standard LLM inference methods and parameters
- "Edge" and server inference
- Assisted inference and speculation
- Guided text generation
- Inference-time model "adaptation"
- Watermarking

# Standard LLM Inference

## Recap: Sampling

- **Greedy decoding**: always choose the most probable token $w_t = \arg\max_{w} P(w|w_{t-c}, \ldots, w_{t-1})$
- **Random sampling**: sample from the distribution $P(w|w_{t-c}, \ldots, w_{t-1})$
- **Beam search**: keep the $b$ most probable sequences at each step
- **Stochastic beam search**: sample from the $b$ most probable sequences

## Probability distributions

Softmax is used to convert the model's logits to a probability distribution. In order to draw probability mass from the most probable tokens (thus making the model more "creative" and less repetitive), we can use temperature scaling:

$P(w|w_{t-c}, \ldots, w_{t-1}) = \frac{\exp(\textit{logits}(w) / T)}{\sum_{w'} \exp(\textit{logits}(w') / T)}$

where $T$ is the temperature parameter.

## Top-k sampling

Calculating the softmax over the entire vocabulary is expensive, low-scoring tokens are often not interesting, thus virtually restricting the vocabulary at each step to the top $k$ tokens is viable. This is called top-$k$ sampling.

## Top-p sampling

Nucleus sampling, or top-$p$ sampling, restricts the vocabulary in a different way: it keeps the smallest set of most probable tokens whose combined probability mass meets (and exceeds) a threshold $p$.

Given the following set of tokens and probabilities:
**apple** (0.3), **banana** (0.2), **cherry** (0.15), **date** (0.1), **elderberry** (0.1), **fig** (0.1), **grape** (0.05)
A top-$p$ with $p=0.6$ would keep **apple**, **banana**, and **cherry**.

## Logit biasing

We can also bias the logits of the model to favor certain tokens. This can be used to prevent the model from generating harmful content, or to make it generate content that is more aligned with a certain style. For example intent classification would benefit from adding a large positive offset to class-label tokens, while surpressing the probability of other tokens.

More complex cases include "presence" and "frequency" penalties, where the former decreases bias towards tokens that appear in the current text with a flat penalty, while the latter incrementally decreases it with the number of occurrences.

The formula applied differs from implementation to implementation, but penalties often use exponential forms.

## Beam size, best of N

The beam size $b$ is a hyperparameter of beam search. It is the number of sequences that are kept at each step. A larger beam size will result in more diverse outputs, but also in a significantly slower inference. Beams rank sequences by their commulated probability, and the best of $N$ sequences can be chosen at the end of the inference.

More aggressive methods include full restarts, where the inference is repeated N times from scratch and the best sequence is chosen.

# Efficient and Edge Inference

## CPU inference

Inference on a CPU is generally slow and memory-limited. To overcome these limitations current edge-computing libraries quantize the model's weights down to 4-bit integers (from 32-bit floats). This results in a significant speedup and a smaller memory footprint.

llama.cpp [@Llamacpp] is a popular CPU inference library for LLMs running on $C^{++}$ (with Rust variants also available). It is optimized to use CPU-based advanced vector operations. Notable achievements include running 7B GPT models on desktop CPUs, Raspberry Pi models, Apple Silicon, Android phones and also supporting desktop hybrid CPU-GPU inference.

Latest versions use `mmap` compatible memory mapping for the weights so they can be loaded on and off from disk on demand.

## Efficient GPU inference

Using GPUs for inference the limiting factor aside the memory capacity is memory bandwidth, as generating tokens one-by-one requires a lot of memory access operations. 

- To overcome this **caching** is possible, where we keep previously calculated key and value pairs in memory and reuse them for the next token generation.
- This way the query size is usually $1$ for each element in the batch, and due to the memory limitations batch size is usually kept low (lower than the number of processing units in the GPU). --> **Low GPU utilization**
- **Flash decoding** [@flashdec] parallelizes the $QK$ product calculation over the sequence length, softmax and the output are calculated after the parallel processing is done.

## Flash decoding vs Caching-only

GIFs about the difference between from [@flashdec]:

- [Caching and iterative decoding](https://crfm.stanford.edu/static/img/posts/2023-10-13-flashdecoding/parallelization.gif)
- [Flashdecoding](https://crfm.stanford.edu/static/img/posts/2023-10-13-flashdecoding/parallelization_kv.gif)

## Softmax problems

Calculating the softmax in attention scores could also be a bottleneck. LLMs usually the "max" trick to prevent overflow in the exponential ($\exp{x_i}\rightarrow\exp{x_i - \max{x}}$), but this includes a max calculation which is hard to parallelize. \emph{Flashdecoding++} uses an empirical trick. It uses a fixed global constant based on activation statistics to prevent the overflow, thus the elements of the softmax can be calculated in parallel. If the method meets an overflow it will recompute the softmax with the actual maximum value, but this should happen with $>1\%$ probability.

\emph{Flashdecoding++} [@hong2024flashdecoding] also upgrades General Matrix Multiplication (GEMM) with double buffering to account for memory latency at low batch sizes and heuristically selects the best implementation for the given LLM and batch size.

## Softmax problems

![Softmax calculation types from [@hong2024flashdecoding]](./figures/stable_softmax.png){ width=100% }

## Max attention values

![Max attention values from [@hong2024flashdecoding]](./figures/attn_score_distrib.png){ width=100% }

## Processing concurrent requests

Given a centralized inference server, we usually expect a larger number of multiple requests to be processed in parallel with as little latency as possible. High-performance inference consists of two phases:

- **Prefill**: The user prompt is processed, K and V are calculated and cached. This could be done in a single pass, and it might be a much longer sequence than the generated output. This also includes generating the first output token.
- **Decoding**: This is the iterative process of generating the next token and calculating the next K and V. This cannot be parallelized, but the K and V can be reused from the cache. We only need to calculate a single Q for each pass.

## Flashdecoding++

![Decoder inference from prefill and decode phases need a substantially different calculation method. [@hong2024flashdecoding]](./figures/flashdecpp.png){ height=70% }

## Problems with concurrent requests

- **Monolithic KV cache**: KV caches for long sequences could lead to memory fragmentation that could slow down inference and use memory inefficiently.
- **Short sequences**: Short sequences do not utilize the input size of the transformer, thus classically padding them would be a solution, which is wasteful in terms of memory and computation.
- **Varying prefill and decoding times**: We can estimate the prefill time for a given request however the decoding time is hard to predict. This could lead to blocking and bubble effects in the processing queue.
- **Suboptimal GPU utilization**: Using the correct batch vs sequence length ratio is crucial for efficient GPU utilization. We can not control the length of incoming requests.

## Solving cache problems

Cache paging is an efficient method inspired by virtual memory management that addresses a variety of caching problems, such as **memory fragmentation**, **unknown decoding length**, **shared sequence prefixes**.

Pages (small fixed size memory blocks) are used to store the KV caches in a way that logically contiguous sequences are stored in non-contiguous physical memory. \emph{PagedAttention} [@kwon2023efficient] is then utilized which is a partial attention (that can be parallelized in a \emph{flashdecoding++} like manner) with page-based indirection. vLLM is a framework that implements this method.

## Memory problems

![Memory problems from [@kwon2023efficient]. Preallocated but unused and fragmented sequences can cost 5-15\% of the GPU memory.](./figures/paged_attn_fragment.png){ width=100% }

## vLLM virtualization

![Virtualized cache handling [@kwon2023efficient]](./figures/vllm.png){ height=75% }

## Logical vs Physical Memory

![Logical vs Physical Memory [@kwon2023efficient]](./figures/paging.png){ height=75% }

## Solving cache problems

This method allows dynamic memory allocation and deallocation for decoding length variations, as well as sharing caches between sequences and getting rid of duplicates for inputs with the same prompt or beam search.

Shared prefixes are really common in chat models (typically each user interacts with the same system prompt). Hydragen [@juravsky2024hydragen] proposes further optimizations for not just caching but QK product calculation. By separately calculating the QK product for the prefix and the rest of the sequence (maybe in a separate pass) computation is saved and the prefix is read only once in the GPU working memory from the pages.

## Hydragen

![Hydragen's separate prefix and suffix calculation [@juravsky2024hydragen]](./figures/hydragen.png){ width=100% }

## Piggybacking, continuous batching

Small input sequences can be combined together to form a longer sequence with multiple partitions to calculate attention over (using masking for example). This way we can have multiple decoded tokens calculated in a single pass. This is called continuous batching or piggybacking.

Mixed prefill and decoding batches are also possible where one part is used to calculate KV caches, while the other is used for generating tokens. This is useful to elliminate bubble effects during decoding (where a long sequence processing must be finished before we can start working on the next task, thus the GPU is underutilized).

## Microbatching

Splitting long sequences into smaller parts and processing them in parallel, while stuffing as much decoding tasks into the continuous batch as possible (decode-maximal microbatching) is a good method to tackle bubbles that arise from different sequence lengths. This is what Sarathi [@agrawal2023sarathi] achieves. However, decoding times for different requests and the general difference in prefill and decode processing times could lead to bubbles that microbatching can not solve.

DeepSpeed-FastGen [@holmes2024deepspeedfastgen] also measures the optimal GPU throughput curves and uses this heuristic to find a context length that is optimal for a given LLM, batch size and GPU. This is usually around a few hundred tokens only.

## Bubble effects solved by Sarathi

![Bubble effects solved by Sarathi's efficient microbatching [@agrawal2023sarathi]](./figures/sarathi.png){ height=75% }

## GPU utilization

![GPU utilization curves for different context lengths, over ~400 tokens there is no advantage in throughput [@holmes2024deepspeedfastgen]](./figures/splitfuse_saturation.png){ height=75% }

## Mixing prefill and decoding

There is also a different limitation characteristic for each task:

- Prefill is compute limited
- Decoding is memory limited and latency critical

Joint optimization of them often results in interference (you cannot simultaneously optimize for memory access and compute). Solution: disentangle them, by having another abstraction layer that maps logical prefill and decode requests to different physical resources (GPUs).

GPUs are allocated to prefill or decode tasks based on the current load and the expected decoding lengths (solutions like TetrInfer [@hu2024inference] develop a length prediction model for this).

## Decoupled prefill and decode

![By using separate resources for prefill and decode we can optimize for both tasks differently. [@hu2024inference]](./figures/tetrinfer.png){ height=75% }

# Assisted Inference and Speculation

## Assisted inference

Assisted inference or speculative inference is a method where our large autoregressive model is guided by a smaller "draft" or "assistant" model. The idea is to run the assistant model autoregressively and generate a sequence of a few tokens and then run the original model for a single step. This way the original model evaluates the assistant's whole "speculation" in a single pass (we check the output of each newly added token not just the last one). [@xia2023speculative,@leviathan2023fast,@chen2023accelerating,gante2023assisted]


## Assisted inference
Possible outcomes are:

- The assistant model gets the first token wrong $\rightarrow$ the original model will correct that token.
- The assistant model gets some tokens right $\rightarrow$ the original model accepts them and corrects the first wrong token.
- The assistant model gets all tokens right $\rightarrow$ the original model accepts the whole sequence and generates the next token.

## Assisted inference

![Validating the assistant's output. Black is validated, green is accepted, red is rejected with blue as a correction. [@leviathan2023fast]](./figures/speculative_decoding.png){ width=100% }

Further video resource from @gante2023assisted: [LINK](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/assisted-generation/gif_4_1080p.mov)

## How many tokens to speculate?

- Given the number of speculated tokens $\gamma$,
- the acceptance rate of the assistant model's tokens $\beta$ over a given sequence,
- the expected acceptance rate of the model in general $\alpha = \mathbb{E}[\beta]$,
- the cost coefficient as $\frac{T_{\text{assistant}}}{T_{\text{original}}}$ for a single run

The model always generates between $1$ and $\gamma+1$ tokens. The average number of accepted tokens is $\frac{1-\alpha^{\gamma+1}}{1-\alpha}$. The expected cost for producing a token is then $(c\gamma + 1)T_{\text{original}}\cdot\frac{1-\alpha}{1-\alpha^{\gamma+1}}$. The total improvement is then $\frac{1-\alpha^{\gamma+1}}{(c\gamma + 1)(1-\alpha)}$.
Given enough memory to speculate and $\alpha > c$ we choose the integer $\gamma$ that maximizes the improvement.

## Speculative decoding results

Given a standard LLM such as Chincilla(70B) and a 4B assistant model, using $\gamma=3$ with $c=0.13$ the expected acceptance rate is around $\alpha=0.7$. The improvement this way is around $1.82$ (times faster inference).

T5 models could also be improved by this method where for English to German translation the XXL (11B) model could be assisted by the small (60M) model. Here the expected acceptance rate is around $\alpha=0.75$ and the improvement is around $3.4$ with $\gamma=7$.

Sampling usually reduces the acceptance rate but the improvement is still significant [@chen2023accelerating].


## Further directions

- **Blockwise decoding and validation**: Previous research suggests that with a small fine-tuning one can attach multiple output heads to a model that would not only predict the next token, but also the upcoming $k$ tokens. We can then assume that the $k$ tokens are generated by the assistant model and use the original output as the validator. The algorithm is then similar to the one described above but with a single model with multiple outputs [@stern2018blockwise].
- **Reusing**: Assistance can come from previously generated tokens as well. Such as the prompt itself, cached history, sessions from other users, etc. Validation is then done by the original model [@yang2023inference].

## Blockwise decoding heads

![Blockwise decoding creates multiple output heads that predict the $k$-th next token, they serve as the weakly trained assistants, while k=1 is the original validator. [@stern2018blockwise]](./figures/block_decode.png){ width=100% }

## Reusing tokens

![Three simple cases to reuse previously generated tokens (and KV caches) [@yang2023inference]](./figures/reference_cases.png){ height=75% }

# Guided Text Generation

## Guided text generation

Logit bias values can be set dynamically based on auxiliary scoring functions. This can be used to guide the model to generate text that is more aligned with a certain usecase, or to restrict the model's output to a certain style.

This could also include finite state machines, where the model's output dictionary is restricted according to the current state of the machine. If we construct the FSM according to for example a given regular expression we can guarantee that the model will generate a text that is aligned with the regex. [@willard2023efficient]

## FSM-based guidance

![A FSN-based guidance system over a small vocabulary. Black tokens are disallowed [@willard2023efficient]](./figures/fsm_sample.png){ height=75% }

## Classifier guidance

Classifiers are also viable guidance tools, where the output logits are biased towards the desired class using an auxiliary classifier. This is proven to be useful in harmful text avoidance. This would be highly unfeasible to do for all the vocabulary elements thus a smaller set of top scoring tokens are chosen for re-scoring [@sitdikov2022classifiers].

Small expert language models and expert + anti-expert pairs could also be used to output a logit distribution that is then used for guidance. In case of an expert + anti-expert pair we take the difference of the two models' logits and use that as the guidance signal [@liu2021dexperts].

## Expert guidance

![Expert guidance using a small expert and anti-expert model. The expert logit difference is used as the guidance signal [@liu2021dexperts]](./figures/expert_guide.png){ height=75% }

## Inference-time model adaptation

Proxy-tuning is a promising method that utilises expert and anti-expert models, where the expert is a small fine-tuned model, and the anti-expert is the original untuned version of the proxy. This way instruction fine-tuning and various alignment methods (filtering toxic responses, etc.) is possible, as well as improvement in the model's performance on downstream tasks. Proxy-tuning's performance shines in "communication style"-like tasks, but also improves in factuality and coherence. 

Proxy tuning stands out as it is *model agnostic* (only bound to the vocabulary), *portable* and reusable, *hardware efficient* (there is no need to finetune the large models which would be expensive), and *combinable*, as multiple experts and anti-experts can be used at the same time [@liu2024tuning].

## Proxy-tuning

![Proxy-tuning setup (similar to expert guidance and classifier-free guidance) [@liu2024tuning]](./figures/proxy_tune.png){ height=75% }

## Proxy-tuning results

![Proxy-tuning results on various tasks. The improvement is significant for alignment tasks. Using the tuned proxy is less beneficial than using a proxy-tuned large model. [@liu2024tuning]](./figures/proxy_tune_res.png){ height=75% }

# Watermarking

## Why do we need watermarking?

As LLMs are becoming increasingly performant, the need for accountability and detectability of a model's output is rising. Watermarking [@kirchenbauer2023watermark] is a method where we embed a unique patter in the model's output so that:

- It has negligible effect on the model's performance (not detectable by humans)
- It is easy and fast to verify
- We don't need to know the model's parameters to verify it
- It works on a relatively small set of tokens
- It is not too easy to remove (partial removal or modification should still be detectable)
- There is no need to retrain the model

## A hard red-list strategy

Red-green list strategies are a simple way to watermark a model. During each step of the inference we select a set of red-list tokens that are not allowed to be generated (the rest are green-list tokens). This way we can watermark the model's output with a unique pattern as follows:

1. We take the last token at $t-1$ and using a hash function produce a random seed.
2. We use the seed to generate random numbers that separate the vocabulary into red and green halves.
3. We sample a token from the green list using the LLM's logits.

## Detecting a hard red-list watermark

A baseline detection method is to start out from the fact that without the knowledge of the red-list tokens, the probability of generating a $T$ long sequence without violating the red-list is $\left(\frac{1}{2}\right)^T$. This is a very low probability for even a short sequence.

A more sophisticated method is to use a z-test on the following H0 hypothesis:
\emph{The text sequence is generated with no knowledge of the red list rule.}

Given that the number of green-list tokens is $G$ and the expected value for it is $0.5T$ with $0.25T$ variance we can calculate the z-score as:

$$z = 2(G-0.5T)/\sqrt{T}$$

## Detecting a hard red-list watermark

We reject the null hypothesis if $z$ is above a given threshold. Authors suggest using $z>4$ for this rejection as in this case a false-positive rate is $3\cdot10^{-5}$, and we detect all the watermarks with $16$ or more tokens.

Given a token flip by an adversary results in a violation at $t$ and also at $t+1$ in worst case as the hash function depends on the previous token.

This means that for $T=1000$ tokens modifying $200$ tokens results in at most $400$ violations, for this the $z$ score is still around $6.3$.

Removing the watermark generally needs a modification in at least $25\%$ of the tokens.

## Low entropy sequences

Low entropy sequences (where the model's output is highly predictable) are problematic from the perspective of hard watermarking. 

First of all there is a high chance that a human would also generate the same sequence (such as following up \emph{Barack} with \emph{Obama}). Watermarking such sequences is counterproductive.

Second, hard watermarking usually distrupts such sequences, as the high probability tokens could fall into the red list.

## Soft watermarking

A solution for the low entropy sequence watermarking problem could be soft watermarking, where the green-list tokens gain only a small (and not a complete) advantage over the red-list tokens.

We add a small $\delta$ to the logits of the green-list before applying the softmax. This way the green-list tokens gain a relatively high advantage when the entropy is high, but in case of a low entropy a single best token with $p\sim1$ has no disadvantage even if it is in the red list.

As another extension we can choose a fraction $\gamma$ of the vocabulary to be green-list tokens. This is a hyperparameter that is usually kept at $0.5$.

## Detecting a soft watermark

Detecting a soft watermark is more difficult than detecting a hard one. The z-test still holds:

$$z = (G-\gamma T)/\sqrt{T\gamma(1-\gamma)}$$

The false positive rate is still low, however the detection rate erodes for low entropy sequences.

The worst-case perplexity increase for maximally deviated distributions at a given token's generation is $(1+(e^\delta-1)\gamma)P_{\text{original}}$ (which is approximately $4$ for $\delta=2$ and $\gamma=0.5$).

## Soft watermark erosion

The watermark is weak when the logit distribution has a spike concentrated on a few tokens only. 

In case of average entropy sequences the watermark is still detectable $98.4\%$ of the time in $T=200$ tokens with $\gamma=0.5$ and $\delta=2$.

The detection erodes for low entropy sequences. This is the case for repeated specific text and memorized sequences, where the model essentially reproduces the exact same text it has seen before.

Repetitive text can be accounted for by including only the first occurence of the n-gram in the z-test, or by using more previous tokens to calculate the hash function (thus the red list will not be the same for all shorter n-grams).

## Private watermarking

Infering the watermarking method from the detector's decision is possible by submitting $|V|^h$ tokens to the detector, where $h$ is the number of tokens used in the hash function. To counteract decyphering we can use a larger $h$, but that would also introduce difficulties to detection as flipping a single token could affect the following $h$ tokens and corrupt $0.5h$ tokens on average.

Using more sophisticated methods that depends on the current token and one of the $h$ previous tokens is also a viable solution, where the error rate decreases to $1/h$.

Private watermarking is also possible with a cryptographic hash function that has a secret key such as AES or SHA3. This way the adversary cannot detect the watermarks without the knowledge of the key.



# References {.allowframebreaks}
\small
