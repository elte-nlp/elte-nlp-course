---
title: "Natural Language Processing"
subtitle: "Lecture 13: Prompt engineering"
author: András Simonyi
institute: "Eötvös University, Department of Artificial Intelligence"
date: 2023
theme: Marburg
colortheme: orchid
fontsize: 14pt
linkcolor: blue
header-includes:
  - \let\emphasized\emph
  - \let\strong\textbf
  - \renewcommand{\textbf}[1]{\textcolor{blue}{\strong{#1}}}
link-citations: true
aspectratio: 1610
output: 
  beamer_presentation:
    slide_level: 2
bibliography: nlp_14-prompt_engineering.bib
---

# Pretrain, prompt and predict


## Pretrain, prompt and predict

The paper describing GPT-3 [@brown2020language]  introduced a new paradigm of using
pretrained LMs for downstream tasks: just prompt the model suitably, and map the
output onto the task's output domain without any fine-tuning. They
distinguish 3 scenarios:

![img](./figures/zero-shot.png){width=70%}

## Pretrain, prompt and predict cont.

![img](./figures/one-shot.png)

## Pretrain, prompt and predict cont.

![Few-shot or in-context learning [from @brown2020language].](./figures/few-shot.png){width=80%}

## Pretrain, prompt and predict cont.

An important feature of the the prompting paradigm is that task performance is
very sensitive to the prompt's details:

-   example selection,
-   example ordering, and
-   task formulation

can all have a huge effect, because the models have various biases, among them:

-   **majority label** (imbalance) bias,
-   **recency** bias (later label(s) are more influential), and
-   **common token** bias (more common tokens are more probably predicted).


## Prompt engineering

The listed biases (and others) make it imperative to **optimize** the prompt used
for a large language model-based, zero- or few-shot solutions for a task, that is,
to use suitable **prompt engineering** methods.



# Task formulation


## Prompt mining

Given a supervised dataset $\{\langle x_i, y_i \rangle\}_{i=1}^{N}$, one can
take a corpus (e.g., Wikipedia) and **search for words or syntactic constructs
that connect** the $x$-es with the corresponding $y$-s. Variants [the
examples are from @jiang2020can]:


###  Middle-word prompts
*Barack Obama was born in Hawaii* $\Rightarrow$ [x] *was born in* [y]

###  Dependency-based prompts\

Take the minimal span containing the shortest dependency path between $x$ and $y$:
    
France $\xleftarrow{pobj}$ of $\xleftarrow{prep}$ capital $\xleftarrow{nsubj}$ is 
    $\xrightarrow{attr}$ Paris	$\Rightarrow$\
	*capital of* [x] *is* [y]

## Prompt paraphrasing

Starts with a **seed prompt** and generates candidate prompts by **paraphrasing** it
(e.g., by translation and back translation).

::: {.block}
### Example

seed: [x] *shares a border with* [y] $\Rightarrow$\
[x] *has a common border with* [y]
  
$\vdots$
   
[x] *adjoins* [y]
::: 

[Example from @jiang2020can.] The optimal prompt then can be selected
 by choosing the candidate which performs best on the target task's training
 data.


## Gradient-based search

Builds a prompt template consisting of **trigger tokens**, e.g., the AutoPrompt
algorithm [@shin2020autoprompt]:

![img](./figures/autoprompt.png)



## Gradient-based search cont.

The tokens are found by an algorithm related to coordinate-descent:

1.  Initialize a starting list of length $L$ filled with mask tokens.
2.  For each $i \in 1 \dots L$ token position:
    * compute the $\mathbf{g}$ gradient of the log-likelihood of the training
      data for the token embedding in the position,
    * select the top $k$ words with the closest embeddings to $\mathbf{g}$
      [@shin2020autoprompt uses dot-product as a metric],
    * of these, select the one with the largest log-likelihood and replace the
      current token in position $i$ with it.

###  
This obviously assumes that the gradients are accessible, although
doesn't require __changing__ the parameters.

## Prompt generation

One can treat prompt generation as a conditional text generation problem and use
a standard seq2seq model to solve it. E.g., @gao2020making uses a 
pretrained T5 to generate prompt candidates having high log-likelihood on the
dataset (using beam search) and then fine-tune them (presumably using
gradient-based search):

![img](./figures/generation.png){width=70%}

## Prompt scoring

::: columns

:::: column

Finally, the BERT-based common sense knowledge extractor of
@davison2019commonsense is based on a set of hand-engineered prompt templates,
but for any concrete data point selects the template instantiation which has the
highest probability according to a second, pretrained unidirectional language
model (measuring "coherence").

::::

:::: column

![From @davison2019commonsense.](./figures/scoring.png){width=100%}

::::

:::


# Example selection


## Similar examples in embedding space

@liu2021makes chooses the most similar examples from the training data
for few-shot prediction:

![img](./figures/kate.png)\

## Contrastive learning

@rubin2022learning rely on contrastive learning to find the most useful examples.
The proposed method is to

-   Use a (typically smaller) scoring $LM$ to find positive and negative $( e
      , x)$ pairs in the training data, where the score is simply $P_{LM}(y | e,
      x)$.
-   Using contrastive learning train a metric embedding model that can be used to
    assign a score to *any* (example, $x$) pair.
-   For any $x$, retrieve positive and negative examples that contain the top and
    bottom $k$ scoring examples according to the model and use those in the
    few-shot prompt.



# Continuous ("soft")
## Prefix tuning

Learns task-specific embedding vector sequences to be prefixed to the actual
input (and output for encoder-decoders) embeddings [@li2021prefix]:

![img](./figures/prefix_tuning.png)



## Prefix tuning cont.

-   The vectors are fine-tuned simply using a log-likelihood objective on the training set.
-   The authors experimented with treating only the **input** embeddings of the
    prefix as learnable parameters vs. the prefix embeddings in **all layers** and
    the latter approach led to radically better results.
-   The method was performing similarly to full fine-tuning.



## Prefix tuning variants

Variations on the continuous prefix tuning theme:

-   **Discrete initialization**: Instead of random initialization,
    the optimization can start from an (automatically or manually) created
    discrete prompt for the task.
-   **Discrete-continuous hybrid tuning**: It is also possible to fix some discrete
    parts of the prompt (using "anchor tokens") and treat only the rest of the
    prefix as learnable parameters.
-   **Auxiliary network**: It turned out to be very useful to model the interactions
    between the prefix embeddings using (relatively) simple networks, e.g., LSTMs.



# Answer engineering
## Answer engineering

The mapping from LM outputs to the output domain of the downstream task can also
be optimized.

\smallskip

Depending on the architecture and the task, the output to be
mapped can be a

-   **Token**: this is a frequent choice for classification tasks.
-   **Span**: containing a few tokens, typically used with "cloze prompts".
-   **Sentence**: natural choice for language generation tasks.

\bigskip

Using the LM output "as is" can work for some tasks, e.g., for text generation,
but a mapping is needed when the $\mathcal{Y}$ output space is different or constrained,
e.g., for classification  or NER tasks.



## Answer engineering cont.

A trivial mapping example: a $v(\cdot)$ "verbalizer" function maps the
downstream topic classication task's class labels to answer tokens.

![From @schick2020few](./figures/verbalizer.png)

## Answer engineering cont.

Methods for finding suitable answer sets corresponding to each $y\in \mathcal Y$
include

###  Answer paraphrasing

A manually engineered seed answer set for $y$ is extended with paraphrases.

### Prune-then-search

An initial set is created, e.g., by paraphrasing, and this set is searched for
the optimal answer for $y$, e.g., by choosing the alternative with the largest
log-likelihood on the training dataset.

# Combining prompts
## Prompt ensembling

Like model ensembling, combining the LM's answers to multiple
*unanswered* prompts for the same $x$ input can lead to better or more stable
performance. The combination method can be

-   **uniform averaging**: the answer probability distributions to the combined
    prompts are simply averaged;
-   **weighted averaging**: the final distribution is the weighted average of the
    answer distributions -- weights can come from the prompts' performance
    on the training dataset;
-   simple **majority voting** can also be used for classification.

Combining prompts for text generation is not so straightforward, but one way of
doing it is to use the average of all next word probability distributions for
generating the next word at every generation time step.

## General instruction prompting strategies

-   Instructions should be as detailed, specific and precise as possible;
-   Specifying the *intended audience* of the output, if applicable, is typically useful;
-   Complex instruction prompts can also include
    -   persona description,
    -   in context examples,
    -   constraints (e.g., template for the expected output format),
    -   required steps for solution &#x2013; this leads to "Chain of thought prompting".



# Thought-structure based prompting 

## Chain of thought prompting

For tasks involving complex reasoning, e.g. math problem solving or planning,
providing step by steps demonstrations can hugely improve performance. E.g.,

### 
\footnotesize
Question: Tom and Elizabeth have a competition to climb a hill. Elizabeth takes
30 minutes to climb the hill. Tom takes four times as long as Elizabeth does to
climb the hill. How many hours does it take Tom to climb up the hill?

Answer: It takes Tom 30*4 = 120 minutes to climb the hill. It takes
Tom 120/60 = 2 hours to climb the hill. So the answer is 2.

&#x2014;

Question: Jack is a soccer player. He needs to buy two pairs of socks and a pair
of soccer shoes. Each pair of socks cost $9.50, and the shoes cost $92. Jack has
$40. How much more money does Jack need?

Answer: The total cost of two pairs of
socks is $9.50 x 2 = 19. The total cost of the socks and the shoes
is $19 + $92 = 111. Jack need $111 - $40 = $71 more.
So the answer is 71.

&#x2014;



## Chain of thought prompting cont.


::: {.block}
#### 
\small

Question: Marty has 100 centimeters of ribbon that he
must cut into 4 equal parts. Each of the cut parts must be divided into 5 equal
parts. How long will each final cut be?\
Answer:

\normalsize

:::

[The example is from @weng2023prompt.]
Even more surprisingly, "zero-shot chain of thought", without examples also
works [the example is also from @weng2023prompt]:

### 
\small Question: Marty has 100 centimeters of ribbon that he must cut
into 4 equal parts. Each of the cut parts must be divided into 5 equal parts.
How long will each final cut be?

Answer: Let's think step by step.

## Self-consistency sampling for COT

Results can often be improved by sampling several answers, i.e., several
reasoning paths instead of a single, say, greedy, decoding, and ensembling the
results, e.g., by taking the majority vote.

## Self-ask

Prompting the model to explicitly ask and answer follow up questions is also a
useful strategy [@press2023measuring]:

![img](./figures/self-ask.png){width=70%}

# References

## References {.allowframebreaks}
\footnotesize
