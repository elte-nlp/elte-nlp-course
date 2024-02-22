---
title: "Natural Language Processing"
subtitle: "Lecture 9: RNNs and language modeling"
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
  - \usepackage{tikz}
  - \usepackage{tikz-qtree}
  - \tikzset{every tree node/.style={align=center,anchor=north}}
link-citations: true
aspectratio: 1611
output: 
  beamer_presentation:
    slide_level: 2
---
# Introduction

## Language modeling with NNs

As we have seen, feedforward NN language models with learned word embeddings
already performed better than traditional $n$-gram models:

![The neural language model of  @bengio2003neural.](figures/neural_lm.eps){width="50%"}

## Language modeling with NNs cont.

@bengio2003neural reports 24% improvement in perplexity compared to
the best $n$-gram model.

But these models still share an important limitation of $n$-gram models:
the continuation predictions are based on a *fixed size context window*,
without any information on earlier history:
$$\hat P(w_{t}~|~w_0,\dots,w_{t-1}) = \phi(w_{t-k},\dots, w_{t-1}),$$
where $\phi(\cdot)$ is a function computed by a feedforward neural
network.

## Recurrent Neural Networks (RNNs)

*Recurrent neural networks*, in contrast, are not limited to
fixed-length input sequences, and can form, at least in theory, useful
internal representations of *arbitrary long* histories. They can process
sequential input step-by-step and keep an internal state which can be
updated at each step:

![The operation of an RNN [from
@olah2015understanding].](figures/RNN-unrolled.eps){width="100%"}

## Recurrent Neural Networks cont.

RNNs can be rather simple, e.g., the once widely used Elman network
[-@elman1990finding] has the structure
```{=latex}
\begin{center}
```
![image](figures/elman.eps){width="40%"}\
```{=latex}
\end{center}
```
$$h_t = a_h(U x_t + W h_{t-1} + b_h ),$$ $$o_t = a_o(Vh_{t} + b_o ).$$

## Backpropagation through time

The standard optimization method for RNNs is *backpropagation through
time* (BPTT), which is backpropagation applied to the time-unfolded
network:
```{=latex}
\begin{center}
```
![image](figures/rnn_unrolling.eps){width="95%"}\
```{=latex}
\end{center}
```

## Backpropagation through time cont.

Since the depth of an unrolled RNN grows linearly with the number of
time steps through which it is unrolled, it is often unfeasible to do
backpropagation through all time steps until the first one.

In these cases, unrolling and backpropagation of error is only done for
a certain number of time steps -- **backpropagation is truncated**. In
practice, most neural network frameworks implement truncated
backpropagation.

## RNN training challenges

Training RNNs poses significant challenges:

-   An RNN unrolled through several timesteps is behaving like a deep
    feedforward network with respect to backpropagation, so both
    **vanishing** and **exploding gradients** can be a problem, exacerbated
    by the fact that the exact same layers are repeated.

-   Vanishing gradients, in particular, mean that the RNN does not learn
    **long-term dependencies**, which, in theory, should be its strength.

# Long Short-Term Memory Networks

## Long Short-Term Memory (LSTM)

@hochreiter1997long introduced an elaborate gated topology to endow
RNNs with long-term memory and solve the vanishing/exploding gradients
problem.

![LSTM architecture [from @olah2015understanding].](figures/LSTM3-chain.eps){width="100%"}

## Cell state

The LSTM's cell state acts as an "information conveyor belt", on which
information can travel across time steps.

![LSTM cell state [from @olah2015understanding].](figures/lstm_c_line.eps){width="65%"}

## Forget gate

The forget gate calculates an $f_t\in (0,1)^d$ mask for removing
information from the cell state:
$$f_t=\sigma(W_f[h_{t-1}, x_t] + b_f).$$

![LSTM forget gate [from @olah2015understanding].](figures/lstm_forget.eps){width="57%"}

## Input gate and update vector

An $i_t$ input mask and a $\tilde C_t$ update vector is calculated:
$$i_t=\sigma(W_i[h_{t-1}, x_t] + b_i),$$
$$\tilde C_t = \tanh(W_C[h_{t-1}, x_t] + b_C).$$

![LSTM input gate [from @olah2015understanding].](figures/lstm_update.eps){width="57%"}

## Computing the new cell state

The new cell state is computed using $f_t, i_t$ and $\tilde C_t$:
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde C_t.$$

![LSTM cell state update [from @olah2015understanding].](figures/lstm_c.eps){width="57%"}

## Output

Finally, an output, $h_t$ is generated:
$$o_t = \sigma(W_o[h_{t-1}, x_t] + b_o),$$
$$h_t = o_t \odot \tanh(C_t).$$


![LSTM output generation [from @olah2015understanding].](figures/lstm_out.eps){width="57%"}


## LSTM advantages

The gated LSTM architecture solves the problem of vanishing/exploding
gradients by ensuring that the gradient can flow to distant time steps.

The fact that the updates are *additive* means that gradients are not
multiplied as in the Elman network's case, and the gates can acquire
weights during training that allow the network to exhibit long-range
dependencies between input and output values.

## LSTM variants: Peephole connections

Peephole connections extend the LSTM architecture by giving access to
the actual cell state to the gates:

![LSTM peepholes [from @olah2015understanding].](figures/peepholes.eps){width="65%"}

## LSTM variants: Gated Recurrent Unit 

Gated Recurrent Unit (GRU) is a simplified LSTM variant, it gets rid of the
separate cell state and merges the forget and input gates:

![Gated Recurrent Unit (GRU) [from @olah2015understanding].](figures/gru.eps){width="65%"}

# Language modeling with RNNs

## Language modeling with RNNs

![RNN-based LM architecture.](figures/rnn_lm.eps){width="70%"}

## Language modeling with RNNs cont.

The most notable features of the model are

-   previous words ("left context") are processed step-by-step, one word
    at a time step;

-   the first layer is a static word embedding;

-   the $h_t$ RNN direct output (hidden state) gets transformed to a
    continuation probability distribution over the vocabulary by an
    affine transformation and the $\mathop{\mathrm{softmax}}$
    nonlinearity.

## Sequence elements

Although traditionally RNN language models were word based, i.e., the
sequence elements were words, there are two important alternatives:

-   __*character-level*__ language models treat characters as the sequence
    elements, and predict the next character based on the previous ones.

-   __*subword-level*__ language models are based on subword tokenization
    (e.g., BPE) and predict the next *subword* in the vocabulary.

Both types of model can utilize corresponding -- character- and subword-
-- embeddings.

## Training

RNN-based language models, as all parametric language models, are
trained using the usual negative log-likelihood loss: if the training
sequence is $\langle w_1,\dots, w_n \rangle$ and $\hat P_i$ is the
model's output distribution for the $i$th continuation probability, then
the loss is

$$- \sum_{i=1}^n \log \hat P_i(w_i).$$ But what should the *input* of
the RNN be at each time step during training? Should it come from the
training data, or from the RNN's previous prediction?

## Training cont.

RNN language models are typically trained using the training data as
input. This is called *teacher forcing*.

![Using the models own output vs. teacher forcing.](figures/teacher_forcing.eps){width="50%"}

## Exposure bias

Although teacher forcing is by far the most used training method, it has
a major problem, the phenomenon called __*exposure bias*__:

-   Language models trained with teacher forcing are only exposed to
    situations in which the entirety of their input comes from the
    training corpus.

-   During *inference*, in contrast, they have to produce continuations
    for texts not in the training data, most importantly, during text
    generation they have to continue *their own output*.

## Exposure bias: solutions

-   __*Scheduled sampling*__ [@bengio2015scheduled]: randomly choose at each time
    step between using the training data as input or sampling from the model's
    prediction. The probability of choosing from the training set starts from
    1.0 and is slowly decreased during training.
-   __*Differentiable sampling*__: In original scheduled sampling the error
    was not backpropagated through the used sampling operation, because
    it was undifferentiable. In response, alternative sampling solutions
    have been developed that are differentiable, the most important is
    using the so-called Gumbel softmax reparametrization
    [@jang2016categorical].

## Multiple RNN layers

Modern RNN-based architectures frequently stack multiple RNN cells on
top of each other as layers, analogously to multi-layer feedforward
networks:

![Stacked unidirectional (left) and bidirectional (right ) LSTM
layers.](figures/lstm_layers.eps){width="1.\\textwidth"}

## Performance

Before the appearance of transformers, LSTM-based language models
performed consistently better than other architectures, and they are
pretty competitive even now.

On 5 of the 9 language modeling datasets tracked by
[NLP-progress](http://http://nlpprogress.com), models based on an
LSTM-variant, the so-called Mogrifier LSTM have the best performance,
and LSTM-based models are very close to the (transformer produced)
state-of-the-art on 3 of the 4 remaining datasets.

# References

## References 

\footnotesize

