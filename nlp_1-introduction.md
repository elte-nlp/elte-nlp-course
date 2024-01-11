---
title: "Natural Language Processing"
subtitle: "Lecture 1: Introduction"
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
aspectratio: 1610
output: 
  beamer_presentation:
    slide_level: 2
---
## Course textbooks

-   Dan Jurafsky and James H. Martin,\
    *Speech and Language Processing* 3rd ed.\
    Draft available at
    [https://web.stanford.edu/$\sim$jurafsky/slp3](https://web.stanford.edu/~jurafsky/slp3).

-   Jacob Eisenstein,\
    *Natural Language Processing*.\
    Available at <https://github.com/jacobeisenstein/gt-nlp-class>.

These slides are in large part based on the first, \"Introduction\"
chapter of the Eisenstein book.

## What is Natural Language Processing? 

__NLP__ is an interdisciplinary field concerned with making natural languages
accessible to computers.

-   __Natural language__ in this context means the ordinary languages humans
    use to communicate in speech or writing, e.g., English, Chinese,
    Spanish etc.
-   Being able to __access__ natural language covers a wide range of
    capabilities, with important areas being

    -   __communication__: the ability to accept input and produce output in
        natural language;
    -   __understanding__: being able to access and utilize the
        informational and emotional content;
    -   __linguistic assistance__: the ability to help humans to express
        themselves linguistically.


# Related fields

## Computational linguistics 

The scientific study of language using computational methods.

-   Perhaps the closest field to NLP, but with a different focus: NLP is
    *not* concerned with theoretical insights into natural languages
    *per se* but only with the design and analysis of methods useful for
    computational language processing.

-   Instead of direct implementations of theoretical ideas, it often
    provides __architectural inspiration__ for NLP systems.


## Artificial intelligence (AI) 

There is obviously a large overlap between the NLP objective and AI's goal of
building intelligent systems:

-   language use is strongly interdependent with the conceptual,
    representational and reasoning capabilities required for being
    intelligent,

-   in practice, large-scale knowledge acquisition is also impossible
    without the ability to extract information from natural language
    input.

The above characteristics make especially the AI subfields __Knowledge
Representation__ and __Reasoning__ very relevant for NLP.


## Machine learning 

Modern NLP relies on machine learning techniques to a huge extent, in fact, in
recent years the linguistic applications of general ML methods have dominated
the area.

-   Mostly supervised or semi-supervised methods are utilized, but the
    use of reinforcement learning is also increasing.

-   Texts are sequences of discrete symbols, so ML models capable of
    dealing with this type of input (and output in case of generation)
    are needed.


## Speech processing 

The processing and generation of acoustic speech signals is traditionally not
considered part of NLP, which is concerned primarily with __texts__, but is
obviusly closely related:

-   Speech2text provides input for NLP applications,

-   NLP apps provide input for speech synthesis;

-   Both processing and synthesizing speech requires linguistic
    knowledge that is also relevant for NLP: especially __language
    modeling__ has a central role in both areas.


# Applications

## Application examples

-   __machine translation__,
-   __document retrieval__: retrieving free-text documents matching a user
    query,
-   __question answering__, e.g., smart phone assistants' ability to answer
    questions,
-   __text classification__, e.g. detecting e-mail spam,
-   __chatbots__, e.g., a chatbot for buying a train ticket,
-   __spell checking__ and __grammar checking__,
-   __auto-completion__ for free-text input,
-   __document summarization__,
-   __text generation__ from structured data (from stock exchange news to
    error messages).

# Central themes

## Pipeline vs end-to-end architectures 

An influential view of NLP considers its core task to provide a __pipeline of
modules__ that successively produce general-purpose linguistic analyses, each
module building on the outputs of the previous ones:

![From the [documentation of the spaCy NLP
library](https://spacy.io/usage/processing-pipelines).](figures/pipeline.eps){width="100%"}

Specialized NLP applications are then built as relatively simple
additions on top of elements of this universal pipeline.


## Pipeline vs end-to-end cont. 

The opposite view concentrates on building NLP applications as __end-to-end__
machine learning models that learn to transform the raw input to the required
output without specialized linguistic analyzer modules.

State-of-the art NLP applications frequently fall in between these two
extremes: they use some universal analyzer modules, e.g., for word
segmentation or stemming, and also rely on ML models that skip some of
the traditional pipeline steps to produce the required output.


## Transfer learning 

An interesting, relatively recent, development is the appearance of neural
models that are end-to-end pretrained on unsupervised tasks on very large text
collections, and can act as a replacement for the traditional processing
pipelines:

-   Specialized models can be built by adding a few very shallow layers
    to the architecture but keeping the pretrained weights perhaps with
    a bit of fine-tuning.

-   It seems that some components of traditional pipeline have neural
    analogues in these models: certain layers seem to learn (more)
    morphology, others semantics etc.


## Learning and search 

A large number of supervised NLP tasks which we will encounter can be formulated
as an optimization problem of the form $$\hat y = \mathop{\mathrm{argmax}}_{y\in
Y(x)}\Psi_\theta(x, y)$$ where

-   $x\in X$ and $Y(x)$ are the task's input and potential outputs,

-   $\Psi_\theta: X\times Y \rightarrow \mathbb R$ is a scoring function
    or model that assigns scores to $\langle x, y \rangle$ input-output
    pairs and is parametrized by a $\theta$ vector, and

-   $\hat y$ is the predicted output.


## Learning and search cont. 

For instance,

-   $X$ could contain movie reviews and $Y$ the sentiment labels
    [Positive]{.smallcaps}, [Negative]{.smallcaps} and
    [Neutral]{.smallcaps}, and $\Psi_\theta$ could be a function
    assigning probabilities to the possible sentiment labelings of the
    reviews.

-   Also, $X$ could be the set of German texts and $Y$ their potential
    English translations, with $\Psi_\theta$ assigning translation
    quality scores to the candidates.


## Learning and search cont. 

This formulation makes it possible to factorize the problem into two
optimization subproblems solved by two distinct modules:

-   __Learning__: Finding the optimal $\theta$ parameters. This is typically
    done by optimizing $\theta$ on a large supervised data set
    $\{\langle x_i, y_i \rangle\}_{i=1}^N$ using numerical optimization
    methods.

-   __Search__: Finding the best scoring $y$ for a specific $x$, i.e.,
    computing the value of the $\mathop{\mathrm{argmax}}$ in the
    formula. Since the search space $Y(x)$ is often large because the
    potential $y$s have a complex structure (think, e.g., of a parse
    tree), this problem frequently requires combinatorial optimization.


## Semantic perspectives: relational 

Consider the utterance

\bigskip

> *My uncle's bought a cat. He's perhaps the most obnoxious animal I've ever
> met.*


How do we know that "animal" is said about the mentioned cat? One factor
is that we know that *cat* is a subcategory of *animal*: they are
connected by the [is_a]{.smallcaps} relationship.

The __relational perspective__ concentrates on these semantic/conceptual links
between the senses of expressions, which together constitute semantic networks:


## Semantic perspectives: relational

Lexical semantic ontologies like [WordNet](https://wordnet.princeton.edu/) and
[FrameNet](http://framenet.icsi.berkeley.edu/) are attempts to enumerate the
semantic relations between a large number of word senses.


![A semantic network fragment ([Wikipedia: Semantic
Networks](https://en.wikipedia.org/wiki/Semantic_network)).](figures/semantic_net.png){width="70%"}

## Semantic perspectives: compositionality 

The relational view sees word meanings as atomic nodes in a network. The
__compositional perspective__, in contrast, analyzes an expression's meaning
according to its internal composition.

E.g., the decomposition

*un$\vert$bear$\vert$able$\vert$s*

allows us to see the meaning of *unbearables* as being composed of the
meanings of its parts *un*, *bear*, *able* and *s*.

## Semantic perspectives: compositionality 

The principle of compositionality:

\bigskip

> *The meaning of a complex expression is determined by the meanings of
> its constituent expressions and the rules used to combine them.*[^1]

The principle can be applied to larger linguistic units than words:
sentences or even paragraphs etc. 

One (traditional) approach is to represent meanings with logical formulas and
associate syntactic rules of combination with semantic/logical ones:

## Semantic perspectives: compositionality cont.

```{=latex}
  \begin{center}
    \Tree 
    [.{\textit{John visits Julie} (S)}
        [.{\textit{John} (NP)} ]
        [.{\textit{visits Julie} (VP)} 
             [.{\textit{visits} (VT)} ] 
             [.{\textit{Julie} (NP)} ] ] ]
  \end{center}

\bigskip
  
  \begin{center}
    \Tree 
    [.{\textsc{visits}(\textsc{john},\textsc{julie})}
        [.{\textsc{john}} ]
        [.{$\lambda x.$\textsc{visits}$(x,$\textsc{julie}$)$} 
             [.{$\lambda y.\lambda x.$\textsc{visits}$(x, y)$} ] 
             [.{\textsc{julie}} ] ] ]
  \end{center}
```
  
## Semantic perspectives: distributional 

What does "bardiwac" mean?[^2]

-   He handed her a glass of __bardiwac__.
-   Beef dishes are made to complement the __bardiwacs__.
-   The drinks were delicious: blood-red __bardiwac__ as well as light,
    sweet Rhenish.
-   Nigel's face flushed from too much __bardiwac__.
-   Malbec is one of the lesser-known __bardiwac__ grapes.
-   I dined off bread, cheese and this excellent __bardiwac__.

$\Rightarrow$ Bardiwac is a heavy red alcoholic beverage made from
grapes.


## Semantic perspectives: distributional 

Even if we don't know the place of "bardiwac" in a semantic network nor the
meanings of its parts, the *contexts* in which it occurs provide a large amount
of information about it's meaning.

The distributional hypothesis:

-   "You shall know a word by the company it keeps." [^3]

-   "Linguistic items with similar distributions have similar meanings."
    [^4]


## Semantic perspectives: distributional 

An important practical advantage of the distributional approach to meaning is
that it makes it possible to learn the semantics of words automatically from
large but unlabeled text collections, no expert knowledge and annotations are
needed.

The approach is not without limitations, of course:

-   has problems with rare words; and

-   learns the similarities without providing any explanation *why*
    these distributions are similar.


[^1]: [Wikipedia: Principle of
    Compositionality.](https://en.wikipedia.org/wiki/Principle_of_compositionality)

[^2]: The example is from Stefan Evert's [Distributional semantics
    slides](https://esslli2016.unibz.it/wp-content/uploads/2015/10/dsm_tutorial_part1.slides.pdf).

[^3]: J.R. Firth, *Papers in Linguistics 1934--1951 (1957).*

[^4]: [Wikipedia: Distributional
    semantics.](https://en.wikipedia.org/wiki/Distributional_semantics)
