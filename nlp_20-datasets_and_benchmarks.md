---
title: "Natural Language Processing"
subtitle: "Lecture 20: Datasets and Benchmarks"
author: "Dávid Márk Nemeskey"
institute: "Eötvös University, Department of Digital Humanities"
date: 2023
theme: Marburg
colortheme: orchid
fontsize: 12pt
linkcolor: blue
header-includes: |
  \let\emphasized\emph
  \let\strong\textbf
  \DeclareMathOperator*{\argmax}{arg\,max}
  \renewcommand{\textbf}[1]{\textcolor{blue}{\strong{#1}}}
  `\setbeamertemplate{navigation symbols}{}`{=latex}
  `\setbeamertemplate{footline}[page number]`{=latex}
link-citations: true
---
# Introduction

It is common knowledge that LLMs require huge text corpora to (pre-)train.
In actual fact, there are several types of datasets we use to train and/or
evaluate LLMs:

- pretraining corpora,
- fine-tuning datasets,
- instruction fine-tuning datasets,
- benchmarks.

In this lecture, we shall talk about these types in detail and also get
acquainted with the most popular examples for each.

# Pretraining

LLMs (as their name implies) are always trained with some form of language
modeling objective:

- causal (autoregressive) language modeling,
- masked language modeling (MLM),
- etc.

This type of pretraining can only be done on huge text corpora (depending on
model size). LLMs require much more textual data than what a human child / young
adult ever encounters.  On the other hand,

- LLMs don't have multisensory input (some do to some extent);
- we have seen previously that the one-shot labels slow down convergence.

## Pretraining Corpora Sizes

![Optimal token count as a function of parameter size (via FLOPs). A 67B model
needs about 1.5T tokens for pretraining. From @hoffmann2022an.](figures/compute_vs_tokens.png){width=100%}

## Sources

Pretraining corpora usually come from a mixture of sources:

1. Web text
1. Books and entertainment
1. Academic repositories
1. Program code
1. Dialog data
1. Miscellaneous

## Source -- an Example

Composition of **the Pile** [@gao2020pile]: a 800GB English corpus for LLM
pretraining. It was created from 22 datasets with the following composition:

![Treemap of Pile components by effective size.](figures/the_pile.png){width=80%}

## Web Text

Usually the largest component in any pretraining corpus. 

Pros:

- readily accessible, usually in a web crawl format;
- lots of data.

Cons:

- quality varies, in general lower than other sources;
- even good pages contain non-content elements (e.g. ads);
- text duplication; 
- biased, toxic, extremist content;
- AI generated / auto-translated content.

## Web Text Corpora

**[Common Crawl (CC)](https://commoncrawl.org/)**:

\small
\vspace{-1em}

- a free, open repository of web crawl data, made available in WARC format[^1];
- a new crawl roughly each month;
- petabytes of data; the 2023 Sept/Oct crawl is 100TB;
- forms the base of most web text corpora used for pretraining.

\normalsize

```{=latex}
\begin{minipage}{.5\textwidth}
\vspace{-1em}
\begin{center}
```
![](figures/cc_monthly.png){width=75%}
```{=latex}
\end{center}
\end{minipage}%
\begin{minipage}{.5\textwidth}
\vspace{-1em}
\begin{center}
```
![](figures/cc_cumulative.png){width=75%}
```{=latex}
\end{center}
\end{minipage}
```

[^1]: [Web ARChive format](https://en.wikipedia.org/wiki/WARC_(file_format))

## Web Text Corpora (English)

**C4** [@raffel-t5]:

- created from the April 2019 CC dump; 750GB;
- used to pretrain T5;
- filters documents with bad words ($3\times$ decrease);

**WebText** [@radford2019language]:

- GPT-2's pretraining corpus;
- 8 million documents, 40GB;
- created from ``curated'' documents: outbound links from Reddit with at least
  3 karma;
- does not include Wikipedia to avoid test-on-train issues for GPT-2;
- proprietary.

**OpenWebText** [@Gokaslan2019OpenWeb]:

- open-source reimplementation of WebText.

## Web Text Corpora (Multilingual)

**OSCAR** [@abadji-etal-2022-towards]:

- huge multilingual corpus created from a single monthly CC crawl
- [Ungoliant](https://github.com/oscar-project/ungoliant) data pipeline
- token numbers[^2]:
  - English: 377B
  - Hungarian: 4.6B
  - Yoruba: 1k

**ROOTS** [@laurencon2022the]:

- a 1.6TB corpus
- compiled by [BigScience](https://bigscience.huggingface.co/)
    - collaboration of international researchers
    - backed by Hugging Face
- used to pretrain BLOOM.

[^2]: C4 also has a multilingual version (similar `en-hu` ratio).

## Languages in ROOTS

Language distribution of ROOTS. Hungarian is unfortunately missing. Also,
English is highly downsampled compared to other corpora, e.g. OSCAR.

![Overview of ROOTS languages.](figures/roots_languages.png){width=100%}

## How to Preprocess Common Crawl

Creating a web text corpus for a particular language based on Common Crawl seems
trivial, but it is a multi-step process with several pitfalls. Here we
review the steps taken by [cc_corpus](https://github.com/DavidNemeskey/cc_corpus),
the pipeline used to create Webcorpus 2 [@Nemeskey:2020].

**Requirement**: download all (multiple) monthly dumps, because only English has
enough tokens in one.

1. Downloading the index
    - CC index is per-domain, not per-language;
    - as an example, for Hungarian we download 
        - the `.hu` top level domain,
        - other domains that contain many Hungarian pages, based on OSCAR
          statistics;
    - deduplicate the URLs between monthly index dumps.

## How to Preprocess Common Crawl cont.

2. Downloading the data
    - CC should not be DDoS'ed;
    - the WARC files need a _lot_ of space.

3. Boilerplate removal
    - remove the non-content part of web pages (navigation, ads, images,
      tables (!), etc.);
    - we use jusText [@pomikalek2011removing] with custom code to remove
      JS / cookie warnings;
    - need to handle various file types (HTML, RSS, text, etc).

4. Filtering
    - language filtering;
    - quality-based filtering:
        - document length,
        - could also have used e.g. boilerplate ratio, lengths of certain HTML
          tags, etc.

## How to Preprocess Common Crawl cont.

5. Deduplication
    - document-level deduplication to maintain text integrity:
        - MinHash--LSH setup
        - needs a lot of memory or will be very slow;
    - optional: deduplicate frequent paragraphs per domain (content-based
      boilerplate removal).

Hardware setup:

- runs on individual servers; multi-server communication is in the works;
- all map-like steps are highly parallel to fully utilize multicore/CPU servers;
- one server with 768GB memory for deduplication.

## Special Web Text Datasets

**Wikipedia**:

- very good quality, edited resource, mostly truthful
- size is language-dependent, e.g. English is $10\times$ the size of Hungarian
- preprocessing is not trivial because of the markup format:
    - [wikiextractor](https://github.com/attardi/wikiextractor) attempts this
    - [zim_to_corpus](https://github.com/DavidNemeskey/zim_to_corpus) extracts
      the text from already preprocessed .zim archives from
      [Kiwix](https://kiwix.org/)

**Stack Overflow**, **Reddit**:

- curated datasets (points / karma)
- can be used for question answering, programming, etc.

## Edited Texts

Edited texts are an important source of high quality text. Unfortunately, it is
much harder to come by in quantity than web text.

Edited texts usually come in two formats:

1. _Born digital_: texts that have been prepared for digital consumption
                   (from the beginning). Usually usable as-is, but
    - might need boilerplate removal: tables, figures, headers/footers
    - encoding issues do happen, esp. with PDFs
2. Scanned: digitized documents originally on paper. The quality of the
            _layout analysis_ and _optical character recognition (OCR)_
            can range from acceptable to very bad.

## Edited Texts / Prose

Regular prose, such as books, have been part of the LLM training regime since
BERT. The level of the text varies by genre, which results in a diverse
training corpus.

**BookCorpus**

\footnotesize
\vspace{-1em}

- A 985M word corpus created from 7,185 self-published books.
- Used to train GPT and BERT, but it has since been withdrawn and is not
  publicly available[^3].
- BookCorpus2 (the Pile): an extension of BookCorpus, around 17k books.

\normalsize
\vspace{-1em}

**Published book corpora**:

\footnotesize
\vspace{-1em}

- Books1-2 (GPT-3): 67B tokens
- Books3, Project Gutenberg (the Pile): approx 187k and 27k books, resp.
- Hungarian Elektronic Library (MEK): 32,830 books, 800M tokens.

\normalsize

[^3]: Although https://huggingface.co/datasets/bookcorpus.

## Edited Texts / Prose

**OpenSubtitles**:

- @lison-tiedemann-2016-opensubtitles2016 created 1689 bitexts from movie and
  TV subtitles;
- it is possible to extract an approx. 300M words corpus from it;
- the corpus consists mostly of dialogs.

Books, etc. are a large and very useful part of pretraining corpora. However,
they are not fully safe:

- there might be problematic content (pornography, toxicity, etc);
- using them in models might result in **copyright violation**.

## Edited Text / Professional

Very high quality, usually professional texts with their own terminology.

1. **Academic repositories**:
    - very high level text;
    - many tables, figures, etc., which break the text flow;
    - usually university access (and a crawler) is required to download the papers.
2. **Parliamentary proceedings**:
    - national / EU / etc.
    - some might provide REST APIs, some have to be crawled.
3. Laws, rulings, regulations, etc.

## Edited Text / Professional cont.

4. **News**:
    - large and important source, but can also be biased and toxic;
    - extreme duplication;
    - often behind a paywall.
5. **Private data**:
    - company rules and intra-company communications;
    - know-how, etc.

## Miscellaneous Data

**Dialogs**

- very important for chat bots;
- for generic conversation: movies, books, etc.
- most important sources: internet forums, actual customer service interactions.

**Programming**

- open source projects from CVS services (GitHub, SourceForge, etc.);
- **copyright** and **licence violation** are a problem with possible
  [legal consequences](https://www.theverge.com/2022/11/8/23446821/microsoft-openai-github-copilot-class-action-lawsuit-ai-copyright-violation-training-data).


# Instructions

## Instruction Acquisition

We discussed in a previous lecture how instruction fine-tuning datasets are
compiled:

1. Manual / crowdsourcing effort;
1. Data collection from users;
1. Conversion of NLP tasks into instructions;
1. Self-instruct.

We have seen how FLAN [@wei2022finetuned] converts NLP tasks into instructions,
but we skipped over the first category.

## Manual Instructions

Creating an instruction dataset manually requires crowdsourcing. Two examples:

1. Databricks' **[Dolly](https://github.com/databrickslabs/dolly)**
   [@DatabricksBlog2023DollyV2]:
    - contains 15,000 prompt/response pairs,
    - created by 5,000+ Databricks employees.
1. LAION's **[Open-Assistant](https://open-assistant.io/)**
    - Compiled by volunteers
    - English and Spanish are well represented, but the rest of the languages 
      are not:

![](figures/open_assistant.png){width=100%}


# Fine-tuning

## Fine-tuning Datasets

It has been shown that LLMs with a classifier head can be fine-tuned on NLP
datasets to achieve state-of-the-art results. This includes

- traditional NLP tasks (NP chunking, NER, dependency parsing, etc.)
- NLU tasks (question answering, natural language inference, etc.)
- various classification datasets (sentiment analysis, topic classification, etc.)
which are usually trained
on treebanks 

Fine-tuning datasets have train-devel-test splits, so they function as benchmark
datasets as well.

## Traditional Datasets

\scriptsize
+-----------------------------------+--------------------------------------------------------------------------+---------------------------------------------------------------------------------+
| Task                              | English                                                                  | Hungarian                                                                       |
+:==================================+:=========================================================================+:================================================================================+
| NER                               | [CONLL 2003](https://huggingface.co/datasets/conll2003)                  | [NYTK NerKor](https://github.com/nytud/NYTK-NerKor)                             |
+                                   +--------------------------------------------------------------------------+---------------------------------------------------------------------------------+
|                                   | [other datasets](https://github.com/juand-r/entity-recognition-datasets) |                                                                                 |
+-----------------------------------+--------------------------------------------------------------------------+---------------------------------------------------------------------------------+
| NP Chunking                       | [CONLL 2003](https://huggingface.co/datasets/conll2003)                  | [Szeged TreeBank](https://rgai.inf.u-szeged.hu/node/113)                        |
+-----------------------------------+--------------------------------------------------------------------------+---------------------------------------------------------------------------------+
| Dependency                        | [Universal Dependencies](https://universaldependencies.org/)                                                                                               |
+-----------------------------------+--------------------------------------------------------------------------+---------------------------------------------------------------------------------+
| Parsing                           | [Penn TreeBank](https://catalog.ldc.upenn.edu/LDC99T42)                  | [Szeged Dependecy TreeBank](https://rgai.inf.u-szeged.hu/node/158)              |
+-----------------------------------+--------------------------------------------------------------------------+---------------------------------------------------------------------------------+
\normalsize

**Other resources**

- Progress on NLP tasks can tracked on the aptly named
[NLP-progress page](http://nlpprogress.com/);
- There are various lists for NLP datasets:
    - [Awesome NLP / Datasets](https://github.com/niderhoff/nlp-datasets)
    - [nlp-datasets](https://github.com/niderhoff/nlp-datasets)
    - [Awesome Hungarian NLP / Datasets](https://github.com/oroszgy/awesome-hungarian-nlp#datasets)

## NLU Datasets

These include tasks that traditional NLP could (and cannot) solve, but LLMs can.
Consequently, these datasets serve as handy benchmarks for LLMs.

\footnotesize
1. **[GLUE](https://gluebenchmark.com/)** [@wang-etal-2018-glue]:
    - a NLU benchmark with 9 tasks (sentence similarity, paraphrasing, QA, etc.)
    - test set is not shared; online leaderboard.
2. **[SuperGLUE](https://super.gluebenchmark.com/)** [@wang-etal-2019-superglue]:
    - 8 curated tasks (open, hard, permissive licence, etc.)
3. **[SQuAD2.0](https://rajpurkar.github.io/SQuAD-explorer/)** [@rajpurkar-etal-2018-know]
    - Compiled by crowdworkers;
    - 100k questions plus 50k adversarial, unanswerable questions.
4. **[MMLU](https://github.com/hendrycks/test)** [@hendrycks2021measuring]
    - A test-only benchmark with 15,687 multiple choice questions in 57 topics

\normalsize

## Adversarial Benchmarking

**Problem**: benchmarks are ``cleared'' too quickly by ever better models. Can
we create really hard benchmarks that last longer?

**Model brittleness**: Evidence [@gururangan-etal-2018-annotation,
@poliak-etal-2018-hypothesis, ...] shows that 

- Natural Language Inference (NLI) datasets exhibit spurious statistical
  patterns due to (annotator) bias;
- models actually learn these patterns, _not reasoning_;
- as such, they are brittle and can be broken by non-expert annotators.

**Idea**: Human-And-Model-in-the-Loop Enabled Training (HAMLET).

## Adversarial NLI

**Adversarial NLI (ANLI)** [@nie-etal-2020-adversarial] was compiled by
introducing an ``arms race'' between annotators and models.

![](figures/hamlet.png){width=100%}

This results in

- a good training set that transfers well to other NLI benchmarks;
- a very hard training set.

## Test Harnesses

LLM testing is increasingly becoming automatized via **test harnesses**:

- Google's [BIG-bench](https://github.com/google/BIG-bench)
- EleutherAI's [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)

Both contain 200+ tasks and offer

- easy integration for new tasks;
- evaluating a model with all of them.

Reproducible testing enables competition such as the
**[Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)**.

## XXX

$$\mathnormal{\hat{En} = \argmax_{En} P({Ru}|{En})}$$

\begin{align*}
  \mathnormal{\hat{En}} &= \mathnormal{\argmax_{En} P({Ru}|{En})} \\
  \mathnormal{\hat{En}} &= \mathnormal{\argmax_{En} \frac{P({En}|{Ru})P({En})}{P({Ru})}} \\
  \mathnormal{\hat{En}} &= \mathnormal{\argmax_{En} P({En}|{Ru})P({En})}
\end{align*}

# References

## References {.allowframebreaks} 
\footnotesize
