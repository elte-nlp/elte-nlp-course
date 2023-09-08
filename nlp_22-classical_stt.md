---
title: "Natural Language Processing"
subtitle: "Lecture 22: Classical Speech-to-Text"
author: András Simonyi
institute: "Eötvös University, Department of Artificial Intelligence"
date: 2023
theme: Frankfurt
colortheme: orchid
fontsize: 12pt
linkcolor: blue
header-includes:
  - \let\emphasized\emph
  - \let\strong\textbf
  - \renewcommand{\textbf}[1]{\textcolor{blue}{\strong{#1}}}
link-citations: true
---

# The speech-to-text task

The **speech-to-text** (STT) task, also called **automatic speech recognition** (ASR):

* The input is acoustic signal containing speech (e.g., a `wav` file).
* Output is a written transcript of the spoken content.
* The produced transcripts are not necessarily segmented into sentences or
  contain proper punctuation and capitalization.
* The task is _supervised_: models are trained on transcribed speech corpora.
   
# Challenges

Challenges stem from the differences between speech and writing, and context
dependence:

* **Segmentation**: word boundaries in writing are frequently not indicated by
  the acoustic segmentation of speech by silences, and, vice versa, speech
  silences are not necessarily indicative of word boundaries.
* **Ambiguity**: differently written texts can be pronounced the same way, e.g.,
  in English *bare* and *bear* has the same pronunciation.
* The phenomenon of **coarticulation**: speech sounds following each other can
  interact and influence each other's pronunciation, e.g., the *v* in *I have
  to* is pronounced as *f* (in fast speech) because of the following voiceless
  *t*.

# Challenges cont. 

* The so-called **Lombard effect**: One cannot augment data sets simply by adding
  noise because people change the way they speak in noisy environments (and it
  is not just speaking louder...).
* Speech, in contrast to typical written language, can contain agrammatical
  constructs, incomplete sentences or words, corrections, word/syllable
  repetitions and interruptions.
* **Speaker adaptation**: there are huge differences between how people of
  different gender, age, cultural background etc. pronounce words.

# References {.allowframebreaks}
