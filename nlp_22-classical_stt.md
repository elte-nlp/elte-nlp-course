---
title: "Natural Language Processing"
subtitle: "Lecture 22: Classical Speech-to-Text"
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
---
# The speech-to-text task
## The speech-to-text task

The **speech-to-text** (STT) task, also called **automatic speech recognition** (ASR):

* The input is acoustic signal containing speech (e.g., a `wav` file).
* Output is a written transcript of the spoken content, frequently without
  punctuation and capitalization.
* The produced transcripts are not necessarily segmented into sentences or
  contain proper punctuation and capitalization.
* The task is _supervised_: models are trained on transcribed speech corpora.
   
## Challenges

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

## Challenges cont. 

* The so-called **Lombard effect**: One cannot augment data sets simply by adding
  noise because people change the way they speak in noisy environments (and it
  is not just speaking louder...).
* Speech, in contrast to typical written language, can contain agrammatical
  constructs, incomplete sentences or words, corrections, word/syllable
  repetitions and interruptions.
* **Speaker adaptation**: there are huge differences between how people of
  different gender, age, cultural background etc. pronounce words.
* Human speech understanding heavily relies on contextual background information on
  admissible interpretations -- we actively "perceive"/"hear" speech using
  contextual clues. A dramatic example from @hiphi2019 can be heard
  [here.](http://drive.google.com/uc?export=view&id=1ICNa4Hj-lU_4POjdSCk_-Zyly93SUTNK)

## Task variants

* __Continuous__ vs __isolated__ recognition: 
  * In the __isolated__ case the input either consists of or can easily be
    segmented into single words (because there are separating bits of silence).
  * In the __continuous__ case words can follow each other without any silence
    between them, as in normal speech. Continuous speech recognition is
    significantly harder.
* __Joint__ recognition (possibly with __diarization__): Basic speech
  recognition is for one speaker: a more complex variant is where there are more
  speakers (e.g., in a dialogue), and, optionally, the transcript has to
  indicate who says what (diarization). Overlapping speech can be an especially
  difficult problem in this setting.
  
## Evaluation

The most common metric is __word error rate__ (WER), which is based on the
word-level edit distance compared to the correct transcript. If $\hat{W}$ is the
output and $W$ is the correct transcript then the WER is simply defined as

$$
\frac{\mathrm{Levenshtein}(\hat W, W)}{\mathrm{length}(W)},
$$

i.e., the average number of word-level editing operations per word necessary to
get the output from the correct transcript.

# Training data
## Training data

In general, training data consists of __recorded speech audio__ with
__time-aligned written transcipts__.

In the past, transcripts were __phonetic__, and aligned at the phone level, so
annotators had to determine phone boundaries by listening and looking at
spectrograms:

![x](figures/speech_transcript.jpg)\

## Training data cont.

Improvements in training methods made phone level alignmemt obsolete: modern ASR
data sets contain __normally written__ transcripts which have to be time aligned
only at a __sentence level__.

Despite these improvements, it is still a huge amount of work to create good ASR
data sets, since usable corpus size starts at 20 hours of speech from several
speakers, both male and female. Because of the associated costs, the number of
freely available corpora is low even for the most widely spoken languages, and
for many languages no free data set exists at all.

## LCD data sets

For English, until recently, most public data sets were published by the LDC,
the Linguistic Data Consortium. These include the

* Wall Street Journal audio corpus (read newspaper articles, 80h, 1993)
* Fisher corpus (telephone speech, 1600h, 2004/2005)
* Switchboard corpus (telephone speech. 300h, 1993/1997/2000)
* TIMIT corpus (read example sentences, limited grammatical/vocab. variability, 1986) 
  
More recently, data sets in other languages got added to the LDC catalog, now it
contains, among others, Spanish, Mandarin and Arabic.

## Open initiatives

Unfortunately, LCD data sets are typically not free, either LCD membership or
payment is required for accessing most of them.

Recent initiatives to create and curate freely available data sets:

* The [Open Speech and Language Resources
  page](https://www.openslr.org/resources.php) lists several free data sets for
  various languages, among them the important __LibriSpeech__ corpus, which
  contains ~1000h speech from audio books.

*  __Common voice__: A Mozilla project to collect ASR data sets for as many
   languages as possible. Already collected and validated 2484hs of transcribed
   English speech and other languages are progressing as well, German is at
   1290hs, French at 958hs at the time of writing.

# Speech signal processing

## Continuous speech signal in time

When speech  is recorded the air pressure changes move the
microphone's diaphragm, and these movements are converted to changes in
electronic current -- as a consequence, the speech gets represented as a
continuous signal:

![From @santos2014.](figures/time_speech_cont.png){width=60%}

## Discrete speech signal in time

This is a continuous, analog signal, which can be digitalized by sampling with a
certainly rate:

![From @santos2014.](figures/time_speech_disc.png){width=60%}

## Transforming to the frequency domain

# References
## References 

\small

