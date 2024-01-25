---
title: "Natural Language Processing"
subtitle: "Lecture 3: Tokenization"
author: "András Simonyi"
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
aspectratio: 1611
output: 
  beamer_presentation:
    slide_level: 2
bibliography:
---

## Baseline: splitting on whitespace

For many writing systems, splitting
the text at *white spaces* is a useful baseline:

'This isn't an easy sentence to tokenize!' $\Rightarrow$\
\['This', \"isn't\", 'an', 'easy', 'sentence', 'to', 'tokenize!'\]

Problems:

-   we typically want to treat punctuation marks as separate tokens (but
    only if they are really punctuation, think of 'U.K.' or
    '10,000.00\$');
-   this solution cannot separate token pairs without white space
    between them, e.g., expressions with clitics like \"isn't\".


# Regular expressions and languages
## Regular expressions I

We need to introduce more sophisticated patterns to
describe token boundaries in context-dependent way. A popular solution
is using __regular expressions__ (regexes for short).

Given a finite $\Sigma$ alphabet of symbols, regexes over $\Sigma$ and
their matches in $\Sigma^*$ are defined by simultaneous recursion as
follows:

1.  The empty string and any single symbol in $\Sigma$ is a regex over
    $\Sigma$ and matches itself.


## Regular expressions II

2.  If $r_1$ and $r_2$ are regexes over $\Sigma$ then

    1.  their __concatenation__, $r_1 r_2$ is also a regex over $\Sigma$
        and matches exactly those strings that are the concatenation of
        a string matching $r_1$ and a string matching $r_2$, and

    2.  their __alternation__, $r_1 \vert r_2$ is also a regex over
        $\Sigma$ and matches exactly those strings that match either
        $r_1$ or $r_2$.

3.  If $r$ is a regex over $\Sigma$ then applying the __Kleene star__
    operator to $r$ we can form a new regex $r^*$ which matches exactly
    those strings that are the concatenation of 0 or more strings each
    matching $r$.


## Formal languages I

Given a finite alphabet $\Sigma$, a __formal language__
$\mathcal L$ is an arbitrary set of strings over $\Sigma$. These strings
are often defined via a __grammar__.

Based on the complexity of the grammar, languages can be classified into
different __types__. The most well-known is the Chomsky hierarchy:

![Figure from  [Wikipedia](https://en.wikipedia.org/wiki/Chomsky_hierarchy).](figures/chomsky_hierarchy.eps){width="40%"}

## Formal languages II

- Regular languages: can describe linear structures

    -   Programming: state machines
    -   Languages: words, noun phrases

-   Context-free languages: can describe tree structures

    -   Programming: XML DOM, parse trees
    -   Languages: phrase structure grammars (of sentences)

-   Context sensitive languages: most human languages are mildly context
    sensitive

-   Recursively enumerable: all problems that can be solved by an
    algorithm


## Back to regular expressions\...

A formal language $\mathcal L$ is
__regular__ iff there exists a regular expression that matches exactly
$\mathcal L$'s elements.

There are simple formal languages that are not regular, e.g., the "twin
language" $\{ww ~\vert~ w \in \{a, b\}^* \}$.

Nonetheless, regular expressions are flexible enough for a lot of
practical tasks, and there are higly efficient algorithms for deciding
whether an $s$ string matches a regex (with
$\mathcal O(\mathrm{length}(s))$ time complexity).


## Regular languages and  FS acceptors I

Finite state acceptors are finite state machines that consume an input sequence
of characters and can \"accept\" or \"reject\" the input. They differ from the
simplest possible FSA-s by having

-   an explicit __start state__,

-   a set of designated __accepting states__, and

-   their transitions labeled with symbols from a finite alphabet, or
    with the empty string.

An FS acceptor __accepts__ an input iff it has a sequence of transitions
which starts from the start state, ends in an accepting state, and the
concatenation of the transition labels is the input in question.


## Regular languages and FS acceptors II

An acceptor for the words "car", "cars", "cat" and "cats":
```{=latex}
\begin{center}
```
![image](figures/fsa1.eps){width="50%"}\
```{=latex}
\end{center}
```
Can be simplified to (figures from @buutbogel2009fsmorph):

```{=latex}
\begin{center}
```
![image](figures/fsa2.eps){width="50%"}\
```{=latex}
\end{center}
```

## Regular languages and FS acceptors III

The connection between FS acceptors and regular languages/expressions is
established by Kleene's __Equivalence Theorem__:

-   A language is regular iff there is an FSA acceptor that accepts
    exactly its elements.[^1]

This equivalence is important both theoretically and practically: there
are very efficient algorithms to simplify/minimize FS acceptors and also
to decide whether they accept a string or not.


## Regular expressions: extensions 

__Convenience extensions__ not increasing
expressive power just adding useful shortcuts, e.g.;

-   character classes matching any single symbol in a set, e.g. \[0-9\];
-   complemented character classes matching any character *not* in the
    complemented set, e.g. \[\^ab\];
-   operators for specifying the required number of pattern repetitions,
    e.g., $r\{m,n\}$ matches $s$ if $s$ repeats the $r$ pattern $k$
    times, with $m\leq k \leq n$.
-   optional matching: $r? = r\{0,1\}$
-   Kleene plus: $r+ \approx r\{1,\infty\}$

## Regular expressions: back-references

So-called __back-reference__ constructions, in contrast, that allow naming and
referencing match(es) corresponding to earlier parts of the regex *increase* the
expressive power.

For example, most current regex libraries allow a regex similar to

```{=latex}
\begin{center}
```
(?P$<$a$>$\[ab\]\*)(?P=a)
```{=latex}
\end{center}
```
which uses back-reference to define exactly the aforementioned
non-regular "twin language".

## Regex-based find and replace

In addition to matching whole strings
against a regex, two regex-based tasks are very common:

-   finding substrings of a string that match a regex,

-   regex-based find-and-replace: in its simplest form this is replacing
    matching substrings with a given string, but two notable extras are
    provided by modern regex libraries:
    -   the regex can have look-ahead and look-back parts, that are used
        when finding a match but do not count in the part which is
        replaced;
    -   replacements do not have to be fix -- they can contain
        back-references to parts of the match.

# Rule-based tokenization
## Regex cascade-based tokenization I

Core idea: perform regex-based substitutions on the input so that in the end
it's enough to split on white spaces.

The [tokenizer sed
script](ftp://ftp.cis.upenn.edu/pub/treebank/public_html/tokenizer.sed)
accompanying the Penn Tree Bank is a good example. A few representative
rules (\\& refers back to the full match, \\$n$ to the $n$-th group):

-   '\...' $\Rightarrow$ ' \... ' (separate ellipsis)
-   '\[,;:#\$%&\]' $\Rightarrow$ ' \\& ' (separate various signs)
-  (\[\^.\])(\[.\])(\[\])}\"'\]\*)\[\]\*\$ $\Rightarrow$ '\\1 \\2\\3'
   (assume sentence input and split FINAL periods only)
-  \"'ll\" $\Rightarrow$ \" 'll\" (separate clitic 'll)


## Regex cascade-based tokenization II

The main problem for the approach is the proper handling of exceptions: e.g.,
word ending periods should be split, *except for abbreviations*.

The standard solution is to replace the problematic expressions with
unproblematic placeholders before executing the substitutions in
question, e.g.
```{=latex}
\begin{center}
```
(etc\\.$\vert$i\\.e\\.$\vert$e\\.g\\.) $\Rightarrow$ $<$abbrev$>$
```{=latex}
\end{center}
```
This solution requires keeping track of the placeholder substitutions
and restoring the originals after executing the problematic rules.

## Lexer-based solutions I

They use off-the shelf "lexers" (lexical
analyzers), originally developed for the tokenization/lexical analysis
of computer programs.

A typical lexer takes a character stream as input and produces a stream
of classified tokens from it:

```{=latex}
\begin{center}
```
![x](figures/lexer.eps){width="60%"}\
```{=latex}
\end{center}
```

## Lexer-based solutions II

Most lexers are actually lexical analyser
generators. Their input is a list of token classes (types), regular
expression patterns and

```{=latex}
\begin{center}
```
\[[RegexPattern]{.smallcaps}\] $\Rightarrow$ \[[Action]{.smallcaps}\]
```{=latex}
\end{center}
```

rules (where the most important action is classifying the actual match
as a token of a given type), and they generate a concrete, optimized
lexical analyzer implementing the given rules, e.g., by generating the C
source code of the analyzer.


## SpaCy's rule-based tokenizer I

1.  The input text is split on white space.[^2]
2.  Then, the tokenizer processes the text from left to right. On each
    substring, it performs two checks:

    1.  Does the substring match a tokenizer exception rule? For
        example, \"don't\" does not contain whitespace, but should be
        split.
    2.  Can a prefix, suffix or infix be split off? For example
        punctuation like commas, periods.

    If there's a match, the rule is applied and the tokenizer continues
    its loop, starting with the newly split substrings.


## SpaCy's rule-based tokenizer II

A simple example: tokenizing *"Let's go to N.Y.!"*

![from the [spaCy
documentation](https://spacy.io/usage/linguistic-features#tokenization).](figures/spacy_tokenizer.eps){width="90%"}

# Edit distance
## Edit distance I

In addition to segmenting the input into units,
tokenization also involves classifying tokens into types, deciding e.g.,
which type(s)

```{=latex}
\begin{center}
```
'Apple', 'apple', 'appple'
```{=latex}
\end{center}
```

belong to. In many cases, these decisions require a *similarity metric*
between strings.

One of the most important metric families in this domain is the
so-called __edit distance__ family, which measures the distance between
two strings by the minimal number of edit operations required to
transform them into each other.


## Edit distance II

Given

-   a set of __editing operations__ (e.g., removing or inserting a
    character from/into the string) and

-   a __weight function__ that assigns a weight to each operation,

the __edit distance__ between two strings, a source and a target, is the
minimum total weight that is needed for transforming the source into the
target.


## Levenshtein distance

One of the most important variants is the so-called
Levenshtein distance, where the operations are the

-   deletion,
-   insertion, and
-   substitution of a character,

and the weight of all operations is 1.0.

![image from [Devopedia](https://devopedia.org/levenshtein-distance).](figures/levenshtein.eps){width="40%"}



# Subword tokenization
## Tokenization vs subword tokenization I

Classical tokenization aims at
segmenting the input character stream precisely into linguistically
motivated units: words and punctuation.

This type of tokenization is useful for human linguistic analysis, but
not for building large language models: it

-   is rather challenging, because there is a wide variety of writing
    systems and languages that all require (sometimes radically)
    different rules/models;

-   generates huge vocabularies on larger corpora, and still leads to
    problems with out of vocabulary words.


## Tokenization vs subword tokenization II

A recently developed alternative is __subword tokenization__. A large number of
modern deep, end-to-end NLP architectures use subword tokenization instead of
classical tokenization for segmenting the input. The main advantages are

-   requires no or only very minimal pretokenization;
-   statistical and data-driven: learns the segmentation model from a
    corpus (no manual rule writing);
-   vocabulary size can be freely chosen; it will contain the most
    frequent word and subword units;
-   writing and language-agnostic.

## Tokenization vs subword tokenization III

What this means for the text:

-   the input text will be segmented into the most frequent words and
    subwords
-   quality of the segmentation depends on the vocabulary size,
    -   for a single language, 30,000 types is enough
    -   a regular vocabulary is in the hundred thousands or even
        millions
-   *no out-of-vocabulary words* (with the right settings!),
-   the subword segments are in usually informative, boundaries are
    frequently close to morphological ones.

## Byte Pair Encoding (BPE) I

BPE was originally a simple compression technique for byte sequences, but can be
generalized to any sequence consisting of symbols from a finite alphabet. To
generate an encoded / compressed version of a sequence over an alphabet,

1.  initialize the symbol list with the symbols in the alphabet, and

2.  repeatedly count all symbol pairs and replace each occurrence of the
    most frequent pair ('A', 'B') with a new 'AB' element, and add 'AB'
    to the list of symbols.

## Byte Pair Encoding (BPE) II

How can this technique be used for subword tokenization of texts? The trick is
to learn the vocabulary used for segmentation from a training corpus by applying
BPE to the text. Modifications:

-   start with a rough pretokenization into words (frequently highly
    simple, e.g. split on white space),
-   do not allow BPE merges to cross word boundaries.

In practice these modifications are frequently implemented by adding a
new '\_' word-start (or word-end) symbol to the alphabet, and
stipulating that '\_' can only end (or start) merged items.

## Byte Pair Encoding (BPE) III

A simple example: BPE encoded versions of a sentence after different numbers of
merge operations.

![Table from @heinzerling2017bpemb.](figures/bpe.eps){width="1.\\textwidth"} 

As the number of merges increases, more and more symbols are full words.

## Greedy BPE subword tokenization 

Processing a corpus with BPE results in a 'vocabulary' of all characters
together with the results of all merges. New pretokenized input is then subword
tokenized by greedily matching the words against this vocabulary/dictionary from
left to right:

![From @jurafsky2019speech.](figures/max_match.eps){width="1.\\textwidth"}

## WordPiece 

WordPiece is another subword tokenization method that is only slightly different
from BPE. The differences are:

-   WordPiece works with word-start symbols instead of word-end symbols
    which is traditional for BPE;
-   merges are performed depending on which resulting merged symbol
    could be used for a statistical language model with the lowest
    perplexity (maximal likelihood) on the training corpus. (These
    concepts will be explained in detail in a later lecture.)



## Subword sampling

The default subword tokenization strategy of BPE and
WordPiece deterministically produces the greedily matching decomposition
of words, even if there are informative alternative segmentations:

```{=latex}
\begin{center}
```
*unrelated* = *unrelate* + *d*\
  
*unrelated* = *un* + *related*

```{=latex}
\end{center}
```
To solve this problem, solutions were developed to *probabilistically
sample* from the possible alternative decompositions: *Subword
regularization* for WordPiece [@kudo2018subword] and *BPE dropout* [@provilkov2019bpe].

## SentencePiece 

In their original form, BPE and WordPiece require (crude) pretokenization as a
preprocessing step. The [SentencePiece](https://github.com/google/sentencepiece)
tokenizer, in contrast, treats every character, even spaces in the same way, and
applies BPE or WordPiece on raw sentences or even paragraphs eliminating the
need for pretokenization.

As a result, SentencePiece is one of the most popular solutions for
generating the input for deep end-to-end models from raw text.

# References
## References

\footnotesize

[^1]: See, e.g., <https://bit.ly/2ZHlKWG> for a proof.

[^2]: The algorithm description is from the [spaCy
    documentation.](https://spacy.io/usage/linguistic-features#tokenization)

[^3]: See [Kudo: Subword
    Regularization](https://arxiv.org/pdf/1804.10959.pdf) and [Provilkov
    et al.: BPE-Dropout](https://arxiv.org/pdf/1910.13267.pdf).
