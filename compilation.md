# Compilation

## Markdown (.md) to PDF

```
pandoc -t beamer -o output.pdf --citeproc --bibliography=bibliography.bib --slide-level 2 markdown.md
```

## LaTeX (.tex) to PDF

The .tex slides use the `powerdot` document class, which causes some complications.
The first of these is that `pdflatex` doesn't work. One solution is to use
`latexmk`:

```
latexmk -pdfps nlp_3-tokenization.tex
```

The other complication is that it includes some esoteric style files. On Ubuntu,
the following packages need to be installed to make it work:

```
sudo apt install latexmk
sudo apt install texlive-pstricks texlive-science texlive-humanities
```

## Automatic conversion in Git Repository

A Github Action pipeline is configured to load pandoc conversion of all .md files in lecture_source. It also automatically uploads the PDFs to Google Drive. The pipeline is triggered by pushing to the main branch and including `<|CONV|>` in the commit message.
