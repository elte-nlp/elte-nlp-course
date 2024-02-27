# Compilation

```
pandoc -t beamer -o output.pdf --citeproc --bibliography=bibliography.bib --slide-level 2 markdown.md
```
## Automatic conversion in Git Repository

A Github Action pipeline is configured to load pandoc conversion of all .md
files in lecture_source. It also automatically uploads the PDFs to Google Drive.
The pipeline is triggered by pushing to the main branch and including `<|CONV|>`
in the commit message.
