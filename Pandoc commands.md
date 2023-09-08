To turn Markdowns into Pdf:
```
pandoc -t beamer -o output.pdf --citeproc --bibliography=bibliography.bib --slide-level 2 markdown.md
```