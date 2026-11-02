# Behavioral Language Modeling paper

The draft reuses the main source and bibliography from the supplied COLM 2026
archive. The official style, bibliography style, and bundled packages are unchanged.
The original formatting instructions remain available in the root template ZIP.

- `main.tex`: abstract, introduction, related work, background and problem setting, method, experimental setup, results and discussion, and conclusion.
- `main.bib`: verified references.
- `main.pdf`: compiled draft in official submission format.
- `draft_notes.md`: source provenance and questions to resolve before experiments.

Build from the repository root with:

```sh
/Library/TeX/texbin/latexmk -norc -pdf -interaction=nonstopmode -halt-on-error -cd paper/main.tex
```

On other systems, use the installed `latexmk` executable. `-norc` avoids personal
LaTeX settings (including automatic previewers) affecting the build.

This is an initial draft. Submission-mode anonymity, headers, and line numbers
come from the official template; they do not indicate that it has been submitted.
