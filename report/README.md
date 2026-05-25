# FP Report

Compile the preliminary/final PDF:

```bash
cd report
pdflatex report.tex
pdflatex report.tex   # second pass for references
```

Optional figures (add before final submission):

- `figure/kaggle_snapshot.png` — Kaggle leaderboard screenshot
- `figure/training_curves.png` — train/val RMSE vs epoch from `log/*.csv`

Course-required sections (see `(114-2)FinalProject.pdf`):

1. Introduction
2. Related works
3. Method / Approach
4. Experimental Results (+ Kaggle rank screenshot)
5. Conclusion
6. Reference
7. Team contribution table
