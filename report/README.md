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

Course-required sections — see `docs/COURSE_REQUIREMENTS.md` (from `(114-2)FinalProject.pdf` + `Tips_FinalProj_Presentation.pdf`):

1. Introduction (problem, importance, difficulty)
2. Related works (grouped; pros/cons)
3. Method / Approach (overview figure + details)
4. Experimental Results (+ **Kaggle rank screenshot**, SOTA comparison, ablations)
5. Conclusion (summary + lessons learned)
6. Reference
7. **Code link** (GitHub/GitLab)
8. **Team contribution table** (five tasks, % per member — also on slides)

**Deadline:** report + code to E3 by **2026/05/31 23:59**; presentation **2026/06/02**.
