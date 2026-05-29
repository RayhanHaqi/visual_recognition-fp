# Course requirements checklist (FP)

Mapped from `(114-2)FinalProject.pdf` and `Tips_FinalProj_Presentation.pdf`. Use with `docs/EXPERIMENT_LOG.md` (trials + decisions) and `docs/REPORT_OUTLINE.md` (report/slides content).

**Our topic:** #3 — [NOAA Fisheries Steller Sea Lion Population Count](https://www.kaggle.com/competitions/noaa-fisheries-steller-sea-lion-population-count) (Detection, **RMSE**, ~103 GB).

**Our slot:** Presentation **2026/06/02** (Topic 3 block) → report + code deadline **2026/05/31 23:59** (late **−20 pts/day**). Slides may be revised after; **report cannot**.

---

## Grading (what to optimize)

| Component | Weight | Requirement |
|-----------|--------|-------------|
| **Model performance** | 50% | See NOAA bands below |
| **Presentation** | 30% | Completeness, innovation, organization (Tips PDF) |
| **Report & code** | 10% | Sections + code link + quality bar (Tips PDF) |
| **Within-group peer review** | 10% | Teammates grade contribution table |

### NOAA — model performance bands (RMSE, lower is better)

| Band | Points | Criterion (course sheet) |
|------|--------|---------------------------|
| Golden medal | 25 | Historical golden range **15.88009 – 10.85644** |
| Top-3 (class) | 40 | Strong baseline / top-3 on **course** leaderboard |
| Near best | 50 | Within **3%** of best score (detection rule) |

**Course hint for top-3 band:** private RMSE **&lt; ~13.05** (team target; confirm on course Kaggle when screenshot taken).

**Near-best math (detection, 3%):** if class best RMSE = \(B\), aim for RMSE \(\leq B \times 1.03\) (RMSE is “lower is better”).

**Reproduce SOTA:** course encourages reproducing/combining public solutions (outrunner, Lopuhin, Asanakoy) — tracked in `EXPERIMENT_LOG.md` §1 and §6.

---

## Deliverables and deadlines

| Item | Due | Owner | Status |
|------|-----|-------|--------|
| Report PDF + code zip/link on **E3** | **2026/05/31 23:59** | Team leader | ☐ |
| Slides on **Google Drive** (see naming below) | Before 06/02 (update allowed) | Team | ☐ |
| Live presentation **12 min** (10 + 2 Q&A) | **2026/06/02** | Team | ☐ |
| Peer review of teammates | Per course schedule | Each member | ☐ |

### Slide upload filename (required)

```text
Group[GROUP_ID]_[TOPIC]_[LEADER_STUDENT_ID].pptx
```

Example: `Group99_NOAA Fisheries Steller Sea Lion Population Count_112233445.pptx`  
(`GROUP_ID` = two digits, zero-padded.)

---

## Mandatory content (report **and** presentation)

From Final Project PDF + Tips — **all required**:

| # | Item | Report (`report/report.tex`) | Slides | Notes |
|---|------|------------------------------|--------|-------|
| 1 | **Introduction** | §1 | Yes | Problem, importance, motivation/difficulty (Tips) |
| 2 | **Related work** | §2 | Yes | **Groups** + pros/cons per group (Tips) |
| 3 | **Method / proposed approach** | §3 | Yes | Overview figure + details; technically sound (Tips) |
| 4 | **Experimental results** | §4 | Yes | Dataset, metric, **SOTA comparison**, **ablations** (Tips) |
| 5 | **Conclusion** | §5 | Yes | Summary + **what you learned** (Tips) |
| 6 | **References** | §6 | Optional on slides | Bib/links |
| 7 | **Code link** | Yes | Yes | GitHub/GitLab URL |
| 8 | **Kaggle rank screenshot** | `figure/kaggle_snapshot.png` | Yes | Highlight best submit |
| 9 | **Team contribution table** | Yes | Yes | Five tasks, % per member (see below) |

### Team contribution table (five tasks)

Fill with student IDs and percentages (each row sums to 100% across members):

| Task | Member A % | Member B % | Member C % |
|------|------------|------------|------------|
| Literature survey | | | |
| Approach design | | | |
| Approach implementation (experiment) | | | |
| Report writing | | | |
| Slide making and oral presentation | | | |

Example format is on Final Project PDF p.17.

---

## Tips PDF — quality bar (beyond “checkbox” sections)

Meeting all sections ≈ **80%** of report/code portion; full credit needs:

| Section | Grader expectation |
|---------|-------------------|
| Introduction | How you **advance** this topic |
| Related work | **Advantages** of your method vs existing |
| Method | Design matches claims; **technically sound** |
| Results | **SOTA comparison**; ablations support claims |
| Conclusion | **New / insightful** findings |

**Feed from experiments:** `EXPERIMENT_LOG.md` §2–3 → ablation table and decision narrative in report §4–5.

---

## Presentation structure (10 min talk)

Suggested flow (align with Tips):

1. **Introduction** (~1.5 min) — problem, data scale, why hard  
2. **Related work** (~1.5 min) — tile CNN / UNet / ensembles; grouped  
3. **Method** (~3 min) — pipeline diagram: dots → train tiles → infer → pup scale  
4. **Results** (~3 min) — Kaggle screenshot, phase table, best RMSE **14.58**, ablations  
5. **Conclusion** (~1 min) — findings + ongoing v9/v10 (if still running, say “in progress” in report only if deadline passed)  

Reserve **2 min Q&A**.

---

## Repo ↔ course mapping

| Course ask | Where in repo |
|------------|----------------|
| Code link | `README.md`, report, slides → `https://github.com/RayhanHaqi/visual_recognition-fp` (verify) |
| Experiments documented | `docs/EXPERIMENT_LOG.md` |
| Report draft | `report/report.tex` → `report/report.pdf` |
| Kaggle screenshot | `report/figure/kaggle_snapshot.png` (add before E3) |
| Reproduce public solutions | `scripts/import_dot_coords.py`, Phase 6/7 scripts, refs in report §2 |
| Lab repro | `AGENTS.md`, `docs/LAB_SETUP.md` |

---

## Pre-submission checklist (05/31)

- [ ] `EXPERIMENT_LOG.md` updated with final Kaggle scores (v8, v9, v10, sweeps)  
- [ ] `report/report.pdf` built; **no** post-deadline edits to report narrative  
- [ ] `figure/kaggle_snapshot.png` inserted  
- [ ] Team contribution table filled  
- [ ] Code link valid; leader uploads report + code to **E3**  
- [ ] Slides on Drive with correct filename; include snapshot + table + code link  
- [ ] Presentation rehearsed to **≤10 min**  

---

## After presentation (06/02+)

- [ ] Slides may be updated on Drive  
- [ ] Complete peer review  
- [ ] If pipeline plateau: start `EXPERIMENT_LOG.md` §5 (VGG / UNet backlog) for follow-on work (not for report if deadline passed)
