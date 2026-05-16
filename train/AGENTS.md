# Repo Agent Notes

This codebase is aiming for a Kaggle competition, where the judgement is based on f1-score on validation set.

## Patch Tool Fallback

- built-in `apply-patch` is not working in current workspace. Prefer direct `bash` editing by default.
- Use direct shell tools such as `sed -i`, `perl -0pi`, or a small here-doc rewrite for most edits in this repo.
- Re-read the edited file after writing.
- Run a lightweight validation step when practical, such as `python -m py_compile`, after editing Python files.

## DOCS
You should always read and update the docs.

- ARCHITECTURE.md: current codebase architecture
- DESIGN.md: core algorithm choice/design to achieve a better score in Kaggle.
- PLAN.md: future plans

## Submission Convention
When producing a final submission CSV for Kaggle, copy it to the project root with the naming pattern:
```
submission_{run_name}_testf1_{score}.csv
```
Example: `submission_bbox02_testf1_0.64516.csv`
