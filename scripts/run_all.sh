#!/bin/bash
# Runs the full experiment queue; each step is skipped if its result file already exists.
cd "$(dirname "$0")"
PY=../../.venv/Scripts/python.exe
R=../results
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
[ -f $R/ner_pubmedbert_results.json ] || $PY ner_bert.py pubmedbert --epochs 2 > $R/log_ner_pubmedbert.txt 2>&1
[ -f $R/re_results.json ] || RE_EPOCHS=2 $PY relations.py > $R/log_re.txt 2>&1
[ -f $R/ner_bert-base_results.json ] || $PY ner_bert.py bert-base --epochs 2 > $R/log_ner_bert-base.txt 2>&1
echo ALL_DONE > $R/ALL_DONE
