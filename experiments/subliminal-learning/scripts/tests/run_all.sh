#!/usr/bin/env bash
# Everything that guards the Stage D pipeline, in about a minute, on no GPU.
#
# These exist because the pipeline they cover costs six training runs and most of
# a day of four cards, and because they have twice caught a defect that would
# have spent that budget measuring the wrong thing: once when the splits were
# being cut on a different score than the one being audited, and once when the
# arms default moved to fox-only and dryrun.sh was still asserting two arms.
#
# Each file is standalone and self-locating -- run one directly if you prefer.
# `python test_shard.py` loads a tokenizer; the rest touch no model.
set -u
cd "$(dirname "$0")"
PY=../../../../.venv/bin/python

fails=0
for t in test_splits test_shard test_report test_confounds test_degen_probe \
         test_degen_plot; do
  out=$("$PY" "$t.py" 2>&1)
  last=$(printf '%s\n' "$out" | tail -1)
  if [ "$last" = "ALL CHECKS PASSED" ]; then
    printf '  ok   %s\n' "$t"
  else
    printf '  FAIL %s\n%s\n' "$t" "$out"
    fails=$((fails + 1))
  fi
done

out=$(bash dryrun.sh 2>&1)
last=$(printf '%s\n' "$out" | tail -1)
if [ "$last" = "ALL CHECKS PASSED" ]; then
  printf '  ok   dryrun.sh\n'
else
  printf '  FAIL dryrun.sh\n%s\n' "$out"
  fails=$((fails + 1))
fi

echo
[ "$fails" = 0 ] && echo "ALL SUITES PASSED" || echo "$fails SUITES FAILED"
exit $([ "$fails" = 0 ] && echo 0 || echo 1)
