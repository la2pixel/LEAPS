"""Action-representation model scripts (synergy AEs, rollout extractors).

The runnable scripts here (train_speed_decoder.py, train_phase_vae.py,
get_synergies.py, extract_rollout_*.py) are `python -m leaps.models.<name>`
entry points and pull what they need directly from leaps.synergy_common /
leaps.data.

Was re-exporting ActionModel/HausdorferAE/... from old_code.py "while models/
is redesigned"; that re-export broke on a missing leaps.evaluation import and
nothing consumed it (grep: no `from leaps.models import` anywhere), so it was
dropped 2026-08-27. old_code.py is still on disk for reference.
"""
