# NC-081: a neural concept selector guiding verified search

A small network reads a goal's examples and chooses 12 of 64 library concepts for search to start from. It executes nothing and accepts nothing. This is implementer evidence, not independent verification. Outcome: NO_GAIN; see `CANDIDATE_MANIFEST.json`.

    OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m unittest -q artifacts/NC-081/test_selector.py
    OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-081/selector.py --libraries 11=artifacts/NC-081/library_seed11.jsonl 19=artifacts/NC-081/library_seed19.jsonl 29=artifacts/NC-081/library_seed29.jsonl --exam-goals 300 --out /tmp/nc081-replay

Dependencies: Python 3, NumPy, PyTorch CPU, and `logos-autonomy/discovery/library.py`, imported read-only at the hash in the manifest.
