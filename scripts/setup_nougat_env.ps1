# One-time setup for the isolated Nougat extraction environment. Not run
# automatically. Nougat (nougat-ocr, last released 2023-10-04 as version
# 0.1.17) is incompatible with this project's main environment's modern
# transformers/albumentations/pypdfium2 versions - a direct in-process
# import attempt found six independent, unrelated version-drift failures
# across three fix rounds (see docs/superpowers/specs/2026-08-22-nougat-
# math-extraction-design.md's "Amendment" section for the full history).
# This isolated venv is pinned to period-correct versions instead, so it
# never shares (and never fights) the main project's dependency graph.
#
# Pinned versions and why (Step 1 research, all verified against the real
# facebookresearch/nougat GitHub repo and PyPI, not guessed):
#
# - nougat-ocr==0.1.17
#     The exact version this project targets. Confirmed via PyPI JSON API
#     (upload date 2023-10-04T09:29:52Z) and cross-checked against the
#     facebookresearch/nougat commit that bumped nougat/_version.py to
#     "0.1.17" (commit 47c77d70727558b4a2025005491ecb26ee97f523, dated
#     2023-10-04T09:28:53Z - matches the PyPI upload time to the minute).
#
# - transformers==4.38.2
#     nougat-ocr 0.1.17's own setup.py at that exact release commit pins
#     only "transformers>=4.25.1" (no upper bound). But the *current*
#     facebookresearch/nougat main branch setup.py (fetched live during
#     this task) has since been tightened by the upstream maintainers to
#     "transformers>=4.25.1,<=4.38.2" - i.e. the project's own authors
#     later added this exact upper bound in response to the same kind of
#     transformers-API-drift breakage this isolation is working around
#     (PretrainedConfig rename, post_init() requirement, etc., all first
#     appearing well after 4.38.2). Pinning the top of that maintainer-
#     verified range gives the newest transformers nougat's own authors
#     confirmed compatible, while staying well clear of the 5.x breakage.
#
# - albumentations==1.4.24
#     Same story: nougat-ocr 0.1.17's release-time setup.py only declares
#     "albumentations>=1.0.0", but upstream's current main branch has
#     since tightened this to "albumentations>=1.0.0,<=1.4.24" - and
#     1.4.24 is also the exact version already confirmed working in this
#     session's earlier direct-import investigation (albumentations 2.x
#     rewrote ImageCompression's constructor signature, breaking
#     nougat/transforms.py's module-level code).
#
# - pypdfium2==4.24.0
#     nougat-ocr declares no version bound at all for pypdfium2, and
#     PdfDocument.render() - which nougat's rasterize_paper relies on -
#     was deprecated starting in 4.25.0 (released 2023-12-10) and removed
#     entirely in 5.0.0. 4.24.0 (released 2023-11-10) is the newest
#     release before that deprecation window opens, keeping this
#     comfortably inside nougat's own 2023 era. (All pypdfium2 wheels are
#     "py3-none-<platform>" - not tied to a CPython minor version - so
#     this pin is unaffected by which Python this venv's interpreter is.)
#
# - timm==0.5.4 (pulled in transitively via nougat-ocr's own hard pin)
#     Verified via PyPI JSON API that 0.5.4 ships only as a pure-Python
#     "py3-none-any" wheel - i.e. it carries no C-extension/Python-version
#     coupling, so it installs cleanly under this machine's Python 3.13
#     despite being a 2021-era release.
#
# - datasets[vision]==2.14.5
#     nougat-ocr declares an unbounded "datasets[vision]" dependency. Left
#     unpinned, pip's resolver spends 10+ minutes backtracking through
#     every datasets release from 2.14 up to 5.0 (each pulling in
#     different dill/multiprocess/pyarrow ranges) before giving up -
#     observed live during this task's own bootstrap run. 2.14.5 (released
#     2023-09-06, the newest datasets release before nougat's own
#     2023-10-04 release date) is the period-correct pin: it resolves
#     immediately and keeps datasets contemporaneous with nougat itself.
#
# torch itself is intentionally left unpinned: nougat-ocr does not declare
# it directly (pytorch-lightning, one of nougat's own dependencies, pulls
# it in transitively), and this machine only has Python 3.13 available -
# old torch wheels contemporaneous with nougat's 2023 release predate
# CPython 3.13 wheel support entirely, so pinning an old torch version
# would simply fail to install here. Letting pip resolve torch picks the
# newest release compatible with both transformers==4.38.2 (which has no
# torch upper bound) and this interpreter's Python 3.13 (resolved to
# torch==2.13.0 when this script was last verified).

$ErrorActionPreference = "Stop"

python -m venv .nougat-venv
& .nougat-venv\Scripts\python.exe -m pip install --upgrade pip
& .nougat-venv\Scripts\pip.exe install `
    "nougat-ocr==0.1.17" `
    "albumentations==1.4.24" `
    "transformers==4.38.2" `
    "pypdfium2==4.24.0" `
    "datasets[vision]==2.14.5"

Write-Host "Nougat environment ready at .nougat-venv/"
