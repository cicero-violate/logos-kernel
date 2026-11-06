# Operator-side reference lookup for NC-048

`lookup.json` records what was retrieved at `retrieved_utc`: each reference's HEAD commit, GitHub's detected
licence, the licence file at that exact commit with its SHA-256, and DreamCoder's submodule pins. The licence
files here are unmodified copies. No repository was cloned, and no code or model weights were downloaded.

This is input for the implementer, not independent verification. Check it before relying on it.

| Reference | Commit | Licence file at that commit |
|---|---|---|
| alphageometry | `6777cb586c` | Apache-2.0 |
| TSCL | `ec41eea6aa` | **None found; repository has no licence file** |
| teachDeepRL (ALP-GMM) | `747cda2b90` | MIT |
| DreamCoder `ec` | `cb0e63f5c3` | MIT |
| DreamCoder submodules `pregex`, `pinn`, `pyccg` | see `lookup.json` | **None found at the pinned commits** |
| Stitch | `350804b7b3` | MIT |
| Z3 | `4ef6ad78b6` | MIT text (GitHub reports NOASSERTION because of the Z3 header) |

These are file checks, not legal advice. Absence was checked by these file names only. Main repositories: `LICENSE`,
`LICENSE.md`, `LICENSE.txt`, `LICENSE.rst`, `COPYING`, `LICENCE`, `COPYING.md`. Submodules: `LICENSE`, `LICENSE.md`,
`LICENSE.txt`, `COPYING`. A licence stated elsewhere, such as in a README or source header, was not checked.
Model weights, datasets and other assets may carry separate terms and were not examined.
