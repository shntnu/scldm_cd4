# Reference papers

Author-adjacent reference material for scLDM.CD4. Kept in the repo so future
sessions (human or Claude) have the theoretical context close at hand.

## Files

### [`2026.02.04.703804v1.pdf`](https://www.biorxiv.org/content/10.64898/2026.02.04.703804v1)

**Virtual Cells Need Context, Not Just Scale.**
Dibaeinia, Babu, Knudson, ElSheikh, Wen, Liu, Perera, Khan. bioRxiv 2026.02.04.703804, posted 2026-02-09.

Position paper by the same authors as this repo (`scg_vae` authors in
`pyproject.toml` + Jason Perera from `git log`). Argues that the primary
failure of current "Virtual Cell" models is insufficient *contextual* coverage
rather than insufficient model capacity or data volume. Connects perturbation
prediction to the causal-transportability literature; uses a 22M-cell
immunology dataset as the empirical case study.

This is the theoretical backdrop for scLDM.CD4's explicit donor/time/perturbation
conditioning and context-aware latent tokens.

## Reading inline

The pixi `dev` feature includes `poppler`, so Claude can read these files via:

```bash
pixi run pdftotext -layout papers/<file>.pdf -
```
