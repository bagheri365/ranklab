# RankLab v1.0.0 Release Checklist

Use this checklist immediately before creating the `v1.0.0` Git tag and GitHub release.

## Repository state

- [ ] `git status` reports a clean working tree.
- [ ] `main` is up to date with `origin/main`.
- [ ] `pytest` passes all tests.
- [ ] The release commit contains no generated `runs/` artifacts.
- [ ] Raw KuaiRand-Pure data remain untracked.

## Frozen research hashes

Confirm:

```bash
shasum -a 256 research/reporting/M2.4_RESULTS_DISCUSSION.md
```

Expected:

```text
1905dc489908fdb746775b0b5b33e2666adc5225897f548979f3e5d69ce1ae29
```

If local final M1 artifacts are available, also confirm:

```bash
shasum -a 256 \
  runs/m1/final_results/manifest.json \
  runs/m1/final_results/summary.json \
  runs/m1/final_results/FINAL_RESULTS.md
```

Expected:

```text
ba9334455134d86a14d2310c809c34152c0ce60f135bb62a765cdab9a1737c2f
8735976eb8d5562de9728ddc98179f86ec0e5740f4ee09d8b8f04c0b1995ed33
20ccd9083e71945581f7f2d9970404aa9b9287a30ba62a205dedcc083cf799bf
```

## Release tag

Create an annotated tag from the verified release commit:

```bash
git tag -a v1.0.0 -m "RankLab v1.0.0: frozen benchmark, analysis, and reporting release"
git push origin v1.0.0
```

## GitHub release

Create a GitHub release from tag `v1.0.0`.

Recommended title:

```text
RankLab v1.0.0 — Frozen benchmark and research release
```

Use `research/release/RELEASE_NOTES_v1.0.0.md` as the release body.

## Post-release verification

```bash
git status
git tag --list --sort=-creatordate | head
git log --oneline --decorate -5
```

Confirm that `v1.0.0` points to the intended release commit.
