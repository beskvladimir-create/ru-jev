# What the seed_aug configs actually trained on

`seed_aug_s2.yaml` and `seed_aug_s3.yaml` say they differ from `v1_3.yaml` in
the seed field alone, and that the data were not rebuilt. Both statements are
true of the configs and misleading about the runs. The data build that was on
disk, and was sent to the rented machine, predates commit `96fb67f`
(2026-11-03 03:42), which added the transliterated and clipped intent labels
that make up v1.3. That build has 120 distinct intent surface forms (English
key and Russian word); the v1.3 build has 240.

So `seed-aug-s2`, `seed-aug-s2-ep3` and `seed-aug-s3` are the v1.3 schedule
and seeds on a v1.2-level mixture, without the transliterated and clipped
intent forms. They are not replicates of the released v1.3, and the released
v1.3 has no valid second seed. The evidence is in `results/seeds/FINDINGS.md`.
The configs are left unchanged because their hashes are recorded.
