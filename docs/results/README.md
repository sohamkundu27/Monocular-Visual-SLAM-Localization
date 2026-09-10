# Published result figures

These images are copied out of `outputs/sequence_XX/` so the project README renders them on
GitHub. Run directories themselves are git-ignored, because they are regenerated on every run
and contain large intermediates.

Regenerate after a benchmark run with:

```bash
python scripts/report.py --publish-figures --update-readme
```

Everything here is produced by an actual run of the pipeline against KITTI Odometry. See
`outputs/sequence_XX/metrics.json` for the corresponding numbers.
