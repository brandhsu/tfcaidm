If you want to reweight or remove certain losses include a keyword `msk-` input in the dataset.yml.

example:

```yaml
_id:
  project: covid_biomarker 
  version: null 
_db: /data/ymls/db-generic-all.yml 
batch:
  fold: 0 
  size: 8 
  sampling:
    cohort-uci: 0.5
    cohort-rsna-pna: 0.5
specs:
  xs:
    dat:
      dtype: float32
      loads: dat-512-xr
      norms: # arr = (arr.clip(min=..., max=...) - shift) / scale | range is [-3, 3] for z-score norm above and below 3 std.
        shift: "@mean" 
        scale: "@std"
      rands:
        shift:
          lower: -0.1 
          upper: +0.1 
        scale: 
          lower: 0.9
          upper: 1.1
      shape: [1, 512, 512, 1]
    msk-pna:
      dtype: float32 
      loads: lng-512-xr
      norms: null
      shape: [1, 512, 512, 1]
    msk-ratio:
      dtype: float32
      loads: null
      norms: null
      shape: [1]
  ys:
    pna:
      dtype: uint8 
      loads: pna-512-xr
      norms: null
      shape: [1, 512, 512, 1]
    ratio:
      dtype: float32 
      loads: ratio 
      norms: null
      shape: [1]
  load_kwargs:
      verbose: false
```

Then point to your configuration in `userdef.py`, this wil eventually be made as a separate yaml to point to when loading the dataset.