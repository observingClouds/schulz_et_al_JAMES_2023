# Supplemental material to Schulz et al. (submitted)

This repository contains the analysis scripts of Schulz et al. (submitted) and is archived at under the [DOI 10.5281/zenodo.7582494](www.doi.org/10.5281/zenodo.7582494).

The entire analysis can be reproduced (sufficient compute resources provided) with

```
git clone schulz_et_al_JAMES_2023
mamba env create -n schulz_et_al_2023 -f environment.yaml
dvc repro
```

Parts of the analysis ( see `dvc.yaml` for partial analysis names ) can be reproduced by e.g.
```
git clone schulz_et_al_JAMES_2023
mamba env create -n schulz_et_al_2023 -f environment.yaml
dvc repro pattern_cloudfraction_fig16
```

The exact versions of packages that have been used are provided in `environment.yaml.pinned`
