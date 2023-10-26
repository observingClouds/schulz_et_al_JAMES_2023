# Supplemental material to Schulz et al. (2023)
[![Software](https://img.shields.io/badge/Software-10.5281/zenodo.7591545-blue)](https://doi.org/10.5281/zenodo.7591545)
[![Manuscript](https://img.shields.io/badge/Manuscript-10.1029/2023MS003648-blue)](http://dx.doi.org/10.1029/2023MS003648)
[![Model](https://img.shields.io/badge/Model-10.5281/zenodo.7133783-blue)](https://dx.doi.org/10.5281/zenodo.7133783)

This repository contains the analysis scripts of Schulz et al. (2023) and is archived under the [DOI 10.5281/zenodo.7591545](https://doi.org/10.5281/zenodo.7591545).

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
dvc repro fig14_pattern_cloudfraction
```

The exact versions of packages that have been used are provided in `environment.yaml.pinned`

The model output can be accessed via the [EUREC4A-Intake](https://github.com/eurec4a/eurec4a-intake)-catalog but is also archived on the DKRZ tape. The [tape-catalog](https://github.com/observingClouds/tape_archive_index/blob/main/catalog.yml) provides easy access to those files.
