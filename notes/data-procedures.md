# Data pipeline al neki

1. Dobi dostop do podatkov (fastMRI neki neki)

2. Prenesi podatke (*tar.xz s curl al nekaj)

3. Unzippaj in zippaj nazaj v *zip format

```bash
zip -r <ime.zip> <source-dir> --exclude <kar-nočeš>`
```

4. Preveri velikost datoteke

```bash
du -sh <ime.zip>
```

5. Prenesi kaggle api

```bash
pip install kaggle
```

6. Zrihtaj si kaggle access token in ga daj v `~/.kaggle/kaggle.json`

7. `cd` into data directory z zippom

8. ustvari `dataset-metadata.json`

```bash
kaggle datasets init -p .
```

9. Popravi `dataset-metadata.json` podatke

```bash
# example:
{
  "title": "FastMRI Dataset",
  "id": "janpanjan/fastmri",
  "licenses": [
    {
      "name": "CC0-1.0"
    }
  ]
}
```

10. Uploadaj private dataset

```bash
kaggle datasets create -p .
```
