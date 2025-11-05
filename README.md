# MEDS Visualizations

[![DOI](https://zenodo.org/badge/987265452.svg)](https://doi.org/10.5281/zenodo.17536407)

Visualization tools for MEDS datasets.

> [!WARNING]
> This is a work in progress. The API and functionality are very likely change as we develop the library.

## Installation

```bash
pip install MEDS_visualizations
```

## Usage

In a Jupyter notebook, you can load whatever combination of data extractor and plotter you want:

```python
from MEDS_visualizations.extractors import CodeFrequency
from MEDS_visualizations.plotters import Bar
from MEDS_visualizations.visualization import Visualization

CF = CodeFrequency(as_proportions=True)
P = Bar(top_k=10, y_cols=["n_occurrences"])

V = Visualization(extractor=CF, plotter=P)
V.render(data_shards)
```

In the future, we anticipate

- Registering extractors and plotters via pypi entry points.
- Adding the capability to chain together arbitrary extractors and plotters to make a report in a
    visualization.
- Adding the capability to apply arbitrary filters or transformations to all data shards used to power a
    visualization.
