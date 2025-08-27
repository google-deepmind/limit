# On the Theoretical Limitations of Embedding-based Retrieval

This repository contains the official resources for the paper "[On the Theoretical Limitations of Embedding-based Retrieval](https://arxiv.org/abs/TODO)".
This work introduces the **LIMIT** dataset,
designed to stress-test embedding models based on theoretical principles.
We show that for any given embedding dimension `d`,
there exists a combination of documents that cannot be returned by any query.
We use this theory to instantiate the dataset LIMIT,
finding that even state-of-the-art models struggle: highlighting a fundamental
limitation of the current single-vector embedding paradigm.

![LIMIT Dataset Concept](assets/LIMIT.png)

## Overview

* [Data](#data)
* [Code](#code)
* [Evaluation](#evaluation)
* [Citation](#citation)
* [License and disclaimer](#license-and-disclaimer)

## Data

The datasets used in our experiments are available in the `data/` directory of this repository, formatted in [MTEB](https://github.com/embeddings-benchmark/mteb) style (i.e. json lines).

Each dataset contains:

- A `queries.json` file containing a line for each of the 1000 queries, each with an `_id` and the `text` field.
- A `corpus.json` file containing a line for each of the 50k (or 46 if using the `small` version) documents, each with an `_id`, `text` and empty `title` field.
- A `qrels.json` file containing rows for each of the 2000 relevant query->doc mappings, mapping `query-id` of the queries into the `corpus-id` in the documents, with `score` indicating relevance.

* **Full Dataset (`limit`):** The complete dataset, containing 50k documents.
  * [Link to `data/limit`](./data/limit)

* **Small Sample (`limit-small`):** A smaller version with only the 46 documents relevant to the queries.
  * [Link to `data/limit-small`](./data/limit-small)

## Code

We provide code to generate the LIMIT style datasets,
as well as to run the free embedding experiment in the `code/` folder.

* **Dataset Generation:** To generate the dataset from scratch, you can use the Jupyter notebook located at `code/generate_limit_dataset.ipynb`. This contains all necessary steps and dependencies.
  * [Link to `code/generate_limit_dataset.ipynb`](./code/generate_limit_dataset.ipynb)

* **Free Embedding Experiments:** The script to run the free embedding experiments can be found in `code/free_embedding_experiment.py`.
  * [Link to `code/free_embedding_experiment.py`](./code/free_embedding_experiment.py)

If you use the free embedding code,
you'll need to install the following requirements.

### Installation

We recommend using the [`uv` package manager](https://docs.astral.sh/uv/getting-started/installation/).

```bash
# Create a virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r https://raw.githubusercontent.com/google-deepmind/limit/refs/heads/main/code/requirements.txt
```

## Evaluation

Evaluation was done using the [MTEB framework](https://github.com/embeddings-benchmark/mteb). Please see their Github for details.

## Citation

If you use this work, please cite the paper as:

```
@misc{weller2025theoretical,
      title={On the Theoretical Limitations of Embedding-based Retrieval},
      author={Orion Weller and Michael Boratko and Iftekhar Naim and Jinhyuk Lee},
      year={2025},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```

## License and disclaimer

Copyright 2025 Google LLC

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
