# Unsupervised Cross-Protocol Anomaly Analysis in Mobile Core Networks via Multi-Embedding Models Consensus

This repository contains the **source code** and **supplementary material** accompanying an anonymous submission on cross-protocol anomaly analysis in mobile core networks.

To preserve double-blind review, this repository intentionally omits author names, affiliations, and proprietary data.

---

## Repository structure

- `source_code/`  
  Python code used to:
  - Fuse SS7, Diameter, and GTP signalling into per-subscriber, per-minute fused records.
  - Construct protocol-plausible synthetic anomalies using mutation via field-group swaps across fused records.
  - Embed fused records using multiple embedding models.
  - Apply unsupervised anomaly detectors per embedding model.
  - Compute consensus scores (number of embedding models flagging a record as anomalous).
  - Generate the statistics and tables reported in the paper (e.g., consensus histograms, threshold trade-offs).

  See the docstrings and inline comments in the individual files for script-level details and expected inputs/outputs.

- `supplementary_details/`  
  Additional documentation that complements the main paper, including:
  - A detailed description of the mutation families and field groups used to construct synthetic anomalies.
  - Protocol-level rationale for each mutation (SS7, Diameter, and GTP).
  - Additional explanatory material that did not fit into the main paper.

- `LICENSE`  
  MIT license with an anonymous holder name (`icst2681`).

---

## Data availability

The **underlying signalling data are not included** in this repository.

The dataset consists of operational mobile-core signalling logs (SS7, Diameter, and GTP) containing sensitive subscriber and operator information. These data are:

- Proprietary to the network operator(s),
- Subject to confidentiality and contractual restrictions, and
- Classified as personal and/or commercially sensitive data under applicable privacy and data-protection regulations.

As a result, we cannot redistribute the raw dataset or derived per-subscriber fused records. The code is provided to document the methodology and to facilitate adaptation to other datasets under appropriate agreements.

---

## Code usage

The code is intended for researchers and practitioners with access to their own mobile-core signalling data.

Typical usage pattern:

1. **Prepare fused records**  
   Export signalling logs (SS7, Diameter, GTP) from your environment, and construct per-subscriber, per-minute fused records matching the schema expected by the scripts in `source_code/`.

2. **Generate synthetic anomalies**  
   Use the mutation scripts to create protocol-plausible synthetic anomalies via field-group swaps, similar to those described in the supplementary document.

3. **Embed fused records**  
   Run the embedding scripts to serialize each fused record to text and obtain vector representations using several embedding models.

4. **Run unsupervised anomaly detectors**  
   Apply the anomaly detection scripts to obtain per-model anomaly scores and binary decisions.

5. **Compute consensus scores and statistics**  
   Aggregate per-model decisions into a consensus score per record, analyze distributions across thresholds, and reproduce tables/plots analogous to those in the paper.

Because the dataset is not distributed, paths, environment variables, and model choices may need to be adapted to your local environment.

