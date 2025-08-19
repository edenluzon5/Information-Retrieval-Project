# Information Retrieval Project

## Overview
This project implements a retrieval engine over the English Wikipedia (updated to **August 2021**).  
The objective is to **maximize MAP@40** for the predicted results.

## Approach
- **Weighted BM25** scoring
- **Title weighting** (boosting matches appearing in page titles)
- **PageRank** score per candidate document to further boost relevant results

## Repository Contents
- `createdInvertedIndexGCP.ipynb` — Jupyter notebook for building the inverted indexes.  
- `inverted_index_gcp.py` — Python module that builds the inverted indexes.  
- `paths_to_anchor_bins.txt` — Paths to `.bin`/`.pkl` files for the **anchor** inverted index.  
- `paths_to_body_bin.txt` — Paths to `.bin`/`.pkl` files for the **body** inverted index.  
- `paths_to_titles_bins.txt` — Paths to `.bin`/`.pkl` files for the **title** inverted index.  
- `run_frontend_in_gcp.sh` — Script to create/start a frontend instance on Google Cloud.  
- `search_backend.py` — Implementations of the engine’s retrieval methods.  
- `search_frontend.py` — Exposes six retrieval methods: `search`, `search_body`, `search_title`, `search_anchor`, `get_pagerank`, `get_pageview`.  
- `startup_script_gcp.sh` — Bootstrap script to install required Python dependencies on a GCP instance.

## Dataset
- **Corpus:** English Wikipedia, up to **August 2021**.
