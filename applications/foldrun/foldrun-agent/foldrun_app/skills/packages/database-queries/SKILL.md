---
name: database-queries
description: Querying pre-computed protein structure predictions and variant annotations from AlphaFold DB
metadata:
  adk_additional_tools:
    - query_alphafold_db_prediction
    - query_alphafold_db_summary
    - query_alphafold_db_annotations
---

# Database Queries

- **Check existing structures**: Use query_alphafold_db_summary before running expensive predictions
- **Get detailed predictions**: Use query_alphafold_db_prediction for full structure data
- **Variant annotations**: Use query_alphafold_db_annotations for mutation effects
