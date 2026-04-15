# FoldRun: Enterprise Cost Estimate (12-Month Projection)

This document provides a cost estimate for a large pharmaceutical or biotech organization migrating protein structure prediction workloads to FoldRun on Google Cloud.

## Workload Assumptions

**Organization profile:**
- ~20-30 active drug programs (antibodies, proteins, peptides)
- Each program screening ~50-200 variants/month
- Mix of quick structure checks (AF2) and complex predictions (OF3, protein-ligand)
- Ramp: lighter months 1-3 during migration, full scale months 4-12

**Estimated monthly job volume at steady state:**

| Job Type | Jobs/month | Rationale |
|---|---|---|
| AF2 monomer (L4 GPU) | ~800 | Variant screening, quick structure checks |
| AF2 monomer (A100, large proteins) | ~100 | Proteins >500 residues |
| OF3 predictions (A100) | ~200 | Protein-ligand complexes, multi-chain |
| Re-runs / parameter sweeps | ~200 | Additional seeds, different MSA methods |

## Always-On Infrastructure

| Component | Monthly Cost | 12-Month Cost |
|---|---|---|
| Filestore (2.5TB BASIC_SSD) | $770 | $9,240 |
| GCS (~1TB backups + growing results) | $25 | $300 |
| Artifact Registry | $5 | $60 |
| Agent Engine | $50 | $600 |
| Cloud Run (viewer) | $20 | $240 |
| VPC / NAT | $45 | $540 |
| **Infrastructure subtotal** | **~$915** | **~$10,980** |

## Compute (GPU Predictions)

Ramp schedule: 25% volume months 1-3 (migration), 100% months 4-12.

| Job Type | Cost/job | Steady-state monthly | 12-month (with ramp) |
|---|---|---|---|
| AF2 on L4 (800/mo) | $4 | $3,200 | $31,200 |
| AF2 on A100 (100/mo) | $10 | $1,000 | $9,750 |
| OF3 on A100 (200/mo) | $13 | $2,600 | $25,350 |
| Re-runs (200/mo) | $8 avg | $1,600 | $15,600 |
| **Compute subtotal** | | **~$8,400** | **~$81,900** |

### Per-job cost breakdown

**AF2 prediction (L4 GPU, typical protein <500 residues):**

| Step | Machine | Duration | Cost |
|---|---|---|---|
| MSA / Data Pipeline | c2-standard-16 | ~35 min | ~$0.50 |
| Predict (5 seeds) | g2-standard-12 + 1x L4 | ~17 min each | ~$2.50 |
| Relax (5 seeds) | g2-standard-12 + 1x L4 | ~5 min each | ~$0.75 |
| **Total** | | | **~$4** |

**OF3 prediction (A100 GPU, protein-ligand complex):**

| Step | Machine | Duration | Cost |
|---|---|---|---|
| MSA Pipeline | c2-standard-16 | ~13 min | ~$0.20 |
| Predict (5 seeds) | a2-highgpu-1g + 1x A100 | ~26 min each | ~$12.50 |
| **Total** | | | **~$13** |

## Other Costs

| Item | 12-Month Cost |
|---|---|
| Gemini API (analysis, ~1,300 calls/mo) | ~$600 |
| Cloud Logging / Monitoring | ~$1,200 |
| Egress (downloading results) | ~$500 |
| Database re-downloads (quarterly refresh) | ~$200 |
| **Other subtotal** | **~$2,500** |

## 12-Month Summary

| Category | 12-Month Cost |
|---|---|
| Infrastructure (always-on) | $10,980 |
| Compute (GPU predictions) | $81,900 |
| Other | $2,500 |
| **Total** | **~$95,000** |
| **Monthly average** | **~$7,900** |

## Cost Optimization Levers

| Lever | Potential Savings |
|---|---|
| **DWS reservations** (committed GPU usage) | 40-60% on compute |
| **Filestore lifecycle** (delete when idle, restore from GCS) | Up to $770/mo during idle periods |
| **Spot/preemptible VMs** for non-urgent batch jobs | 60-70% on compute |
| **Reduced BFD** vs full BFD | Smaller Filestore, faster MSA |

With DWS reservations for committed GPU usage, the 12-month compute cost could drop from ~$82K to ~$35-50K, bringing the total to **~$45-60K/year**.

## Scaling Notes

- Costs scale linearly with job volume — 2x jobs = 2x compute cost
- Infrastructure is mostly fixed regardless of job volume
- Multi-region or HA deployments roughly double infrastructure costs
- Additional models (Boltz) share most databases, adding minimal storage cost

## Comparison: Self-Managed vs FoldRun

Beyond infrastructure costs, FoldRun's agentic orchestration layer reduces the operational burden of managing prediction pipelines. Self-managed alternatives typically require:
- Custom pipeline orchestration and monitoring code
- Manual GPU provisioning and job scheduling
- Separate analysis and visualization tooling
- 1-2 FTEs of engineer/scientist time for pipeline maintenance

FoldRun provides all of this through a conversational AI interface backed by 23 native tools, deployed as a managed agent on Vertex AI.
