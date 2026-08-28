# OpenSplice v3.2 development superset result

**Decision:** `phase_r_negative_stage_a_routes_descriptive_only`

The CPU-only analyzer independently reconstructed all endpoint margins and means from the raw relevant/padding logits. Confirmation model outputs, activations and interventions remained unopened. Later confirmation metadata/labels had been exposed post-freeze, so this is not a claim of complete metadata blindness.

## Gates

- Eligible effects: 12 ({'BRAF': 6, 'SLC25A48': 6})
- Phase-R invalid groups: 0/2592
- Mandatory endpoint-level closures: True
- Complete isolated T/E account: True

No frozen Phase-R candidate passed the development gate.

Development-only computational intervention result. It does not establish an RBP, biochemical pathway, endogenous mechanism, or experimental replication, and it cannot open confirmation.

Raw artifact tree SHA-256: `4171d8aebae7fff3b9981d7ab0dc914c659c6fe2916cb9a48bbee87e205beed8`

## Analyzer-only amendments

The model run used v3.2.0. Offline analysis used the prospective v3.2.1 checkpoint-symlink, v3.2.2 protobuf path-key, v3.2.3 preflight-log, and v3.2.4 exact import-alias repairs. Both import alias rows remain present. No repair changed a scientific gate or permitted a model rerun.
