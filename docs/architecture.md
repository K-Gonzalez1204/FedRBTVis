# FedRBTVis Architecture

The owned workbench is a local-first experiment system with a strict one-way
dependency order:

`experiment core -> ArtifactStore/RunManager -> FastAPI -> React frontend`

## Core

`app/backend/fedrbtvis` contains the pure experiment engine. It owns typed
configuration, deterministic partitions, symmetric label noise,
`categorical_emd_01`, LID MLE, TinyCNN/ResNet18 models, local training, real
evaluation, and sample-weighted FedAvg. The core does not import FastAPI or
any web framework.

## Run service

`RunManager` and `ArtifactStore` turn one immutable `RunConfig` into an ordered
event stream and versioned artifacts. A `Run` is the smallest execution unit;
a `Study` expands a whitelisted factor grid into sequential Runs. Events follow
schema v1 and are persisted before broadcast. Terminal manifests record every
artifact path, byte count, and SHA-256.

## HTTP and WebSocket

FastAPI is the only network entry point. It exposes presets, run creation and
listing, event replay, client/aggregation metrics, studies, and the hash-gated
legacy observation endpoint. WebSocket clients reconnect with
`after_sequence` and never invent events.

## Frontend

The React/TypeScript/Vite app consumes only the FastAPI contract. It provides a
Run Launcher, a truthful Run Monitor, and an analysis panel with source badges
for `fixture`, `fresh`, and `legacy` data. Fixture observations are hidden by
default and explicitly marked as non-research.

## Presets

- `test-fixture`: synthetic tensors and TinyCNN for automated tests only.
- `research-lite`: CIFAR-10 and a CIFAR-shaped ResNet18 for local demos.
- `historical-compatible`: CIFAR-10 with the 100+25 structure, expressed for
  compatibility checks, not exact reproduction.

## Verification

Backend: `python -W error::RuntimeWarning -m unittest discover -s app/backend/tests -v`
and `python -m compileall -q app/backend/fedrbtvis app/backend/tests`.

Frontend: `npm test -- --run` and `npm run build`.

Release: `python scripts/verify_owned_release.py`.
