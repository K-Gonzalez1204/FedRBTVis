# FedRBTVis Research-Lite Smoke

- Date: 2026-08-02
- Status: EXECUTED as a controlled cooperative stop; full 15-client
  completion was not awaited on CPU.
- Subrepo commit: `7d22065` (C Task 5 commit)
- Device: local Windows, CPU only
- CIFAR-10: copied from the read-only course cache
  `D:/personal/Lectures of Chen/SUSTech/2023-2024 上/创新实践/FedCorr-main-final/data/cifar10/cifar-10-batches-py`;
  validated with `torchvision` `download=False` as train=50000, test=10000.
  `data/` is ignored by `.gitignore`.
- Run: `a642d214-7391-42af-9d63-3ccf09f38b76`
- Config: `research-lite`, seed 7, cycles 1, local_epochs 1,
  clients_per_step 15, checkpoint_policy `server-only`, CPU.
- Events observed: `run.started` -> `client.started` -> `client.completed`
  (x2) -> `run.stop_requested` -> `run.stopped`.
- Real metrics observed: background client 3 test_accuracy 0.1027,
  test_loss 2.2957; probe client 10 test_accuracy 0.1003,
  test_loss 2.3013. These are early-training CPU metrics and must not be
  treated as research conclusions or resume facts.
- Stop: `POST /stop` returned 202; terminal manifest status `stopped`;
  `events.jsonl`, `client_updates.csv`, `aggregations.csv` (empty),
  `partitions.json` and `server-final.pt` were persisted with a verified
  inventory and hashes.
- Frontend: Vite dev server started; `http://127.0.0.1:5173` returned 200
  and the page contained the FedRBTVis title.
- Limitation: full completion was intentionally not awaited because each
  client evaluates the full 10,000-image test set on CPU, making a full
  15-client run very slow. A complete run should be performed on a GPU or
  with an explicit longer time budget before any performance claim.
