# examples/

Two kinds of programs live here. Everything is **opt-in**: nothing in this
folder runs during `cargo build`, `cargo test`, or `cargo bench`.

## Learn the API

- `basic_usage.rs` — end-to-end tour of the public Rust API (Stieltjes
  transform → RIE shrinkage factors → correlation cleaning). Start here.

## Measurement & profiling tools

Not demos — these generate canonical datasets or support performance work.
Their outputs feed `scripts/` plots and the numbers recorded under
`docs/pareto/`:

- `measure_pareto_frontier.rs` — runtime × accuracy × parallelism sweep
  feeding `scripts/build_pareto_table.py` (regenerates the dispatch table).
- `measure_small_p_crossover.rs` — locates the O(p²)-vs-ChebCode crossover
  below p=1000 (`docs/pareto/small_p.json`).
- `measure_fft_order_sweep.rs` — accuracy/speed landscape of the fft5 grid
  transfer order; analyzed by `scripts/analyze_order_sweep.py`.
- `measure_batch_gamma_sweep.rs` — batch-vs-loop benchmark for the ChebCode
  η-sweep workflow.
- `measure_eta_sweep.rs` — η regularization study (bias, stability,
  runtime); analyzed in `docs/eta_choice.md`.
- `ab_quick_timing.rs` — quick single-method A/B timing from the CLI.
- `profile_hot_loop.rs` — infinite loop for `sample`-based profilers
  (prints its PID first; kill it when done).

Run any of them with `cargo run --release --example <name> -- [args]`.
