# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Free Embedding Optimization Experiments.


This script finds the maximum number of documents (N) for which a model of a
given dimension (d) can successfully learn to distinguish k relevant documents
for every possible query.


It performs a multi-phase search:
1.  **Galloping Search**: Quickly finds a coarse range for the critical N.
2.  **Binary Search**: Narrows down the critical N within that range.
3.  **Sweep**: Explores the vicinity of the found N for fine-grained results.


Training uses full-batch InfoNCE with all documents as negatives.


Example Usage:
--------------
python free_embeddings_experiment.py --d=4 --k=2 --enable_critical_n_search=11 \
--results_output_path='d=4_k=2.json' --device=gpu
"""

import datetime
import itertools
import json
import os
import time
import traceback
from typing import Any, Dict, List, Optional, Set, Tuple

from absl import app
from absl import flags
from absl import logging
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm


tqdm = tqdm.tqdm
DEFAULT_EXPERIMENT_PARAMS: Dict[str, Any] = {
    "q": None,
    "learning_rate": 0.01,
    "num_iterations": 100000,
    "temperature": 0.1,
    "seed": 42,
    "show_progress": True,
    "device": "gpu",
    "log_interval": 50,
    "early_stopping_patience": 1000,
    "early_stopping_min_delta": 0.00001,
    "early_stopping_monitor_metric": "loss",
    "early_stopping_restore_best_weights": False,
}


# Keys used for creating a unique signature for an experiment run.
RELEVANT_PARAMS_FOR_SIGNATURE: List[str] = sorted(
    list(DEFAULT_EXPERIMENT_PARAMS.keys()) + ["n", "d", "k"]
)


class LoggingFile:
  """A file-like object that redirects tqdm's output to the logging module."""

  def __init__(self, log_level: int = logging.INFO):
    self.log_level = log_level

  def write(self, text: str):
    if stripped := text.strip():
      logging.log(self.log_level, stripped)

  def flush(self):
    pass

  def isatty(self) -> bool:
    return False


class NpEncoder(json.JSONEncoder):
  """Compact JSON encoder for NumPy/JAX arrays and scalars, sets, datetimes."""

  def default(self, o):
    # Arrays
    if isinstance(o, (np.ndarray, jnp.ndarray)):
      return np.asarray(o).tolist()
    # Numpy scalars and 0-d arrays
    try:
      return o.tolist()  # works for numpy scalars/arrays
    except AttributeError:
      pass
    # Sets
    if isinstance(o, set):
      return list(o)
    # Datetime
    if isinstance(o, (datetime.date, datetime.datetime)):
      return o.isoformat()
    return super().default(o)


def compute_similarities(q: jnp.ndarray, d: jnp.ndarray) -> jnp.ndarray:
  """Compute dot product similarity matrix between queries and documents."""
  return jnp.dot(q, d.T)


def create_combinatorial_qrels(
    n: int,
    k: int,
    q_limit: Optional[int] = None,
    seed: int = 42,
    verbose: bool = False,
) -> Optional[Dict[int, Set[int]]]:
  """Create query-document relevance judgments (qrels)."""
  if k > n or n == 0 or k == 0:
    return None

  all_combinations = list(itertools.combinations(range(n), k))
  if q_limit is not None and q_limit < len(all_combinations):
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(all_combinations), q_limit, replace=False)
    selected_combos = [all_combinations[i] for i in indices]
  else:
    selected_combos = all_combinations

  qrels = {i: set(combo) for i, combo in enumerate(selected_combos)}
  if verbose:
    logging.info(
        "Generated %d qrels for n=%d, k=%d (total possible: %d)",
        len(qrels),
        n,
        k,
        len(all_combinations),
    )
  return qrels


def qrels_to_sparse_indices(
    qrels: Dict[int, Set[int]], num_queries: int, num_docs: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Converts a qrels dictionary to sparse COO-like indices."""
  rows, cols, data = [], [], []
  for q_id, doc_ids in qrels.items():
    if q_id < num_queries:
      for doc_id in doc_ids:
        if doc_id < num_docs:
          rows.append(q_id)
          cols.append(doc_id)
          data.append(True)
  return (
      np.array(rows, dtype=np.int32),
      np.array(cols, dtype=np.int32),
      np.array(data, dtype=np.bool_),
  )


def convert_qrels_to_jax_matrix(
    qrels: Dict[int, Set[int]],
    num_queries: int,
    num_docs: int,
    device: Optional[jax.Device] = None,
) -> jnp.ndarray:
  """Convert qrels dictionary to a JAX binary relevance matrix."""
  if num_queries == 0 or num_docs == 0:
    mat = jnp.zeros((num_queries, num_docs), dtype=jnp.bool_)
    return jax.device_put(mat, device) if device else mat

  rows_np, cols_np, data_np = qrels_to_sparse_indices(
      qrels, num_queries, num_docs
  )
  rows_jax, cols_jax, data_jax = (
      jnp.asarray(rows_np),
      jnp.asarray(cols_np),
      jnp.asarray(data_np),
  )
  mat = (
      jnp.zeros((num_queries, num_docs), dtype=jnp.bool_)
      .at[rows_jax, cols_jax]
      .set(data_jax)
  )
  return jax.device_put(mat, device) if device else mat


def sample_list(input_list: List[Any], max_length: int = 100) -> List[Any]:
  """Sample a list to have at most max_length items."""
  if len(input_list) <= max_length:
    return input_list
  indices = np.linspace(0, len(input_list) - 1, max_length, dtype=int)
  return [input_list[i] for i in indices]


def get_params_signature(params_dict: Dict[str, Any]) -> str:
  """Generates a unique JSON signature for a set of parameters."""
  sig_dict = {
      key: params_dict.get(key) for key in RELEVANT_PARAMS_FOR_SIGNATURE
  }
  return json.dumps(sig_dict, sort_keys=True, cls=NpEncoder)


def save_results_incrementally(
    results_list: List[Dict[str, Any]], output_path: Optional[str]
):
  """Saves results to a JSON file."""
  if output_path and jax.process_index() == 0:
    json_str = json.dumps(results_list, indent=None, cls=NpEncoder)
    try:
      output_dir = os.path.dirname(output_path)
      if output_dir:
        os.makedirs(output_dir, exist_ok=True)
      with open(output_path, "w") as f:
        f.write(json_str)
    except OSError as e:
      logging.error("Error during incremental save to %s: %s", output_path, e)


def _build_no_qrels_error_result(
    params: Dict[str, Any], duration: float
) -> Dict[str, Any]:
  """Helper to build a standardized error dictionary when no qrels are generated."""
  return {
      "parameters": params,
      "metrics": {
          "actual_q_generated": 0,
          "final_accuracy": 0.0,
          "best_accuracy_monitored": 0.0,
          "best_loss_monitored": float("inf"),
          "error": "No qrels generated",
          "experiment_duration_seconds": duration,
      },
      "data": {},
  }


def jax_loss_fn(
    q_embeds: jnp.ndarray,
    d_embeds: jnp.ndarray,
    relevance_matrix: jnp.ndarray,
    temperature: float,
) -> float:
  """Vectorized InfoNCE loss averaged over all positive pairs.

  For each query i, logits are dot(Q_i, D_j)/T and loss is - average over
  positives of log softmax probability.

  Args:
    q_embeds: Query embeddings.
    d_embeds: Document embeddings.
    relevance_matrix: A binary matrix indicating relevant documents for each
      query.
    temperature: The temperature scaling factor for the logits.

  Returns:
    The scalar InfoNCE loss value.
  """
  logits = jnp.dot(q_embeds, d_embeds.T) / temperature
  log_probs = jax.nn.log_softmax(logits, axis=1)
  mask = relevance_matrix.astype(log_probs.dtype)
  sum_pos_log_probs = jnp.sum(log_probs * mask)
  num_pos = jnp.sum(mask)
  return -sum_pos_log_probs / jnp.maximum(num_pos, 1.0)


def make_jax_update(
    optimizer: optax.GradientTransformation, temperature: float
):
  """Creates a jitted function for a single optimization update step.

  Args:
    optimizer: An optax gradient transformation (e.g., optax.adam).
    temperature: The temperature parameter used in the InfoNCE loss.

  Returns:
    A jitted function `_update` that takes the current parameters, optimizer
    state, and relevance matrix, and returns the new parameters, new optimizer
    state, and the calculated loss. The embeddings are L2-normalized after
    each update.
  """
  @jax.jit
  def _update(
      params: Tuple[jnp.ndarray, jnp.ndarray],
      opt_state: Any,
      relevance_matrix: jnp.ndarray,
  ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], Any, float]:
    def loss_wrapper(p):
      return jax_loss_fn(p[0], p[1], relevance_matrix, temperature)

    loss_value, grads = jax.value_and_grad(loss_wrapper)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    q_embeds, d_embeds = new_params
    q_embeds = q_embeds / jnp.linalg.norm(q_embeds, axis=1, keepdims=True)
    d_embeds = d_embeds / jnp.linalg.norm(d_embeds, axis=1, keepdims=True)
    return (q_embeds, d_embeds), new_opt_state, loss_value

  return _update


def evaluate_top_k_accuracy(
    q_embeddings: np.ndarray,
    d_embeddings: np.ndarray,
    qrels: Dict[int, Set[int]],
    k: int,
) -> float:
  """Evaluates if the top-k retrieved documents are the relevant ones."""
  similarities = compute_similarities(q_embeddings, d_embeddings)
  correct, total = 0, 0
  for query_id, relevant_docs in qrels.items():
    if query_id >= similarities.shape[0]:
      continue  # Skip padded queries
    k_for_query = k if k is not None else len(relevant_docs)
    if k_for_query == 0:
      continue

    top_k_preds = set(np.argsort(-similarities[query_id])[:k_for_query])
    correct += len(relevant_docs.intersection(top_k_preds))
    total += len(relevant_docs)
  return correct / total if total > 0 else 0.0


def get_device_context(device: Optional[str] = "gpu"):
  """Gets the JAX device context, falling back gracefully."""
  for dev_type in (device, "gpu", "tpu", "cpu"):
    try:
      if dev_type and jax.devices(dev_type):
        logging.info("Using device: %s", dev_type.upper())
        return jax.devices(dev_type)[0]
    except RuntimeError:
      continue
  raise RuntimeError("No suitable JAX devices found (GPU, TPU, or CPU).")


def initialize_embeddings(
    q: int, n: int, d: int, key: jax.random.PRNGKey, device_context
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Initializes and normalizes query and document embeddings."""
  key_q, key_d = jax.random.split(key)
  query_embeddings = jax.random.normal(key_q, (q, d), dtype=jnp.float32)
  doc_embeddings = jax.random.normal(key_d, (n, d), dtype=jnp.float32)
  query_embeddings /= jnp.linalg.norm(query_embeddings, axis=1, keepdims=True)
  doc_embeddings /= jnp.linalg.norm(doc_embeddings, axis=1, keepdims=True)
  return (
      jax.device_put(query_embeddings, device_context),
      jax.device_put(doc_embeddings, device_context),
  )


def optimize_embeddings(
    experiment_data: Dict[str, Any],
    q: int,
    n: int,
    d: int,
    k: int,
    learning_rate: float,
    num_iterations: int,
    temperature: float,
    seed: int,
    show_progress: bool,
    log_interval: int,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    early_stopping_monitor_metric: str,
    early_stopping_restore_best_weights: bool,
) -> Dict[str, Any]:
  """Main optimization loop for the embeddings."""
  qrels = experiment_data["qrels_dict"]
  device = experiment_data["device_context"]
  key = jax.random.PRNGKey(seed)

  params = initialize_embeddings(q, n, d, jax.device_put(key, device), device)
  optimizer = optax.adam(learning_rate)
  opt_state = optimizer.init(params)
  relevance_matrix = convert_qrels_to_jax_matrix(qrels, q, n, device=device)
  jax_update = make_jax_update(optimizer, temperature)

  losses, accuracies = [], []
  best_val = (
      float("inf") if early_stopping_monitor_metric == "loss" else float("-inf")
  )
  best_params, best_loss, best_acc, patience_counter = None, None, None, 0
  max_accuracy_observed: float = 0.0
  monitor_is_loss = early_stopping_monitor_metric == "loss"

  iterator = range(num_iterations)
  if show_progress:
    iterator = tqdm(
        iterator,
        desc=f"Optimizing (n={n},d={d},k={k})",
        file=LoggingFile(),
        leave=False,
    )
  last_eval_acc = None

  for i in iterator:
    params, opt_state, loss = jax_update(params, opt_state, relevance_matrix)

    # If monitoring loss, update patience every iteration using the current loss
    if monitor_is_loss:
      current_metric_iter = float(loss)
      is_better_iter = current_metric_iter < (
          best_val - early_stopping_min_delta
      )
      if is_better_iter:
        best_val = current_metric_iter
        best_loss = float(loss)
        if early_stopping_restore_best_weights:
          best_params = jax.tree_util.tree_map(lambda x: x.copy(), params)
        patience_counter = 0
      else:
        patience_counter += 1
      if patience_counter >= early_stopping_patience:
        logging.info(
            "Early stopping at iteration %d for n=%d (loss monitor).", i, n
        )
        break

    if i % log_interval == 0 or i == num_iterations - 1:
      q_cpu, d_cpu = jax.device_get(params)
      acc = evaluate_top_k_accuracy(q_cpu, d_cpu, qrels, k)
      losses.append(float(loss))
      accuracies.append(float(acc))
      # Track the best accuracy observed at any time during training
      if acc > max_accuracy_observed:
        max_accuracy_observed = float(acc)

      last_eval_loss = float(loss)
      last_eval_acc = float(acc)
      if show_progress:
        iterator.set_postfix_str(  # pylint: disable=attribute-error
            f"loss={last_eval_loss:.4f} acc={last_eval_acc:.4f}", refresh=True
        )

      # --- Early Stopping Logic ---
      if not monitor_is_loss:
        # Only handle accuracy-based early stopping during logging steps
        current_metric = acc
        is_better = current_metric > (best_val + early_stopping_min_delta)
        if is_better:
          best_val, best_loss, best_acc = (
              current_metric,
              float(loss),
              float(acc),
          )
          if early_stopping_restore_best_weights:
            best_params = jax.tree_util.tree_map(lambda x: x.copy(), params)
          patience_counter = 0
        else:
          patience_counter += 1
        if patience_counter >= early_stopping_patience:
          logging.info(
              "Early stopping at iteration %d for n=%d (accuracy monitor).",
              i,
              n,
          )
          break
      if acc >= 1.0:
        logging.info(
            "Achieved 100%% accuracy at iteration %d for n=%d. Stopping.", i, n
        )
        break

  if early_stopping_restore_best_weights and best_params:
    params = best_params

  q_final, d_final = jax.device_get(params)
  final_acc = evaluate_top_k_accuracy(q_final, d_final, qrels, k)

  return {
      "final_accuracy": final_acc,
      "final_loss": losses[-1] if losses else float("nan"),
      "best_loss_monitored": best_loss,
      "best_accuracy_monitored": best_acc,
      "losses": sample_list(losses),
      "accuracies": sample_list(accuracies),
      # Provide robust accuracy signals for downstream search logic
      "max_accuracy_observed": max_accuracy_observed,
      "last_logged_accuracy": last_eval_acc,
  }


def run_experiment_base(config_params: Dict[str, Any]) -> Dict[str, Any]:
  """Base function for running a single embedding optimization experiment."""
  n, _, k, seed = (
      config_params["n"],
      config_params["d"],
      config_params["k"],
      config_params["seed"],
  )
  start_time = time.time()

  qrels = create_combinatorial_qrels(
      n=n, k=k, q_limit=config_params.get("q"), seed=seed + 1
  )
  if not qrels:
    logging.warning("No qrels generated for n=%d, k=%d. Skipping.", n, k)
    return _build_no_qrels_error_result(config_params, time.time() - start_time)

  experiment_data = {
      "qrels_dict": qrels,
      "device_context": get_device_context(config_params.get("device")),
  }

  opt_start = time.time()
  params_for_opt = dict(config_params)
  params_for_opt.pop("q", None)
  opt_results = optimize_embeddings(
      experiment_data=experiment_data, q=len(qrels), **params_for_opt
  )
  opt_duration = time.time() - opt_start

  return {
      "parameters": config_params,
      "metrics": {
          **opt_results,
          "actual_q_generated": len(qrels),
          "optimization_duration_seconds": opt_duration,
          "experiment_duration_seconds": time.time() - start_time,
      },
      "data": {},
  }


def _process_experiment_and_save(
    run_params: Dict[str, Any],
    config_idx_str: str,
    current_signature: str,
    processed_signatures: Set[str],
    results_list: List[Dict[str, Any]],
    output_path: Optional[str],
) -> Dict[str, Any]:
  """Processes a single experiment: checks, executes, saves, and returns result."""
  if current_signature in processed_signatures:
    logging.info("Skipping %s (already processed).", config_idx_str)
    return next(
        (
            r
            for r in results_list
            if get_params_signature(r["parameters"]) == current_signature
        ),
        {},
    )

  logging.info("\n--- Running %s ---", config_idx_str)
  try:
    result = run_experiment_base(config_params=run_params)
    results_list.append(result)
    processed_signatures.add(current_signature)
  except (RuntimeError, ValueError, TypeError) as e:
    logging.error("CRITICAL ERROR for %s: %s", config_idx_str, e, exc_info=True)
    result = {
        "parameters": run_params,
        "metrics": {"error": str(e), "traceback": traceback.format_exc()},
    }
    results_list.append(result)
    processed_signatures.add(current_signature)

  save_results_incrementally(results_list, output_path)
  return result


def _run_critical_n_search_for_config(
    base_params: Dict[str, Any],
    processed_signatures: Set[str],
    results_list: List[Dict[str, Any]],
    output_path: Optional[str],
):
  """Performs a multi-phase search to find the critical N value."""
  logging.info(
      "Starting critical N search for base config: d=%s, k=%s",
      base_params["d"],
      base_params["k"],
  )
  initial_n, k_val = base_params["initial_n"], base_params["k"]
  min_n_boundary, accuracy_threshold = 1, 1.0

  def _pick_best_accuracy(metrics: Dict[str, Any]) -> float:
    # Prefer the maximum accuracy signal available
    candidates = [
        metrics.get("max_accuracy_observed"),
        metrics.get("best_accuracy_monitored"),
        metrics.get("last_logged_accuracy"),
        metrics.get("final_accuracy"),
    ]
    values = [float(v) for v in candidates if isinstance(v, (int, float))]
    return max(values) if values else -1.0

  def _evaluate_n(n_to_eval: int) -> float:
    """Runs an experiment for a given N, reusing results if possible."""
    eval_params = {**base_params, "n": n_to_eval, "k": k_val}
    eval_params.pop("initial_n", None)
    sig = get_params_signature(eval_params)

    for res in results_list:
      if get_params_signature(res.get("parameters", {})) == sig:
        acc = _pick_best_accuracy(res.get("metrics", {}))
        if acc >= 0.0:
          logging.info(
              "Reusing cached accuracy for N=%d: %.4f", n_to_eval, acc
          )
          return acc

    result_entry = _process_experiment_and_save(
        run_params=eval_params,
        config_idx_str=f"N-search eval (N={n_to_eval}, k={k_val})",
        current_signature=sig,
        processed_signatures=processed_signatures,
        results_list=results_list,
        output_path=output_path,
    )
    return _pick_best_accuracy(result_entry.get("metrics", {}))

  # --- Phase 1: Galloping Search ---
  n_low = None
  if _evaluate_n(initial_n) >= accuracy_threshold:
    n_low, step = initial_n, 1
    while True:  # Gallop up
      n_test = n_low + step
      if _evaluate_n(n_test) >= accuracy_threshold:
        n_low = n_test
        step *= 2
      else:
        n_high = n_test
        break
  else:
    n_high, step = initial_n, 1
    while n_high > min_n_boundary:  # Gallop down
      n_test = max(n_high - step, min_n_boundary)
      if _evaluate_n(n_test) < accuracy_threshold:
        n_high = n_test
        step *= 2
        if n_test == min_n_boundary:
          break
      else:
        n_low = n_test
        break
  logging.info(
      "Phase 1 (Gallop) Result: N is between %s and %s", n_low, n_high
  )

  # --- Phase 2: Binary Search ---
  critical_n = n_low
  if n_low is not None and n_high is not None:
    low, high = n_low, n_high
    while low <= high:
      mid = (low + high) // 2
      if mid < min_n_boundary:
        break
      if _evaluate_n(mid) >= accuracy_threshold:
        critical_n, low = mid, mid + 1  # This N is good, try higher
      else:
        high = mid - 1  # This N is bad, try lower
  logging.info(
      "Phase 2 (Binary Search) Result: Critical N is likely %s", critical_n
  )

  # --- Phase 3: Fine-grained Sweep ---
  if critical_n is not None:
    logging.info("Phase 3: Sweeping around N=%s", critical_n)
    for offset in range(-4, 6):  # Explore N-4 to N+5
      _evaluate_n(max(critical_n + offset, min_n_boundary))


FLAGS = flags.FLAGS


# Core parameters for the search
_ = flags.DEFINE_integer("d", None, "Dimension of embeddings.", required=True)
_ = flags.DEFINE_integer(
    "k", None, "Number of relevant documents per query.", required=True
)
ENABLE_CRITICAL_N_SEARCH = flags.DEFINE_integer(
    "enable_critical_n_search",
    None,
    "Starting N for the critical N search.",
    required=True,
)


# Path for saving results
RESULTS_OUTPUT_PATH = flags.DEFINE_string(
    "results_output_path", None, "Path to save the experiment results (JSON)."
)


# Optimizer and training parameters
_ = flags.DEFINE_float(
    "learning_rate",
    DEFAULT_EXPERIMENT_PARAMS["learning_rate"],
    "Learning rate.",
)
_ = flags.DEFINE_integer(
    "num_iterations",
    DEFAULT_EXPERIMENT_PARAMS["num_iterations"],
    "Max optimization iterations.",
)
_ = flags.DEFINE_float(
    "temperature",
    DEFAULT_EXPERIMENT_PARAMS["temperature"],
    "Temperature for InfoNCE loss.",
)
_ = flags.DEFINE_integer(
    "seed", DEFAULT_EXPERIMENT_PARAMS["seed"], "Random seed."
)
_ = flags.DEFINE_string(
    "device",
    DEFAULT_EXPERIMENT_PARAMS["device"],
    "Device to use (gpu, tpu, cpu).",
)


# Logging and Early Stopping
_ = flags.DEFINE_boolean(
    "show_progress",
    DEFAULT_EXPERIMENT_PARAMS["show_progress"],
    "Show tqdm progress bar.",
)
_ = flags.DEFINE_integer(
    "log_interval",
    DEFAULT_EXPERIMENT_PARAMS["log_interval"],
    "Logging interval.",
)
_ = flags.DEFINE_integer(
    "early_stopping_patience",
    DEFAULT_EXPERIMENT_PARAMS["early_stopping_patience"],
    "Patience for early stopping.",
)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # --- 1. Load existing results ---
  loaded_results = []
  output_path = RESULTS_OUTPUT_PATH.value
  if output_path:
    try:
      with open(output_path, "r") as f:
        content = f.read()
      if content.strip():
        loaded_results = json.loads(content)
        logging.info(
            "Loaded %d existing results from %s.",
            len(loaded_results),
            output_path,
        )
    except FileNotFoundError:
      logging.info("Results file not found at %s. Starting fresh.", output_path)
    except (json.JSONDecodeError, OSError) as e:
      logging.warning("Could not load %s: %s. Starting fresh.", output_path, e)

  all_results = loaded_results
  processed_signatures = {
      get_params_signature(res["parameters"])
      for res in all_results
      if "parameters" in res
  }

  # --- 2. Set up base parameters for the search ---
  base_params = {
      **DEFAULT_EXPERIMENT_PARAMS,
      **{
          name: value
          for name, value in FLAGS.flag_values_dict().items()
          if value is not None
      },
  }
  # Special handling for the search parameter
  base_params["initial_n"] = ENABLE_CRITICAL_N_SEARCH.value

  logging.info("JAX devices: %s", jax.devices())

  # --- 3. Run the critical N search ---
  _run_critical_n_search_for_config(
      base_params=base_params,
      processed_signatures=processed_signatures,
      results_list=all_results,
      output_path=output_path,
  )

  # --- 4. Final Save ---
  if output_path and jax.process_index() == 0:
    logging.info(
        "--- Critical N search complete. Saving %d total results. ---",
        len(all_results),
    )
    save_results_incrementally(all_results, output_path)
    logging.info("Final results saved to %s", output_path)


if __name__ == "__main__":
  app.run(main)
