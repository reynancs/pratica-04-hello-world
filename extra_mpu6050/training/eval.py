"""Avaliação do classificador de orientação (float32 x int8).

Adaptado do evaluate.py do exemplo hello_world do tflite-micro:
https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/hello_world/evaluate.py

O que mudou em relação ao hello_world:
- Entrada: o split `test` do CSV gerado pelo train.py, em vez de x aleatório.
- Métrica: acurácia, matriz de confusão e precisão/recall por classe, em vez
  do erro absoluto médio (regressão -> classificação).
- Quantização explícita: o modelo int8 recebe a entrada quantizada
  (round + clip) e a saída é desquantizada -- exatamente o que o firmware faz.
- Vetores de referência ("golden"): imprime a saída int8 esperada para 4
  leituras fixas. O firmware roda as mesmas 4 no boot (self-test); os valores
  no monitor serial do Wokwi devem bater com os daqui.
- Interpretador: usa o runtime do TFLM (pacote `tflite-micro`, só Linux) se
  estiver instalado; senão cai para o interpretador LiteRT/TFLite com kernels de
  referência (Windows / Colab sem o pacote).

Executar (a partir desta pasta, depois do train.py):
    python eval.py
    python eval.py --use_tflite    # força o interpretador LiteRT/TFLite
    python eval.py --no-plot
"""

import argparse
import csv
import json
import logging
import os

import numpy as np
import tensorflow as tf

from train import CLASSES, NUM_FEATURES, TILT_BANDS_DEG

runtime = None
try:
  from tflite_micro.python.tflite_micro import runtime
except ImportError:
  pass

# Mesmo interpretador do upstream (ai_edge_litert / LiteRT). Se não estiver
# instalado, usa o tf.lite do próprio TensorFlow (deprecado a partir do TF 2.20).
try:
  import ai_edge_litert.interpreter as tflite_interp
  from ai_edge_litert.interpreter import OpResolverType
except ImportError:
  tflite_interp = tf.lite
  OpResolverType = getattr(tf.lite.experimental, "OpResolverType", None)

_PREFIX_PATH = os.path.dirname(os.path.abspath(__file__))

# Mesmos vetores do self-test em main/main.cc (kGoldenVectors). Se mudar
# aqui, mude lá.
GOLDEN_VECTORS = [
  ("plana, face p/ cima (0 graus)", [0.0, 0.0, 1.0]),
  ("inclinada 45 graus", [0.707, 0.0, 0.707]),
  ("de lado (90 graus)", [1.0, 0.0, 0.0]),
  ("de cabeca p/ baixo (180 graus)", [0.0, 0.0, -1.0]),
]


def load_split(csv_path, split="test"):
  """Lê as linhas de um split do CSV gerado pelo train.py."""
  x_values, y_values, theta_deg = [], [], []
  with open(csv_path, encoding="utf-8") as f:
    for row in csv.DictReader(f):
      if row["split"] != split:
        continue
      x_values.append([float(row["ax_g"]), float(row["ay_g"]), float(row["az_g"])])
      y_values.append(int(row["label"]))
      theta_deg.append(float(row["theta_deg"]))
  return (
    np.array(x_values, dtype=np.float32),
    np.array(y_values, dtype=np.int32),
    np.array(theta_deg, dtype=np.float32),
  )


def read_io_details(model_path):
  """dtype e (scale, zero_point) da entrada e da saída do modelo."""
  interpreter = tflite_interp.Interpreter(model_path=model_path)
  interpreter.allocate_tensors()
  return interpreter.get_input_details()[0], interpreter.get_output_details()[0]


def quantize_input(x_value, input_details):
  """float -> int8, como no firmware: q = round(x / scale) + zero_point."""
  scale, zero_point = input_details["quantization"]
  if input_details["dtype"] == np.int8 and scale != 0.0:
    q = np.round(x_value / scale) + zero_point
    return np.clip(q, -128, 127).astype(np.int8)
  return x_value.astype(input_details["dtype"])


def dequantize_output(y_raw, output_details):
  """int8 -> float: y = (q - zero_point) * scale."""
  scale, zero_point = output_details["quantization"]
  if output_details["dtype"] == np.int8 and scale != 0.0:
    return (y_raw.astype(np.float32) - zero_point) * scale
  return y_raw.astype(np.float32)


def invoke_tflm_interpreter(input_shape, interpreter, x_input, input_index,
                            output_index):
  interpreter.set_input(np.reshape(x_input, input_shape), input_index)
  interpreter.invoke()
  return np.reshape(interpreter.get_output(output_index), -1)


def invoke_tflite_interpreter(input_shape, interpreter, x_input, input_index,
                              output_index):
  interpreter.set_tensor(input_index, np.reshape(x_input, input_shape))
  interpreter.invoke()
  return np.reshape(interpreter.get_tensor(output_index), -1)


def get_tflm_prediction(model_path, x_values):
  """Saídas brutas (N, n_classes) no dtype do modelo, via runtime do TFLM."""
  input_details, output_details = read_io_details(model_path)
  interpreter = runtime.Interpreter.from_file(model_path)
  input_shape = np.array(interpreter.get_input_details(0).get("shape"))

  raw = np.empty((len(x_values), len(CLASSES)), dtype=output_details["dtype"])
  for i, x_value in enumerate(x_values):
    raw[i] = invoke_tflm_interpreter(
      input_shape, interpreter, quantize_input(x_value, input_details), 0, 0
    )
  return raw, input_details, output_details


def get_tflite_prediction(model_path, x_values):
  """Saídas brutas via interpretador LiteRT/TFLite com kernels de referência (os
  mesmos que o firmware usa com o ESP-NN desligado)."""
  kwargs = {"model_path": model_path}
  if OpResolverType is not None:
    kwargs["experimental_op_resolver_type"] = OpResolverType.BUILTIN_REF
  interpreter = tflite_interp.Interpreter(**kwargs)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()[0]
  input_shape = np.array(input_details["shape"])

  raw = np.empty((len(x_values), len(CLASSES)), dtype=output_details["dtype"])
  for i, x_value in enumerate(x_values):
    raw[i] = invoke_tflite_interpreter(
      input_shape, interpreter, quantize_input(x_value, input_details),
      input_details["index"], output_details["index"],
    )
  return raw, input_details, output_details


def predict(model_path, x_values, use_tflite):
  if use_tflite or runtime is None:
    return get_tflite_prediction(model_path, x_values)
  return get_tflm_prediction(model_path, x_values)


def confusion_matrix(y_true, y_pred, n_classes):
  cm = np.zeros((n_classes, n_classes), dtype=np.int32)
  for t, p in zip(y_true, y_pred):
    cm[t, p] += 1
  return cm


def classification_metrics(y_true, y_pred):
  cm = confusion_matrix(y_true, y_pred, len(CLASSES))
  per_class = {}
  for i, name in enumerate(CLASSES):
    tp = cm[i, i]
    precision = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
    recall = tp / cm[i, :].sum() if cm[i, :].sum() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    per_class[name] = {
      "precision": round(float(precision), 4),
      "recall": round(float(recall), 4),
      "f1": round(float(f1), 4),
    }
  return {
    "accuracy": round(float(np.mean(y_true == y_pred)), 4),
    "confusion_matrix": cm.tolist(),
    "per_class": per_class,
  }


def print_metrics(label, metrics):
  print(f"\n=== {label} ===")
  print(f"Acurácia: {metrics['accuracy']:.4f}")
  header = "real \\ previsto".ljust(18) + "".join(c[:10].rjust(12) for c in CLASSES)
  print(header)
  for name, row in zip(CLASSES, metrics["confusion_matrix"]):
    print(name.ljust(18) + "".join(str(v).rjust(12) for v in row))
  for name, m in metrics["per_class"].items():
    print(f"  {name:<10} precision={m['precision']:.3f} "
          f"recall={m['recall']:.3f} f1={m['f1']:.3f}")


def print_golden_vectors(int8_model_path):
  """Saída esperada do firmware para os vetores fixos do self-test."""
  x_values = np.array([v for _, v in GOLDEN_VECTORS], dtype=np.float32)
  raw, input_details, output_details = get_tflite_prediction(int8_model_path, x_values)
  in_scale, in_zp = input_details["quantization"]
  out_scale, out_zp = output_details["quantization"]

  print("\n=== Vetores de referência (compare com o self-test no Wokwi) ===")
  print(f"input : scale={in_scale:.8f} zero_point={in_zp}")
  print(f"output: scale={out_scale:.8f} zero_point={out_zp}")
  rows = []
  for (desc, vec), out_q in zip(GOLDEN_VECTORS, raw):
    in_q = quantize_input(np.array(vec, dtype=np.float32), input_details)
    cls = CLASSES[int(np.argmax(out_q))]
    print(f"  {desc:<32} in_q={in_q.tolist()} out_q={out_q.tolist()} -> {cls}")
    rows.append({"descricao": desc, "entrada_g": vec, "entrada_int8": in_q.tolist(),
                 "saida_int8": out_q.tolist(), "classe": cls})
  return rows


def plot_results(metrics_by_model, theta_deg, correct_int8, save_path):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
  for ax, (label, metrics) in zip(axes[:2], metrics_by_model.items()):
    cm = np.array(metrics["confusion_matrix"])
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("previsto")
    ax.set_ylabel("real")
    ax.set_title(f"{label} | acc={metrics['accuracy']:.3f}")
    for i in range(len(CLASSES)):
      for j in range(len(CLASSES)):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")

  # Onde o modelo erra: taxa de acerto por faixa de inclinação
  bins = np.arange(0, 185, 10)
  idx = np.digitize(theta_deg, bins) - 1
  acc_bin = [correct_int8[idx == b].mean() if np.any(idx == b) else np.nan
             for b in range(len(bins) - 1)]
  ax = axes[2]
  ax.bar(bins[:-1] + 5, acc_bin, width=9)
  for name in CLASSES[1:]:
    ax.axvline(TILT_BANDS_DEG[name][0], color="red", linestyle="--", linewidth=1)
  ax.set_xlabel("inclinação real theta (graus)")
  ax.set_ylabel("acurácia int8")
  ax.set_ylim(0, 1.05)
  ax.set_title("Acurácia por faixa (tracejado = fronteira de classe)")

  fig.tight_layout()
  fig.savefig(save_path, dpi=120)
  logging.info("Gráficos salvos em %s", save_path)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--use_tflite", action=argparse.BooleanOptionalAction, default=False,
    help="Inferência com o interpretador LiteRT/TFLite em vez do runtime do TFLM.",
  )
  parser.add_argument("--models_dir", default=os.path.join(_PREFIX_PATH, "models"))
  parser.add_argument(
    "--data_csv",
    default=os.path.join(_PREFIX_PATH, "data", "mpu6050_orientacao.csv"),
  )
  parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
  args, _ = parser.parse_known_args()

  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
  if runtime is None and not args.use_tflite:
    logging.warning("Runtime do TFLM (pip install tflite-micro) indisponível; "
                    "usando o interpretador LiteRT/TFLite.")

  x_test, y_test, theta_test = load_split(args.data_csv, "test")
  assert x_test.shape[1] == NUM_FEATURES
  logging.info("Split de teste: %d amostras", len(y_test))

  report = {"interpretador": "tflite" if (args.use_tflite or runtime is None) else "tflm"}
  metrics_by_model, predictions = {}, {}
  for label, filename in (("float32", "orientacao_float.tflite"),
                          ("int8", "orientacao_int8.tflite")):
    model_path = os.path.join(args.models_dir, filename)
    raw, _, output_details = predict(model_path, x_test, args.use_tflite)
    # argmax direto na saída int8: a desquantização é linear e crescente,
    # então não muda qual classe tem o maior valor (o firmware faz o mesmo).
    y_pred = np.argmax(raw, axis=1)
    probs = dequantize_output(raw, output_details)
    metrics = classification_metrics(y_test, y_pred)
    metrics["tamanho_bytes"] = os.path.getsize(model_path)
    metrics["confianca_media"] = round(float(probs.max(axis=1).mean()), 4)
    print_metrics(f"{label} ({metrics['tamanho_bytes']} bytes)", metrics)
    metrics_by_model[label] = metrics
    predictions[label] = y_pred

  agreement = float(np.mean(predictions["float32"] == predictions["int8"]))
  delta = metrics_by_model["int8"]["accuracy"] - metrics_by_model["float32"]["accuracy"]
  print(f"\nConcordância float32 x int8: {agreement:.4f}")
  print(f"Delta de acurácia (int8 - float32): {delta:+.4f}")
  print(f"Tamanho float32 -> int8: {metrics_by_model['float32']['tamanho_bytes']} -> "
        f"{metrics_by_model['int8']['tamanho_bytes']} bytes")

  report.update({
    "amostras_teste": int(len(y_test)),
    "modelos": metrics_by_model,
    "concordancia_float_int8": round(agreement, 4),
    "delta_acuracia_int8_menos_float": round(float(delta), 4),
    "vetores_referencia": print_golden_vectors(
      os.path.join(args.models_dir, "orientacao_int8.tflite")
    ),
  })
  with open(os.path.join(args.models_dir, "eval_report.json"), "w",
            encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

  if args.plot:
    plot_results(metrics_by_model, theta_test, predictions["int8"] == y_test,
                 os.path.join(args.models_dir, "eval_plots.png"))


if __name__ == "__main__":
  main()
