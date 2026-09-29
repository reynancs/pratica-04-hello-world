"""Treino do classificador de orientação (MPU6050) para o TFLite Micro.

Adaptado do train.py do exemplo hello_world do tflite-micro:
https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/hello_world/train.py

O que mudou em relação ao hello_world:
- get_data(): em vez de x -> sin(x), gera leituras sintéticas do acelerômetro
  do MPU6050 (ax, ay, az em g) rotuladas pelo ângulo de inclinação.
- create_model(): de regressão (1 saída, MSE) para classificação (3 classes,
  softmax, cross-entropy).
- A quantização int8 (no upstream fica em um script separado, ptq.py) foi
  incluída aqui, junto com a geração do array C (substitui o `xxd -i`, que
  não existe no Windows).

Executar (a partir desta pasta):
    python train.py
    python train.py --epochs 300 --seed 7

Saídas:
    data/mpu6050_orientacao.csv    dataset completo, com a coluna `split`
    models/orientacao_float.tflite modelo float32 (referência)
    models/orientacao_int8.tflite  modelo int8 (o que vai para o ESP32-S3)
    models/model.keras             modelo Keras (com --save_tf_model)
    models/train_summary.json      hiperparâmetros, tamanhos e acurácias
    ../main/model.cc               array C `g_model` embarcado no firmware
"""

import argparse
import json
import logging
import os

import numpy as np
import tensorflow as tf

_PREFIX_PATH = os.path.dirname(os.path.abspath(__file__))

# Contrato com o firmware: a ORDEM das classes é o índice de saída do modelo.
# Se mudar aqui, mude também kLabels em main/orientation_classifier.cc.
CLASSES = ["REPOUSO", "INCLINADO", "INVERTIDO"]

# Faixas do ângulo de inclinação theta (graus) entre o vetor medido e o eixo +Z
# do sensor. theta = 0 -> placa deitada, face para cima (az = +1 g).
TILT_BANDS_DEG = {
    "REPOUSO": (0.0, 30.0),
    "INCLINADO": (30.0, 120.0),
    "INVERTIDO": (120.0, 180.0),
}

NUM_FEATURES = 3  # ax, ay, az (g)


def get_data(samples_per_class=600, noise_g=0.02, magnitude_std=0.03, seed=42):
  """Gera o dataset sintético do acelerômetro, balanceado por classe.

  Física: parado, o acelerômetro mede só a reação à gravidade, um vetor de
  módulo 1 g. A orientação da placa muda a DIREÇÃO desse vetor:
      a = (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta))
  theta = inclinação em relação a +Z; phi = para que lado inclinou.

  Realismo (o que torna a tarefa não trivial):
  - magnitude_std: variação do módulo (vibração, aceleração linear leve);
  - noise_g: ruído branco por eixo (ordem de grandeza do MPU6050 em +/-4 g).
  O rótulo vem do theta REAL, antes do ruído: perto das fronteiras (30 e 120
  graus) o ruído cria ambiguidade, como aconteceria com dados reais.

  Returns:
      (x_values, y_values, theta_deg): features (N, 3) float32, rótulos (N,)
      int32 e o ângulo verdadeiro (N,) para análise no eval.py.
  """
  rng = np.random.default_rng(seed)
  x_parts, y_parts, theta_parts = [], [], []

  for label, name in enumerate(CLASSES):
    lo, hi = TILT_BANDS_DEG[name]
    # theta uniforme DENTRO da faixa -> classes balanceadas.
    # (Uniforme na esfera deixaria REPOUSO com só ~7% das amostras.)
    theta = np.deg2rad(rng.uniform(lo, hi, samples_per_class))
    phi = rng.uniform(0.0, 2.0 * np.pi, samples_per_class)
    magnitude = rng.normal(1.0, magnitude_std, samples_per_class)

    acc = np.stack(
      [
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
      ],
      axis=1,
    ) * magnitude[:, None]
    acc += rng.normal(0.0, noise_g, acc.shape)

    x_parts.append(acc)
    y_parts.append(np.full(samples_per_class, label))
    theta_parts.append(np.rad2deg(theta))

  x_values = np.concatenate(x_parts).astype(np.float32)
  y_values = np.concatenate(y_parts).astype(np.int32)
  theta_deg = np.concatenate(theta_parts).astype(np.float32)

  # Embaralha para as classes não ficarem em blocos
  order = rng.permutation(len(y_values))
  return x_values[order], y_values[order], theta_deg[order]


def split_data(y_values, val_frac=0.15, test_frac=0.15, seed=42):
  """Split estratificado train/val/test. Retorna um array de strings."""
  rng = np.random.default_rng(seed + 1)
  split = np.empty(len(y_values), dtype=object)
  for label in np.unique(y_values):
    idx = rng.permutation(np.flatnonzero(y_values == label))
    n_test = int(round(len(idx) * test_frac))
    n_val = int(round(len(idx) * val_frac))
    split[idx[:n_test]] = "test"
    split[idx[n_test:n_test + n_val]] = "val"
    split[idx[n_test + n_val:]] = "train"
  return split


def save_dataset_csv(path, x_values, y_values, theta_deg, split):
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "w", encoding="utf-8", newline="\n") as f:
    f.write("ax_g,ay_g,az_g,theta_deg,label,label_name,split\n")
    for (ax, ay, az), label, theta, sp in zip(x_values, y_values, theta_deg, split):
      f.write(
        f"{ax:.5f},{ay:.5f},{az:.5f},{theta:.2f},{label},{CLASSES[label]},{sp}\n"
      )
  logging.info("Dataset salvo em %s", path)


def create_model() -> tf.keras.Model:
  model = tf.keras.Sequential()

  # Entrada: 3 features (ax, ay, az) em g. Sem normalização: os valores já
  # ficam em ~[-1.2, 1.2], e a escala int8 é calibrada pelo representative
  # dataset na quantização.
  model.add(tf.keras.Input(shape=(NUM_FEATURES,)))

  # Duas camadas ocultas de 8 neurônios. A fronteira "theta = 30 graus" é um
  # cone em volta de +Z: não é linearmente separável em (ax, ay, az), por isso
  # as camadas ocultas com ReLU.
  model.add(tf.keras.layers.Dense(8, activation="relu"))
  model.add(tf.keras.layers.Dense(8, activation="relu"))

  # Uma saída por classe; softmax transforma em probabilidades
  model.add(tf.keras.layers.Dense(len(CLASSES), activation="softmax"))

  # Classificação multiclasse com rótulos inteiros -> sparse categorical CE
  model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
  )
  return model


def convert_tflite_model(model):
  """Converte o modelo Keras para .tflite float32 (igual ao upstream)."""
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  return converter.convert()


def convert_quantized_tflite_model(model, x_train, num_samples=500):
  """Quantização pós-treino full-integer (entrada, pesos e saída em int8).

  Equivale ao ptq.py do upstream. O representative_dataset mostra ao
  conversor a faixa real de cada tensor para calcular scale/zero_point.
  """

  def representative_dataset():
    for i in range(min(num_samples, len(x_train))):
      yield [x_train[i].reshape(1, NUM_FEATURES)]

  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
  converter.inference_input_type = tf.int8
  converter.inference_output_type = tf.int8
  converter.representative_dataset = representative_dataset
  return converter.convert()


def save_tflite_model(tflite_model, save_dir, model_name):
  os.makedirs(save_dir, exist_ok=True)
  save_path = os.path.join(save_dir, model_name)
  with open(save_path, "wb") as f:
    f.write(tflite_model)
  logging.info("Tflite model saved to %s (%d bytes)", save_path, len(tflite_model))
  return save_path


def export_c_array(tflite_model, cc_path, provenance):
  """Gera model.cc com o array `g_model` (substitui `xxd -i`).

  Mantém o contrato do hello_world: `alignas(8) const unsigned char g_model[]`
  e `const int g_model_len`, declarados em main/model.h.
  """
  lines = [
    "// ARQUIVO GERADO por training/train.py -- não edite à mão.",
    "// Para regenerar: cd training && python train.py",
    "//",
  ]
  lines += [f"// {k}: {v}" for k, v in provenance.items()]
  lines += [
    "",
    '#include "model.h"',
    "",
    "// Alinhado em 8 bytes: o TFLM lê o flatbuffer direto da flash e exige",
    "// acesso alinhado.",
    "alignas(8) const unsigned char g_model[] = {",
  ]
  for i in range(0, len(tflite_model), 12):
    chunk = tflite_model[i:i + 12]
    lines.append("    " + ", ".join(f"0x{b:02x}" for b in chunk) + ",")
  lines += ["};", f"const int g_model_len = {len(tflite_model)};", ""]

  os.makedirs(os.path.dirname(cc_path), exist_ok=True)
  with open(cc_path, "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(lines))
  logging.info("Array C salvo em %s", cc_path)


def list_tflite_ops(tflite_model):
  """Lista as operações do modelo: são as que o op resolver do firmware
  precisa registrar (MicroMutableOpResolver)."""
  interpreter = tf.lite.Interpreter(model_content=tflite_model)
  try:
    return sorted({op["op_name"] for op in interpreter._get_ops_details()})
  except AttributeError:  # API privada; pode mudar entre versões do TF
    return ["(indisponível nesta versão do TF)"]


def train_model(
  epochs,
  x_train,
  y_train,
  x_val,
  y_val,
  save_tf_model=False,
  save_dir="models",
):
  model = create_model()
  model.summary(print_fn=logging.info)
  model.fit(
    x_train,
    y_train,
    epochs=epochs,
    validation_data=(x_val, y_val),
    batch_size=32,
    verbose=2,
  )

  if save_tf_model:
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "model.keras")
    model.save(save_path)
    logging.info("TF model saved to %s", save_path)

  return model


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--epochs", type=int, default=200,
                      help="número de épocas de treino.")
  parser.add_argument("--samples_per_class", type=int, default=600,
                      help="amostras sintéticas por classe.")
  parser.add_argument("--seed", type=int, default=42,
                      help="semente do dataset, do split e do TensorFlow.")
  parser.add_argument("--save_dir", default=os.path.join(_PREFIX_PATH, "models"),
                      help="pasta dos modelos gerados.")
  parser.add_argument(
    "--data_csv",
    default=os.path.join(_PREFIX_PATH, "data", "mpu6050_orientacao.csv"),
    help="CSV do dataset gerado.",
  )
  parser.add_argument(
    "--model_cc",
    default=os.path.join(_PREFIX_PATH, "..", "main", "model.cc"),
    help="arquivo .cc do firmware que recebe o array g_model.",
  )
  parser.add_argument("--save_tf_model", action=argparse.BooleanOptionalAction,
                      default=False, help="salva também o model.keras.")
  args, _ = parser.parse_known_args()

  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

  # Reprodutibilidade: mesma semente -> mesmo dataset e (quase) mesmos pesos.
  # O treino em CPU ainda pode variar na última casa decimal.
  np.random.seed(args.seed)
  tf.keras.utils.set_random_seed(args.seed)

  x_values, y_values, theta_deg = get_data(
    samples_per_class=args.samples_per_class, seed=args.seed
  )
  split = split_data(y_values, seed=args.seed)
  save_dataset_csv(args.data_csv, x_values, y_values, theta_deg, split)

  is_train, is_val = split == "train", split == "val"
  trained_model = train_model(
    args.epochs,
    x_values[is_train], y_values[is_train],
    x_values[is_val], y_values[is_val],
    args.save_tf_model, args.save_dir,
  )

  # float32: referência para medir a perda da quantização no eval.py
  tflite_float = convert_tflite_model(trained_model)
  save_tflite_model(tflite_float, args.save_dir, "orientacao_float.tflite")

  # int8: o modelo que vai para o microcontrolador
  tflite_int8 = convert_quantized_tflite_model(trained_model, x_values[is_train])
  save_tflite_model(tflite_int8, args.save_dir, "orientacao_int8.tflite")

  is_test = split == "test"
  _, keras_test_acc = trained_model.evaluate(
    x_values[is_test], y_values[is_test], verbose=0
  )
  ops = list_tflite_ops(tflite_int8)

  summary = {
    "classes": CLASSES,
    "tilt_bands_deg": TILT_BANDS_DEG,
    "seed": args.seed,
    "epochs": args.epochs,
    "samples_total": int(len(y_values)),
    "samples_per_split": {s: int((split == s).sum()) for s in ("train", "val", "test")},
    "params": int(trained_model.count_params()),
    "float_tflite_bytes": len(tflite_float),
    "int8_tflite_bytes": len(tflite_int8),
    "keras_test_accuracy": round(float(keras_test_acc), 4),
    "int8_ops": ops,
    "tensorflow": tf.__version__,
  }
  with open(os.path.join(args.save_dir, "train_summary.json"), "w",
            encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

  export_c_array(
    tflite_int8,
    os.path.abspath(args.model_cc),
    {
      "Modelo": "MLP 3 -> 8 -> 8 -> 3 (softmax), quantizado int8",
      "Classes (índice)": ", ".join(f"{i}={c}" for i, c in enumerate(CLASSES)),
      "Parâmetros": summary["params"],
      "Seed / épocas": f"{args.seed} / {args.epochs}",
      "Acurácia Keras (teste)": summary["keras_test_accuracy"],
      "Operações": ", ".join(ops),
      "TensorFlow": tf.__version__,
    },
  )

  logging.info("Resumo: %s", json.dumps(summary, ensure_ascii=False))
  logging.info("Próximo passo: python eval.py")


if __name__ == "__main__":
  main()
