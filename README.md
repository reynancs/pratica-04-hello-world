# Prática 4/6 — Hello World com TensorFlow Lite Micro

- **UC:** IA Embarcada e Modelos Compactos — Pós-graduação em Inteligência Artificial Aplicada, UniSENAI
- **Prof.:** MSc. Rodrigo Kobashikawa Rosa
- **Aluno:** Renan Cardoso

---

## O desafio

**Atividade principal (Parte A)**

1. Reproduzir os passos do Hello World;
2. Print da tela do Wokwi rodando o Hello World;
3. Análise do código e documentação em um breve relatório sobre as observações encontradas.

**Atividade extra (Parte B, +1 ponto)**

- Modificar o código para uma nova aplicação, com novo sensor e novo dataset;
- Não pode ser outro exemplo do `esp-tflite-micro`, exceto com novo retreino/fine-tuning do modelo para novas funcionalidades.

### Requisitos e como foram atendidos

| # | Requisito | Como foi atendido | Status |
|---|---|---|---|
| 1 | Reproduzir os passos do Hello World | Projeto ESP-IDF criado do zero, componente `espressif/esp-tflite-micro` do Registry, exemplo `hello_world` copiado para `main/`, ESP-NN desligado para o Wokwi, build e simulação. Treino, quantização int8 e avaliação refeitos no Colab ([notebook](notebooks/tflite_hello_world_training.ipynb)) | ⚠️ falta embarcar o modelo retreinado |
| 2 | Print do Wokwi rodando o Hello World | Simulação com a senoide no monitor serial — ver [Evidências](#evidências) | ✅ screenshot |
| 3 | Relatório de análise | [`RELATORIO.md`](RELATORIO.md) | ✍️ em redação |
| Extra | Nova aplicação, novo sensor, novo dataset | Classificador de orientação com MPU6050 (REPOUSO / INCLINADO / INVERTIDO), dataset sintético e treino próprio — ver [`extra_mpu6050/`](extra_mpu6050/README.md) | 🔧 código e treino prontos; falta build + evidências |

### Entrega

- [ ] Link do repositório Git (`link_diretorio_github.txt`)
- [x] Screenshot — build concluído
- [x] Screenshot — Wokwi rodando o Hello World
- [ ] Screenshot — Wokwi rodando o modelo retreinado no Colab
- [ ] Screenshot — gráfico float32 × int8 do Colab
- [ ] Relatório de análise

---

## A solução

O Hello World do TensorFlow Lite Micro é uma rede neural minúscula que aprendeu a aproximar **`y = sin(x)`** para `x` em `[0, 2π]`. O objetivo não é o seno: é exercitar o pipeline completo de IA embarcada — **treinar → converter para `.tflite` int8 → embutir como array C → inferir no microcontrolador**.

| Item | Escolha |
|---|---|
| Placa | ESP32-S3-DevKitC-1 (simulada no Wokwi) |
| Framework | ESP-IDF v6.1 (a aula foi gravada na v5.5.3) |
| Linguagem | C++ (o TFLite Micro é C++; por isso os arquivos são `.cc`) |
| Biblioteca | `espressif/esp-tflite-micro` **1.4.1** (fixada em [`main/idf_component.yml`](main/idf_component.yml)) + `espressif/esp-nn` **1.4.1** (versões do [`dependencies.lock`](dependencies.lock)) |
| Modelo | MLP 1 → 16 → 16 → 1 (321 parâmetros), quantizado em int8 |
| Sensor | Nenhum — a entrada `x` é gerada pelo próprio firmware |

### Circuito

Só a placa. O [`diagram.json`](diagram.json) liga `esp:TX → $serialMonitor:RX` e `esp:RX → $serialMonitor:TX`; sem essas duas conexões a simulação roda, mas o terminal fica mudo.

---

## Como rodar

### 1. Ativar o ambiente ESP-IDF (PowerShell, instalação via EIM)

```bash
. 'C:\Espressif\tools\Microsoft.v6.1.PowerShell_profile.ps1'
```

### 2. Primeiro build (baixa as dependências)

Na raiz deste projeto:

```bash
idf.py build
```

O gerenciador de componentes baixa o `esp-tflite-micro` e o `esp-nn` para `managed_components/`.

### 3. ESP-NN desligado para o Wokwi — já vem configurado

O **ESP-NN** acelera as operações int8 com as instruções vetoriais (SIMD) do ESP32-S3, que **o Wokwi não simula**. O componente `esp-tflite-micro` compila com `-DESP_NN`; o [`main/CMakeLists.txt`](main/CMakeLists.txt) acrescenta `-UESP_NN` às flags do componente, e como o `-U` vem depois do `-D` na linha de compilação, a macro fica desligada:

```cmake
idf_component_get_property(tflite_micro_lib espressif__esp-tflite-micro COMPONENT_LIB)
target_compile_options(${tflite_micro_lib} PRIVATE -UESP_NN)
```

Nada precisa ser editado em `managed_components/` (que não é versionada), então o ajuste sobrevive a clone, Full Clean e troca de versão.

> Com o ESP-NN desligado, o TFLite Micro usa os kernels de referência em C++: o resultado da inferência é o mesmo, só mais lento. **Numa placa física, remova essas duas linhas para ligar o ESP-NN.**

### 4. Simular

No VS Code: abrir o `diagram.json` → `Ctrl+Shift+P` → **`Wokwi: Start Simulation`**.

> **Sem build, não há simulação.** O Wokwi executa o `.elf` apontado pelo [`wokwi.toml`](wokwi.toml) (`build/pratica-04-hello-world.elf`). Toda alteração no código exige novo build antes de reiniciar a simulação.

---

## Saída esperada no monitor serial

Uma inferência a cada 500 ms; `x` percorre `[0, 2π)` em 20 passos e recomeça:

```
I (222) main_task: Calling app_main()
x_value: 0.000000, y_value: 0.000000
x_value: 0.314159, y_value: 0.372770
x_value: 0.628319, y_value: 0.559154
x_value: 0.942478, y_value: 0.838731
x_value: 1.256637, y_value: 0.965812
...
```

`y` sobe até ≈ 1, desce até ≈ −1 e volta: é a senoide aproximada pelo modelo int8.

---

## Treino do modelo (Colab)

O notebook [`notebooks/tflite_hello_world_training.ipynb`](notebooks/tflite_hello_world_training.ipynb) reproduz o `train.py`, a quantização e o `evaluate.py` do exemplo oficial: gera 1.000 pontos de `sin(x)`, treina a rede por 300 épocas, converte para `.tflite` float32 e int8 (full integer) e compara os dois.

| Modelo | MAE | RMSE |
|---|---|---|
| float32 | 0,0134 | 0,0181 |
| int8 | 0,0187 | 0,0233 |

Para embarcar o modelo treinado: `xxd -i hello_world_int8.tflite`, e em [`main/model.cc`](main/model.cc) substituir **só os bytes** do array e o valor de `g_model_len`, mantendo `alignas(8) const unsigned char g_model[]` e `const int g_model_len`.

---

## Estrutura

```
pratica-04-hello-world/
├── CMakeLists.txt            # project(pratica-04-hello-world)
├── dependencies.lock         # versões exatas resolvidas — versionado
├── sdkconfig.defaults        # config versionada (target esp32s3); o sdkconfig não vai pro git
├── diagram.json              # circuito do Wokwi (só a placa + monitor serial)
├── wokwi.toml                # aponta para o .elf do build
├── main/
│   ├── main.cc               # app_main (extern "C") → setup() + loop() a cada 500 ms
│   ├── main_functions.cc/.h  # carrega o modelo, op resolver, interpretador, inferência
│   ├── model.cc/.h           # modelo .tflite como array C (g_model)
│   ├── constants.cc/.h       # kXrange = 2π, kInferencesPerCycle = 20
│   ├── output_handler.cc/.h  # imprime x_value / y_value
│   ├── CMakeLists.txt        # lista os .cc em SRCS e desliga o ESP-NN (-UESP_NN)
│   └── idf_component.yml     # dependência espressif/esp-tflite-micro
├── notebooks/
│   └── tflite_hello_world_training.ipynb
└── artefatos/                # screenshots que comprovam a entrega
    ├── 01_screenshot_build_successful.png
    └── 02_screenshot_wokwi_hello_world.png
```

---

## Evidências

### 1. Build concluído e ESP-NN desligado

![VS Code com o CMakeLists.txt do esp-tflite-micro aberto na linha do ESP-NN comentada e o relatório do idf.py size no terminal](artefatos/01_screenshot_build_successful.png)

- **ESP-IDF configurado** — barra de status com `ESP-IDF v6.1.0` e target `esp32s3`
- **ESP-NN desligado** — linha 134 de `managed_components/espressif__esp-tflite-micro/CMakeLists.txt` comentada
- **Build sem erros** — relatório do `idf.py size`: `Total image size: 206988 bytes`, DIRAM com 14,92% em uso

### 2. Wokwi rodando o Hello World

![Wokwi Simulator com o ESP32-S3 e o monitor serial imprimindo os pares x_value / y_value da senoide](artefatos/02_screenshot_wokwi_hello_world.png)

- **Conta Wokwi ativa** — licença `Renan Cardoso — Community License`
- **Inferência funcionando** — 20 linhas `x_value`/`y_value` formando um ciclo completo da senoide
