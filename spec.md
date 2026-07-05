# Speech-JSCM System Specification

> 用途：本 Markdown 作为 Codex 的工程构建说明书，用于逐步实现一个简单、可调试、可迭代的语音信源信道联合调制系统。系统核心为 **Conformer-JSCM**，基线为 **Opus + GFSK**，数据集为 **LibriSpeech train-clean-100**，并预留 **USRP + GNU Radio Companion `.grc`** 实物链路。

---

## 0. 项目目标

构建一个用于比较 **Conformer-JSCM** 与 **Opus baseline** 的语音通信实验系统。

系统应支持三种实验模式：

1. **纯 Python 仿真模式**
   - Conformer-JSCM + AWGN/Rayleigh 信道；
   - Opus 编解码 baseline；
   - 计算 PESQ、STOI、SI-SDR 等指标。

2. **GNU Radio loopback 模式**
   - 不连接 USRP；
   - 用 GRC 软件信道验证 GFSK/FM 调制解调链路；
   - 通过文件输入输出检查中间数据是否正确。

3. **USRP 实物通信模式**
   - Opus baseline：Opus bitstream → GFSK → USRP → GFSK demod → Opus decode；
   - JSCM：JSCM complex symbols → real serialization → FM → USRP → FM demod → complex symbols → JSCM decode。

核心原则：

```text
Python 负责：数据集、模型、训练、Opus、指标、实验管理。
GRC 负责：GFSK/FM 调制解调、USRP 收发、文件/ZMQ 数据接口。
文件接口优先，ZMQ 实时接口后置。
```

---

## 1. 系统总体结构

```text
speech-jscm/
├── README.md
├── environment.yml
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── data/
│   │   └── librispeech_clean100.yaml
│   ├── model/
│   │   └── conformer_jscm_base.yaml
│   ├── train/
│   │   └── train_conformer_jscm.yaml
│   ├── eval/
│   │   ├── eval_jscm_awgn.yaml
│   │   ├── eval_jscm_usrp.yaml
│   │   └── eval_opus_usrp.yaml
│   └── radio/
│       ├── usrp_default.yaml
│       ├── opus_gfsk.yaml
│       └── jscm_fm.yaml
│
├── data/
│   ├── raw/
│   │   └── LibriSpeech/              # 不提交 git
│   ├── manifests/
│   │   ├── train_clean_100.jsonl
│   │   ├── dev_clean.jsonl
│   │   └── test_clean.jsonl
│   └── cache/
│       └── stft/                     # 可选缓存，不提交 git
│
├── src/
│   └── speech_jscm/
│       ├── __init__.py
│       │
│       ├── data/
│       │   ├── librispeech.py
│       │   ├── dataset.py
│       │   ├── segment.py
│       │   └── manifest.py
│       │
│       ├── features/
│       │   ├── stft.py
│       │   └── waveform.py
│       │
│       ├── models/
│       │   ├── conformer_jscm.py
│       │   ├── conformer_blocks.py
│       │   ├── channel_mapper.py
│       │   └── power_norm.py
│       │
│       ├── channels/
│       │   ├── awgn.py
│       │   ├── rayleigh.py
│       │   ├── fm_sim.py
│       │   └── gfsk_sim.py
│       │
│       ├── baselines/
│       │   ├── opus_codec.py
│       │   └── opus_link.py
│       │
│       ├── radio/
│       │   ├── packet.py
│       │   ├── iq_io.py
│       │   ├── zmq_io.py
│       │   ├── usrp_config.py
│       │   └── sync.py
│       │
│       ├── metrics/
│       │   ├── pesq_stoi.py
│       │   ├── sisdr.py
│       │   ├── mcd.py
│       │   └── report.py
│       │
│       ├── losses/
│       │   ├── spectral_loss.py
│       │   └── sisdr_loss.py
│       │
│       └── utils/
│           ├── seed.py
│           ├── logger.py
│           ├── checkpoint.py
│           └── audio_io.py
│
├── scripts/
│   ├── 00_prepare_librispeech.py
│   ├── 01_make_manifest.py
│   ├── 02_train_conformer_jscm.py
│   ├── 03_eval_jscm_sim.py
│   ├── 04_eval_opus_sim.py
│   ├── 05_export_jscm_symbols.py
│   ├── 06_recover_jscm_symbols.py
│   ├── 07_eval_jscm_usrp.py
│   ├── 08_eval_opus_usrp.py
│   └── 09_make_report.py
│
├── grc/
│   ├── README.md
│   ├── common/
│   │   ├── usrp_tx_common.grc
│   │   ├── usrp_rx_common.grc
│   │   └── loopback_channel.grc
│   │
│   ├── opus_gfsk/
│   │   ├── opus_gfsk_tx.grc
│   │   ├── opus_gfsk_rx.grc
│   │   ├── opus_gfsk_loopback.grc
│   │   └── params_opus_gfsk.md
│   │
│   ├── jscm_fm/
│   │   ├── jscm_fm_tx.grc
│   │   ├── jscm_fm_rx.grc
│   │   ├── jscm_fm_loopback.grc
│   │   └── params_jscm_fm.md
│   │
│   └── generated/
│       ├── opus_gfsk_tx.py
│       ├── opus_gfsk_rx.py
│       ├── jscm_fm_tx.py
│       └── jscm_fm_rx.py
│
├── experiments/
│   ├── exp001_conformer_awgn/
│   ├── exp002_conformer_rayleigh/
│   ├── exp003_opus_gfsk_usrp/
│   └── exp004_jscm_fm_usrp/
│
├── checkpoints/
│   └── conformer_jscm_base/
│
├── outputs/
│   ├── wav/
│   ├── symbols/
│   ├── iq/
│   ├── logs/
│   ├── metrics/
│   └── figures/
│
└── tests/
    ├── test_stft_reconstruction.py
    ├── test_power_norm.py
    ├── test_awgn_channel.py
    ├── test_opus_codec.py
    └── test_symbol_io.py
```

---

## 2. 数据集与输入规格

### 2.1 数据集

使用 LibriSpeech：

```text
Training:    LibriSpeech train-clean-100
Validation:  LibriSpeech dev-clean
Test:        LibriSpeech test-clean
Optional:    LibriSpeech test-other, 用于鲁棒性测试
```

第一版不从 `train-clean-100` 内部再切验证集，直接使用官方 `dev-clean` 作为验证集。

### 2.2 每条训练样本大小

采样率：

```text
sample_rate = 16000 Hz
```

训练段长度：

```text
segment_samples = 16384
segment_duration = 1.024 s
```

原始 waveform 输入：

```text
waveform.shape = [1, 16384]
```

训练时处理规则：

```text
如果 utterance 长度 >= 16384 samples：随机裁剪 16384 samples。
如果 utterance 长度 < 16384 samples：zero-pad 到 16384 samples。
```

验证/测试时有两种模式：

```text
早期调试：center crop 16384 samples。
正式测试：完整 utterance 分块，最后一个 chunk padding，重构后裁回原长度。
```

### 2.3 STFT 特征

第一版使用 STFT real/imag，不使用 Mel。

原因：Mel 会丢失相位，后续需要 vocoder 或 Griffin-Lim，会混入声码器误差，不适合作为第一版通信重构评估。

STFT 参数：

```text
n_fft      = 512
win_length = 400      # 25 ms at 16 kHz
hop_length = 160      # 10 ms at 16 kHz
center     = false
window     = hann
feature    = real_imag
```

频率 bin 数：

```text
F = n_fft / 2 + 1 = 257
```

时间帧数：

```text
T = floor((16384 - 512) / 160) + 1 = 100
```

模型输入特征：

```text
x.shape = [2, 257, 100]
```

batch 输入：

```text
x_batch.shape = [B, 2, 257, 100]
```

模型输出：

```text
y_hat.shape = [B, 2, 257, 100]
```

波形重构：

```text
reconstructed waveform = iSTFT(y_hat)
```

---

## 3. Conformer-JSCM 通信原理

### 3.1 基本链路

```text
waveform
→ STFT real/imag
→ Conformer-JSCM encoder
→ channel mapper
→ power normalization
→ complex channel symbols
→ wireless channel / FM radio transport
→ received symbols
→ Conformer-JSCM decoder
→ reconstructed STFT real/imag
→ iSTFT
→ reconstructed waveform
```

### 3.2 数学形式

原始语音：

```text
s ∈ R^N
```

STFT 特征：

```text
X = STFT(s) ∈ R^{2 × F × T}
```

编码器：

```text
Z = f_θ(X, γ)
```

其中：

```text
γ 可以是 SNR embedding 或 CSI embedding。
```

信道映射：

```text
U = g_θ(Z) ∈ R^{2M}
```

将实数向量两两组成复数符号：

```text
x_m = U_{2m} + j U_{2m+1},  m = 1,...,M
```

平均功率归一化：

```text
x_norm = x / sqrt(mean(|x|^2) + eps)
```

约束：

```text
E[|x_norm|^2] ≈ 1
```

AWGN 信道：

```text
y = x_norm + n
n ~ CN(0, σ²)
```

Rayleigh 信道：

```text
y = h x_norm + n
h ~ CN(0, 1)
```

如果接收端知道 h，可先做简单均衡：

```text
y_eq = y / (h + eps)
```

解码器：

```text
X_hat = d_φ(y, γ)
```

重构波形：

```text
s_hat = iSTFT(X_hat)
```

### 3.3 码率/信道符号设置

每条训练样本有 100 个 10 ms STFT 帧。

定义三档 JSCM 传输资源：

| 档位 | 每 10 ms 复数符号数 | 每个 1.024 s segment 复数符号数 | 实数信道维度 | 相对 STFT 实数维度 |
|---|---:|---:|---:|---:|
| Low | 8 | 800 | 1600 | 约 1/32 |
| Mid | 16 | 1600 | 3200 | 约 1/16 |
| High | 32 | 3200 | 6400 | 约 1/8 |

STFT 实数维度：

```text
2 × 257 × 100 = 51400
```

默认第一版使用：

```text
num_complex_symbols = 1600
```

即 Mid 档。

---

## 4. Conformer-JSCM 模型规格

### 4.1 Base 模型

默认使用 Base 级别模型，不做大模型。

```yaml
model: conformer_jscm

input_channels: 2
freq_bins: 257
time_frames: 100

d_model: 192
num_heads: 4
encoder_layers: 6
decoder_layers: 8
ffn_dim: 768
conv_kernel_size: 31
dropout: 0.1

channel_symbols:
  mode: per_segment
  num_complex_symbols: 1600

power_norm:
  average_power: 1.0
```

### 4.2 推荐结构

```text
Input X: [B, 2, 257, 100]

Encoder:
  Conv2D tokenizer / patch embedding
  6 × Conformer blocks
  flatten / pooling
  channel mapper
  power normalization

Channel symbols:
  z: [B, M, 2], M = 1600

Channel:
  AWGN / Rayleigh / identity / radio interface

Decoder:
  symbol embedding
  8 × Conformer blocks
  output projection

Output X_hat:
  [B, 2, 257, 100]
```

### 4.3 Conformer block

每个 Conformer block 包括：

```text
Feed-forward module
Multi-head self-attention module
Convolution module
Feed-forward module
LayerNorm / residual connection
```

第一版可以实现简化版：

```text
LayerNorm
Multi-head self-attention
Depthwise Conv1D
FFN
Residual
```

不要一开始追求完整工业级 Conformer，先保证数据流和重构可用。

### 4.4 损失函数

第一版训练损失：

```text
L = λ1 * L_stft + λ2 * L_wave
```

推荐：

```text
L_stft = L1(real_imag_STFT_hat, real_imag_STFT)
L_wave = L1(iSTFT(STFT_hat), waveform)
```

后续可加入：

```text
SI-SDR loss
multi-resolution STFT loss
magnitude loss
phase-aware loss
```

第一版默认：

```yaml
loss:
  stft_l1_weight: 1.0
  waveform_l1_weight: 0.1
  sisdr_weight: 0.0
```

---

## 5. Opus + GFSK baseline 通信原理

### 5.1 基本链路

```text
waveform
→ Opus encode
→ byte stream
→ packet framing
→ bitstream
→ GFSK modulation
→ USRP TX
→ wireless channel
→ USRP RX
→ GFSK demodulation
→ packet deframing
→ Opus decode
→ reconstructed waveform
```

### 5.2 Opus 设置

第一版建议测试以下 bitrate：

```text
6 kbps
9 kbps
12 kbps
16 kbps
```

默认配置：

```yaml
opus:
  sample_rate: 16000
  channels: 1
  application: voip
  bitrate: 12000
  frame_ms: 20
```

Opus baseline 的输入输出均为 waveform，不经过 STFT。

### 5.3 GFSK 设置

默认配置：

```yaml
gfsk:
  samples_per_symbol: 4
  sensitivity: 1.0
  bt: 0.35

packet:
  preamble: true
  crc: true
  payload_bytes: 64
```

GFSK 适合离散 bitstream，因此用于 Opus baseline。

---

## 6. JSCM + FM 实物链路原理

### 6.1 为什么 JSCM 使用 FM 时需要串行化

Conformer-JSCM 输出的是复数连续符号：

```text
z ∈ C^M
```

GNU Radio 中常规 FM 调制器的输入通常是实值 message signal。因此，如果强制使用 FM 传输 JSCM 符号，第一版采用如下简单映射：

```text
z = a + jb
real_sequence = [a_1, b_1, a_2, b_2, ..., a_M, b_M]
```

FM 解调后得到：

```text
rx_real_sequence = [a'_1, b'_1, a'_2, b'_2, ..., a'_M, b'_M]
```

再还原为复数符号：

```text
z'_m = a'_m + j b'_m
```

该方案优点：

```text
实现简单
便于调试
文件接口清晰
可以直接检查符号 MSE
```

缺点：

```text
不是最优调制方式
FM 会引入幅度/频率响应失真
需要做归一化、pilot、同步和 rescale
```

第一版目标不是最优无线物理层，而是建立可控的 JSCM 与 Opus 对比链路。

### 6.2 JSCM + FM 链路

```text
waveform
→ STFT real/imag
→ Conformer-JSCM encoder
→ z ∈ C^M
→ serialize complex symbols to real sequence
→ add preamble / pilot
→ FM modulation
→ USRP TX
→ USRP RX
→ FM demodulation
→ timing/symbol alignment
→ remove preamble / pilot
→ de-serialize to complex symbols
→ Conformer-JSCM decoder
→ iSTFT
→ reconstructed waveform
```

### 6.3 FM 配置

默认配置：

```yaml
link: jscm_fm

symbols:
  dtype: complex64
  num_complex_symbols_per_segment: 1600
  normalize: true

fm:
  audio_rate: 48000
  quadrature_rate: 192000
  max_deviation_hz: 5000

packet:
  preamble: true
  pilot: true
  payload_symbols: 1600
```

---

## 7. Python 与 GRC 的数据接口

### 7.1 文件接口优先

第一版必须使用文件接口，不直接做 ZMQ 实时闭环。

#### JSCM 文件接口

Python export：

```text
outputs/symbols/jscm_tx_symbols.c64
outputs/symbols/jscm_tx_meta.json
```

GRC input：

```text
jscm_tx_symbols.c64
```

GRC output：

```text
outputs/symbols/jscm_rx_symbols.c64
```

Python recover：

```text
jscm_rx_symbols.c64
→ Conformer-JSCM decoder
→ reconstructed wav
→ metrics
```

#### Opus 文件接口

Python export：

```text
outputs/symbols/opus_tx_bits.bin
outputs/symbols/opus_tx_meta.json
```

GRC input：

```text
opus_tx_bits.bin
```

GRC output：

```text
outputs/symbols/opus_rx_bits.bin
```

Python recover：

```text
opus_rx_bits.bin
→ packet deframing
→ Opus decode
→ reconstructed wav
→ metrics
```

### 7.2 数据格式

#### `.c64`

JSCM complex symbols：

```text
dtype: complex64
layout: interleaved float32 real/imag
shape: [num_segments, num_complex_symbols]
```

内部保存建议：

```python
np.asarray(symbols, dtype=np.complex64).tofile(path)
```

读取：

```python
symbols = np.fromfile(path, dtype=np.complex64)
```

#### `.bin`

Opus bits：

```text
dtype: uint8
content: packed bytes or unpacked bits，需要在 meta.json 中标明
```

第一版建议使用 packed bytes，GRC 侧需要明确 unpack/pack。

#### `meta.json`

JSCM meta 示例：

```json
{
  "utt_id": "19-198-0000",
  "sample_rate": 16000,
  "segment_samples": 16384,
  "num_complex_symbols": 1600,
  "symbol_dtype": "complex64",
  "modulation": "fm",
  "serialization": "re_im_interleave",
  "stft": {
    "n_fft": 512,
    "win_length": 400,
    "hop_length": 160,
    "center": false
  }
}
```

Opus meta 示例：

```json
{
  "utt_id": "19-198-0000",
  "sample_rate": 16000,
  "codec": "opus",
  "bitrate": 12000,
  "frame_ms": 20,
  "modulation": "gfsk",
  "bit_packing": "packed_bytes"
}
```

---

## 8. GRC 文件设计

### 8.1 GRC 目录

```text
grc/
├── opus_gfsk/
│   ├── opus_gfsk_loopback.grc
│   ├── opus_gfsk_tx.grc
│   ├── opus_gfsk_rx.grc
│   └── params_opus_gfsk.md
│
└── jscm_fm/
    ├── jscm_fm_loopback.grc
    ├── jscm_fm_tx.grc
    ├── jscm_fm_rx.grc
    └── params_jscm_fm.md
```

每条链路至少包含：

```text
*_loopback.grc：软件闭环调试，不接 USRP。
*_tx.grc：USRP 发送。
*_rx.grc：USRP 接收。
```

### 8.2 Opus GFSK loopback

目标：验证 Opus bitstream 经过 GFSK 调制解调后是否可恢复。

```text
File Source: opus_tx_bits.bin
→ unpack bits if needed
→ packet framing / preamble
→ GFSK Mod
→ Channel Model
→ GFSK Demod
→ packet deframing
→ pack bits if needed
→ File Sink: opus_rx_bits.bin
```

### 8.3 Opus GFSK TX

```text
File Source / ZMQ Source: opus_tx_bits.bin
→ unpack bits if needed
→ packet framing
→ GFSK Mod
→ Multiply Const
→ UHD: USRP Sink
```

### 8.4 Opus GFSK RX

```text
UHD: USRP Source
→ AGC
→ frequency correction
→ GFSK Demod
→ packet deframing
→ pack bits if needed
→ File Sink / ZMQ Sink: opus_rx_bits.bin
```

### 8.5 JSCM FM loopback

目标：验证 JSCM continuous symbols 经过 FM 调制解调后能否被恢复。

```text
File Source: jscm_tx_real_sequence.f32
→ optional preamble/pilot insertion
→ FM Mod
→ Channel Model
→ FM Demod
→ low-pass / resampler
→ pilot-based scale correction
→ File Sink: jscm_rx_real_sequence.f32
```

注意：GRC 侧可以直接处理 `.f32` 实数序列。Python 侧负责：

```text
complex64 symbols → real interleaved float32 sequence
```

和：

```text
received float32 sequence → complex64 symbols
```

### 8.6 JSCM FM TX

```text
File Source / ZMQ Source: jscm_tx_real_sequence.f32
→ preamble / pilot insertion
→ FM Mod
→ Multiply Const
→ UHD: USRP Sink
```

### 8.7 JSCM FM RX

```text
UHD: USRP Source
→ AGC
→ frequency correction
→ FM Demod
→ low-pass / decimation
→ timing alignment
→ pilot-based amplitude correction
→ File Sink / ZMQ Sink: jscm_rx_real_sequence.f32
```

---

## 9. USRP 配置

默认配置文件：`configs/radio/usrp_default.yaml`

```yaml
usrp:
  device_args: ""
  center_freq: 915e6
  samp_rate: 1e6
  tx_gain: 20
  rx_gain: 20
  antenna: TX/RX
  clock_source: internal

safety:
  tx_amplitude: 0.2
```

注意：

```text
实际 center_freq 必须根据实验室设备、天线和本地频谱规定设置。
不要默认在未授权频段长时间发射。
调试时先接衰减器/线缆闭环，确认功率安全后再空口发射。
```

---

## 10. 配置文件

### 10.1 `configs/data/librispeech_clean100.yaml`

```yaml
dataset: librispeech
sample_rate: 16000

train_manifest: data/manifests/train_clean_100.jsonl
valid_manifest: data/manifests/dev_clean.jsonl
test_manifest: data/manifests/test_clean.jsonl

segment_samples: 16384
random_crop: true

stft:
  n_fft: 512
  win_length: 400
  hop_length: 160
  center: false
  feature: real_imag
```

### 10.2 `configs/model/conformer_jscm_base.yaml`

```yaml
model: conformer_jscm

input_channels: 2
freq_bins: 257
time_frames: 100

d_model: 192
num_heads: 4
encoder_layers: 6
decoder_layers: 8
ffn_dim: 768
conv_kernel_size: 31
dropout: 0.1

channel_symbols:
  mode: per_segment
  num_complex_symbols: 1600

power_norm:
  average_power: 1.0
```

### 10.3 `configs/train/train_conformer_jscm.yaml`

```yaml
seed: 42

data_config: configs/data/librispeech_clean100.yaml
model_config: configs/model/conformer_jscm_base.yaml

train:
  batch_size: 16
  num_workers: 4
  epochs: 100
  lr: 0.0003
  weight_decay: 0.00001
  grad_clip: 5.0
  amp: true

channel:
  type: awgn
  snr_db_min: 0
  snr_db_max: 20

loss:
  stft_l1_weight: 1.0
  waveform_l1_weight: 0.1
  sisdr_weight: 0.0

checkpoint:
  dir: checkpoints/conformer_jscm_base
  save_best: true
  monitor: valid_loss
```

### 10.4 `configs/eval/eval_jscm_awgn.yaml`

```yaml
checkpoint: checkpoints/conformer_jscm_base/best.pt
data_config: configs/data/librispeech_clean100.yaml
model_config: configs/model/conformer_jscm_base.yaml

split: test
full_utterance: true
chunk_samples: 16384
chunk_hop_samples: 16384

channel:
  type: awgn
  snr_db_list: [-5, 0, 5, 10, 15, 20]

metrics:
  pesq: true
  stoi: true
  sisdr: true
  mcd: true
```

### 10.5 `configs/radio/opus_gfsk.yaml`

```yaml
baseline: opus_gfsk

opus:
  sample_rate: 16000
  channels: 1
  application: voip
  bitrate: 12000
  frame_ms: 20

gfsk:
  samples_per_symbol: 4
  sensitivity: 1.0
  bt: 0.35

packet:
  preamble: true
  crc: true
  payload_bytes: 64
```

### 10.6 `configs/radio/jscm_fm.yaml`

```yaml
link: jscm_fm

symbols:
  dtype: complex64
  num_complex_symbols_per_segment: 1600
  normalize: true
  serialization: re_im_interleave

fm:
  audio_rate: 48000
  quadrature_rate: 192000
  max_deviation_hz: 5000

packet:
  preamble: true
  pilot: true
  payload_symbols: 1600
```

---

## 11. 主要 Python 模块职责

### 11.1 `data/`

#### `manifest.py`

功能：

```text
扫描 LibriSpeech 目录。
生成 jsonl manifest。
每行包含 utt_id、speaker_id、wav_path、duration。
```

manifest 示例：

```json
{"utt_id":"19-198-0000","speaker_id":"19","wav":"data/raw/LibriSpeech/train-clean-100/19/198/19-198-0000.flac","duration":14.12}
```

#### `dataset.py`

功能：

```text
读取 manifest。
加载 waveform。
随机裁剪或 padding 到 16384 samples。
输出 waveform 和 STFT 特征。
```

返回：

```python
{
    "utt_id": str,
    "waveform": Tensor[1, 16384],
    "stft": Tensor[2, 257, 100],
}
```

#### `segment.py`

功能：

```text
random_crop_or_pad(waveform, length)
center_crop_or_pad(waveform, length)
split_full_utterance(waveform, chunk_samples, hop_samples)
overlap_add(chunks, hop_samples)
```

### 11.2 `features/`

#### `stft.py`

功能：

```text
waveform_to_stft_ri(waveform) -> [2,257,100]
stft_ri_to_waveform(stft_ri, length) -> waveform
```

必须保证：

```text
STFT + iSTFT 在无模型情况下重构误差很小。
```

对应单元测试：

```text
tests/test_stft_reconstruction.py
```

### 11.3 `models/`

#### `conformer_jscm.py`

主模型，建议接口：

```python
class ConformerJSCM(nn.Module):
    def encode(self, x, snr_db=None):
        ...

    def decode(self, y, snr_db=None):
        ...

    def forward(self, x, channel=None, snr_db=None):
        z = self.encode(x, snr_db)
        if channel is not None:
            z = channel(z, snr_db)
        x_hat = self.decode(z, snr_db)
        return x_hat, z
```

#### `power_norm.py`

功能：

```text
将 complex symbols 归一化到平均功率 1。
```

输入：

```text
z.shape = [B, M, 2]
```

输出：

```text
z_norm.shape = [B, M, 2]
mean power ≈ 1
```

### 11.4 `channels/`

#### `awgn.py`

功能：

```text
对 complex symbols 加 AWGN。
```

输入：

```text
z.shape = [B, M, 2]
snr_db: float or Tensor[B]
```

输出：

```text
y.shape = [B, M, 2]
```

#### `rayleigh.py`

功能：

```text
模拟 flat Rayleigh fading。
可选 equalization。
```

### 11.5 `baselines/`

#### `opus_codec.py`

功能：

```text
encode_opus(wav_path or waveform, bitrate, frame_ms) -> bytes
.decode_opus(bytes) -> waveform
```

实现建议：

```text
优先支持系统 opusenc/opusdec CLI。
如果未安装，给出明确错误提示。
后续可加入 opuslib fallback。
```

#### `opus_link.py`

功能：

```text
waveform → Opus bytes → packet/bits → optional channel → reconstructed waveform
```

### 11.6 `radio/`

#### `iq_io.py`

功能：

```text
write_complex64(path, symbols)
read_complex64(path)
complex_to_interleaved_f32(symbols)
interleaved_f32_to_complex(seq)
write_f32(path, seq)
read_f32(path)
```

#### `packet.py`

功能：

```text
add_preamble(payload)
find_preamble(stream)
add_crc(payload)
check_crc(packet)
pack_bits(bits)
unpack_bits(bytes)
```

第一版可以实现最简单 preamble，不强求复杂同步算法。

#### `sync.py`

功能：

```text
pilot-based scale correction
sequence alignment by cross-correlation
trim payload after preamble
```

### 11.7 `metrics/`

指标：

```text
PESQ
STOI
SI-SDR
MCD
log-STFT distance
```

第一版必须实现：

```text
SI-SDR
PESQ
STOI
```

MCD 可后置。

---

## 12. 训练与评估流程

### 12.1 创建 manifest

```bash
python scripts/01_make_manifest.py \
  --librispeech-root data/raw/LibriSpeech \
  --out-dir data/manifests
```

输出：

```text
data/manifests/train_clean_100.jsonl
data/manifests/dev_clean.jsonl
data/manifests/test_clean.jsonl
```

验收标准：

```text
jsonl 文件存在。
每条记录包含 utt_id、speaker_id、wav、duration。
脚本能打印每个 split 的 utterance 数量和总时长。
```

### 12.2 训练 Conformer-JSCM

```bash
python scripts/02_train_conformer_jscm.py \
  --config configs/train/train_conformer_jscm.yaml
```

输出：

```text
checkpoints/conformer_jscm_base/best.pt
outputs/logs/train_conformer_jscm.log
```

验收标准：

```text
模型能前向传播。
loss 能下降。
validation loss 能正常记录。
checkpoint 能保存和加载。
```

### 12.3 JSCM 仿真评估

```bash
python scripts/03_eval_jscm_sim.py \
  --config configs/eval/eval_jscm_awgn.yaml
```

输出：

```text
outputs/metrics/jscm_awgn_summary.json
outputs/metrics/jscm_awgn_per_utterance.csv
outputs/wav/jscm_awgn/*.wav
```

验收标准：

```text
每个 SNR 都能输出 PESQ/STOI/SI-SDR。
高 SNR 指标应优于低 SNR。
identity channel 下应得到最佳结果。
```

### 12.4 Opus 仿真评估

```bash
python scripts/04_eval_opus_sim.py \
  --data-config configs/data/librispeech_clean100.yaml \
  --radio-config configs/radio/opus_gfsk.yaml \
  --bitrate 12000
```

输出：

```text
outputs/metrics/opus_12kbps_summary.json
outputs/wav/opus_12kbps/*.wav
```

验收标准：

```text
Opus encode/decode 后可以恢复语音。
PESQ/STOI/SI-SDR 可以正常计算。
```

---

## 13. USRP 文件接口流程

### 13.1 JSCM symbols 导出

```bash
python scripts/05_export_jscm_symbols.py \
  --checkpoint checkpoints/conformer_jscm_base/best.pt \
  --data-config configs/data/librispeech_clean100.yaml \
  --model-config configs/model/conformer_jscm_base.yaml \
  --split test \
  --num-utts 10 \
  --out-dir outputs/exp004_jscm_fm_usrp/tx
```

输出：

```text
outputs/exp004_jscm_fm_usrp/tx/utt_0001_symbols.c64
outputs/exp004_jscm_fm_usrp/tx/utt_0001_real_sequence.f32
outputs/exp004_jscm_fm_usrp/tx/utt_0001_meta.json
```

其中：

```text
.c64 用于 Python 内部检查。
.f32 用于 GRC FM 输入。
```

### 13.2 GRC 发送/接收

使用：

```text
grc/jscm_fm/jscm_fm_tx.grc
grc/jscm_fm/jscm_fm_rx.grc
```

输入：

```text
utt_0001_real_sequence.f32
```

输出：

```text
utt_0001_rx_real_sequence.f32
```

### 13.3 JSCM symbols 恢复并解码

```bash
python scripts/06_recover_jscm_symbols.py \
  --checkpoint checkpoints/conformer_jscm_base/best.pt \
  --model-config configs/model/conformer_jscm_base.yaml \
  --rx-real-seq outputs/exp004_jscm_fm_usrp/rx/utt_0001_rx_real_sequence.f32 \
  --meta outputs/exp004_jscm_fm_usrp/tx/utt_0001_meta.json \
  --out-wav outputs/exp004_jscm_fm_usrp/rx/utt_0001_rec.wav
```

验收标准：

```text
能从 rx_real_sequence 还原 complex symbols。
能通过 decoder 输出 waveform。
能计算和原始 waveform 的指标。
```

### 13.4 Opus bits 导出

```bash
python scripts/08_eval_opus_usrp.py \
  --mode export \
  --data-config configs/data/librispeech_clean100.yaml \
  --radio-config configs/radio/opus_gfsk.yaml \
  --split test \
  --num-utts 10 \
  --out-dir outputs/exp003_opus_gfsk_usrp/tx
```

输出：

```text
utt_0001_opus_tx_bits.bin
utt_0001_opus_meta.json
```

GRC 输入：

```text
utt_0001_opus_tx_bits.bin
```

GRC 输出：

```text
utt_0001_opus_rx_bits.bin
```

Opus 恢复：

```bash
python scripts/08_eval_opus_usrp.py \
  --mode recover \
  --rx-bits outputs/exp003_opus_gfsk_usrp/rx/utt_0001_opus_rx_bits.bin \
  --meta outputs/exp003_opus_gfsk_usrp/tx/utt_0001_opus_meta.json \
  --out-wav outputs/exp003_opus_gfsk_usrp/rx/utt_0001_rec.wav
```

---

## 14. 输出目录规范

每个实验单独建目录，禁止覆盖历史结果。

示例：

```text
outputs/exp004_jscm_fm_usrp/
├── config.yaml
├── tx/
│   ├── utt_0001.wav
│   ├── utt_0001_symbols.c64
│   ├── utt_0001_real_sequence.f32
│   └── utt_0001_meta.json
├── rx/
│   ├── utt_0001_rx_real_sequence.f32
│   ├── utt_0001_rx_symbols.c64
│   └── utt_0001_rec.wav
├── metrics/
│   ├── per_utterance.csv
│   └── summary.json
└── logs/
    ├── grc_tx.log
    ├── grc_rx.log
    └── eval.log
```

---

## 15. 指标计算

必须报告：

```text
PESQ
STOI
SI-SDR
```

建议报告：

```text
MCD
log-STFT distance
symbol MSE
latency
channel uses per second
```

### 15.1 SI-SDR

```text
用于衡量波形级重构质量。
```

### 15.2 PESQ

```text
用于衡量感知语音质量。
16 kHz 应使用 wideband 模式。
```

### 15.3 STOI

```text
用于衡量语音可懂度。
```

### 15.4 Symbol MSE

对 USRP/JSCM 特别重要：

```text
symbol_mse = mean(|z_tx - z_rx|^2)
```

这个指标用于定位无线链路是否破坏了 JSCM 符号。

---

## 16. 单元测试

### 16.1 `test_stft_reconstruction.py`

测试：

```text
随机 waveform → STFT real/imag → iSTFT → waveform_hat
检查 waveform_hat 与 waveform 的误差。
```

### 16.2 `test_power_norm.py`

测试：

```text
随机 complex symbols → power_norm
检查平均功率约等于 1。
```

### 16.3 `test_awgn_channel.py`

测试：

```text
输入单位功率信号。
添加指定 SNR 噪声。
估算实际 SNR 是否接近目标 SNR。
```

### 16.4 `test_opus_codec.py`

测试：

```text
短语音 → Opus encode → Opus decode
检查输出 waveform 可读取且长度合理。
```

### 16.5 `test_symbol_io.py`

测试：

```text
complex64 symbols 写入 .c64 再读取，数值一致。
complex symbols → interleaved f32 → complex symbols，数值一致。
```

---

## 17. 构建顺序：给 Codex 的分阶段任务

### Phase 0：初始化工程

目标：创建目录、配置文件、基础 package。

任务：

```text
1. 创建完整文件树。
2. 创建 pyproject.toml 或 setup.cfg，使 src/speech_jscm 可被 import。
3. 创建 requirements.txt 和 environment.yml。
4. 创建 README.md。
5. 创建所有 YAML 配置文件。
```

验收：

```bash
python -c "import speech_jscm; print('ok')"
```

---

### Phase 1：数据集与 STFT

目标：完成 LibriSpeech manifest、dataset、STFT/iSTFT。

任务：

```text
1. 实现 scripts/01_make_manifest.py。
2. 实现 data/manifest.py。
3. 实现 data/dataset.py。
4. 实现 data/segment.py。
5. 实现 features/stft.py。
6. 添加 test_stft_reconstruction.py。
```

验收：

```bash
python scripts/01_make_manifest.py --librispeech-root data/raw/LibriSpeech --out-dir data/manifests
pytest tests/test_stft_reconstruction.py
```

---

### Phase 2：JSCM 模型与 AWGN 信道

目标：完成 Conformer-JSCM 最小可训练版本。

任务：

```text
1. 实现 models/power_norm.py。
2. 实现 models/conformer_blocks.py。
3. 实现 models/channel_mapper.py。
4. 实现 models/conformer_jscm.py。
5. 实现 channels/awgn.py。
6. 实现 losses/spectral_loss.py。
7. 添加 test_power_norm.py 和 test_awgn_channel.py。
```

验收：

```bash
pytest tests/test_power_norm.py
pytest tests/test_awgn_channel.py
python - <<'PY'
import torch
from speech_jscm.models.conformer_jscm import ConformerJSCM
m = ConformerJSCM()
x = torch.randn(2, 2, 257, 100)
y, z = m(x)
print(y.shape, z.shape)
PY
```

期望：

```text
y.shape = [2, 2, 257, 100]
z.shape = [2, 1600, 2]
```

---

### Phase 3：训练脚本

目标：完成端到端训练。

任务：

```text
1. 实现 scripts/02_train_conformer_jscm.py。
2. 实现 utils/checkpoint.py。
3. 实现 utils/logger.py。
4. 支持 AMP、checkpoint、validation。
```

验收：

```bash
python scripts/02_train_conformer_jscm.py --config configs/train/train_conformer_jscm.yaml
```

期望：

```text
loss 能下降。
checkpoint 能保存。
validation 能运行。
```

---

### Phase 4：JSCM 仿真评估

目标：完成 test-clean 上的 AWGN 评估。

任务：

```text
1. 实现 scripts/03_eval_jscm_sim.py。
2. 实现 metrics/sisdr.py。
3. 实现 metrics/pesq_stoi.py。
4. 实现 metrics/report.py。
5. 支持完整 utterance 分块测试。
```

验收：

```bash
python scripts/03_eval_jscm_sim.py --config configs/eval/eval_jscm_awgn.yaml
```

期望输出：

```text
outputs/metrics/jscm_awgn_summary.json
outputs/metrics/jscm_awgn_per_utterance.csv
```

---

### Phase 5：Opus baseline

目标：完成 Opus 编解码 baseline。

任务：

```text
1. 实现 baselines/opus_codec.py。
2. 实现 baselines/opus_link.py。
3. 实现 scripts/04_eval_opus_sim.py。
4. 添加 test_opus_codec.py。
```

验收：

```bash
pytest tests/test_opus_codec.py
python scripts/04_eval_opus_sim.py \
  --data-config configs/data/librispeech_clean100.yaml \
  --radio-config configs/radio/opus_gfsk.yaml \
  --bitrate 12000
```

---

### Phase 6：JSCM 符号导出/恢复

目标：为 GRC/USRP 建立文件接口。

任务：

```text
1. 实现 radio/iq_io.py。
2. 实现 scripts/05_export_jscm_symbols.py。
3. 实现 scripts/06_recover_jscm_symbols.py。
4. 添加 test_symbol_io.py。
```

验收：

```bash
pytest tests/test_symbol_io.py
python scripts/05_export_jscm_symbols.py --checkpoint checkpoints/conformer_jscm_base/best.pt --num-utts 1
python scripts/06_recover_jscm_symbols.py --rx-real-seq <exported_f32> --meta <meta_json> --out-wav outputs/test_rec.wav
```

---

### Phase 7：GRC loopback

目标：不接 USRP，验证 GFSK/FM 软件闭环。

任务：

```text
1. 创建 grc/opus_gfsk/opus_gfsk_loopback.grc。
2. 创建 grc/jscm_fm/jscm_fm_loopback.grc。
3. 对每个 GRC 文件写 params_*.md。
4. 导出 generated Python flowgraph 到 grc/generated/。
```

验收：

```text
Opus: tx_bits.bin → GFSK loopback → rx_bits.bin，BER 可计算。
JSCM: tx_real_sequence.f32 → FM loopback → rx_real_sequence.f32，symbol MSE 可计算。
```

---

### Phase 8：USRP 文件链路

目标：接入 USRP，但仍使用文件输入输出。

任务：

```text
1. 创建 opus_gfsk_tx.grc / opus_gfsk_rx.grc。
2. 创建 jscm_fm_tx.grc / jscm_fm_rx.grc。
3. 先用固定 tone / random symbols 测试。
4. 再发送真实 Opus bits 和 JSCM symbols。
```

验收：

```text
USRP 线缆闭环下：
Opus GFSK 能恢复 bitstream。
JSCM FM 能恢复 real sequence。
Python 能完成最终语音重构和指标计算。
```

---

### Phase 9：实验报告

目标：生成对比结果。

任务：

```text
1. 实现 scripts/09_make_report.py。
2. 汇总 JSCM 与 Opus 指标。
3. 输出 csv/json/markdown 报告。
4. 可选生成 figures。
```

报告至少包含：

```text
方法：Conformer-JSCM, Opus baseline
信道：AWGN/Rayleigh/USRP
JSCM rate：800/1600/3200 complex symbols per segment
Opus bitrate：6/9/12/16 kbps
指标：PESQ/STOI/SI-SDR
```

---

## 18. 调试顺序

必须按以下顺序调试，不要跳步：

```text
1. STFT/iSTFT 重构测试。
2. Conformer-JSCM identity channel 过拟合小 batch。
3. Conformer-JSCM AWGN 仿真训练。
4. Opus 本地 encode/decode。
5. JSCM symbols 文件导出/读取一致性。
6. GRC JSCM FM loopback。
7. GRC Opus GFSK loopback。
8. USRP 线缆闭环 tone 测试。
9. USRP JSCM FM symbols 测试。
10. USRP Opus GFSK bits 测试。
11. 完整 test-clean 子集评估。
```

关键判断：

```text
如果 Python 仿真不通，不要调 GRC。
如果 GRC loopback 不通，不要接 USRP。
如果固定符号不通，不要发送真实语音。
```

---

## 19. 最小可运行版本范围

第一版只实现以下内容：

```text
必须：
- LibriSpeech manifest
- STFT real/imag dataset
- Conformer-JSCM Base
- AWGN channel
- train/eval scripts
- Opus local baseline
- JSCM symbol export/recover
- GRC loopback 文件说明

可以后置：
- Rayleigh
- OFDM
- ZMQ 实时接口
- 复杂 packet/CRC
- 生成式模型
- EnCodec/Mel/vocoder
- 大模型
```

---

## 20. 关键工程约束

1. 不要写一个巨大的 `main.py`。
2. 所有超参数必须放在 `configs/`。
3. 所有中间文件必须带 `meta.json`。
4. GRC 不负责模型推理。
5. 第一版只用文件接口，不做实时 ZMQ。
6. JSCM 输出必须检查平均功率。
7. USRP 前必须先通过 loopback。
8. Opus 和 JSCM 的评价指标必须统一使用原始 waveform 对齐后计算。
9. 测试集 `test-clean` 不能用于调参。
10. 每个实验输出不能覆盖历史结果。

---

## 21. 推荐依赖

`requirements.txt` 建议：

```text
torch
torchaudio
numpy
scipy
soundfile
librosa
pyyaml
tqdm
pandas
matplotlib
pytest
pesq
pystoi
```

系统依赖建议：

```text
opus-tools
ffmpeg
gnuradio
uhd-host
```

GNU Radio、UHD、USRP 驱动建议单独在系统环境安装，不强行放入 Python requirements。

---

## 22. 最终验收目标

当工程完成后，应能执行以下流程：

```bash
# 1. 生成 manifest
python scripts/01_make_manifest.py \
  --librispeech-root data/raw/LibriSpeech \
  --out-dir data/manifests

# 2. 训练 Conformer-JSCM
python scripts/02_train_conformer_jscm.py \
  --config configs/train/train_conformer_jscm.yaml

# 3. 仿真评估 JSCM
python scripts/03_eval_jscm_sim.py \
  --config configs/eval/eval_jscm_awgn.yaml

# 4. 仿真评估 Opus
python scripts/04_eval_opus_sim.py \
  --data-config configs/data/librispeech_clean100.yaml \
  --radio-config configs/radio/opus_gfsk.yaml \
  --bitrate 12000

# 5. 导出 JSCM 符号给 GRC/USRP
python scripts/05_export_jscm_symbols.py \
  --checkpoint checkpoints/conformer_jscm_base/best.pt \
  --data-config configs/data/librispeech_clean100.yaml \
  --model-config configs/model/conformer_jscm_base.yaml \
  --split test \
  --num-utts 10 \
  --out-dir outputs/exp004_jscm_fm_usrp/tx

# 6. GRC/USRP 产生 rx_real_sequence.f32 后，恢复语音
python scripts/06_recover_jscm_symbols.py \
  --checkpoint checkpoints/conformer_jscm_base/best.pt \
  --model-config configs/model/conformer_jscm_base.yaml \
  --rx-real-seq outputs/exp004_jscm_fm_usrp/rx/utt_0001_rx_real_sequence.f32 \
  --meta outputs/exp004_jscm_fm_usrp/tx/utt_0001_meta.json \
  --out-wav outputs/exp004_jscm_fm_usrp/rx/utt_0001_rec.wav
```

最终应该得到：

```text
Conformer-JSCM 在 AWGN/Rayleigh 仿真下的 PESQ/STOI/SI-SDR。
Opus baseline 在不同 bitrate 下的 PESQ/STOI/SI-SDR。
JSCM + FM + USRP 文件链路的恢复语音和 symbol MSE。
Opus + GFSK + USRP 文件链路的恢复语音和 BER/语音指标。
```

---

## 23. 后续扩展方向

在第一版稳定后，再考虑：

```text
1. Rayleigh / Rician / OFDM frequency-selective fading。
2. ZMQ 实时接口。
3. 更严格的 packet framing、CRC、FEC。
4. Conformer-JSCM streaming chunk 推理。
5. 生成式 refinement 分支。
6. VCTK OOD 测试。
7. JSCM FM 替换为更合理的模拟/数字混合调制。
8. 统一 Opus 和 JSCM 的 channel uses per second 计算。
```

---

## 24. 结论

本工程第一阶段应保持简单：

```text
LibriSpeech train-clean-100
STFT real/imag [2,257,100]
Conformer-JSCM Base
AWGN 仿真
Opus baseline
文件接口连接 GRC
JSCM 用 FM，Opus 用 GFSK
USRP 后置
```

最重要的是先跑通以下两个闭环：

```text
JSCM symbols → FM → rx symbols → JSCM decoder → waveform
Opus bits → GFSK → rx bits → Opus decoder → waveform
```

只要这两个中间文件闭环稳定，后续更换模型、码率、信道、调制方式和 USRP 参数都容易迭代。
