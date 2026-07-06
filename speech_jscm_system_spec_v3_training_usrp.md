# Speech-JSCM BLE-like USRP System Specification v3

> 用途：本 Markdown 用于指导 Codex 一步一步构建完整工程，包括 Python 脚本、Conformer-JSCM 模型、Opus baseline、BLE-like 帧结构、GNU Radio Companion `.grc` 流图、双 USRP 收发、SNR 标定和成对对比测试。  
> 本文档是原 `speech_jscm_system_spec.md` 的 v3 更新版，加入了 **训练闭环、可微信道层、纯 Python 仿真验收、双 USRP 同步/CFO 补偿、Opus/JSCM 公平资源统计、GRC runner 规范、安全功率边界和 Codex MVP 构建路径**。

---

## 0. 固定硬件与实验目标

### 0.1 固定硬件角色

本工程使用两台 USRP：

```text
TX USRP: 192.168.10.1
RX USRP: 192.168.10.2
```

GNU Radio / UHD 设备参数必须明确写入配置文件和 GRC 变量：

```yaml
usrp:
  tx_device_args: "addr=192.168.10.1"
  rx_device_args: "addr=192.168.10.2"
```

主机网卡建议配置到同一网段，例如：

```text
Host NIC: 192.168.10.10/24
TX USRP: 192.168.10.1
RX USRP: 192.168.10.2
```

验收命令：

```bash
ping 192.168.10.1
ping 192.168.10.2
uhd_find_devices --args "addr=192.168.10.1"
uhd_find_devices --args "addr=192.168.10.2"
uhd_usrp_probe --args "addr=192.168.10.1"
uhd_usrp_probe --args "addr=192.168.10.2"
```

若主机、网卡或系统服务已经占用 `192.168.10.1`，需要修改主机地址，不能和 TX USRP 冲突。

### 0.2 系统目标

构建一个用于比较以下两种语音通信系统的 BLE-like SDR testbed：

```text
System A: Opus + BLE-like GFSK + USRP
System B: Conformer-JSCM + BLE-like frame + FM + USRP
```

比较目标：

```text
1. 相同 BLE-like RF 参数；
2. 相同 TX/RX USRP、中心频率、采样率、RX gain、带宽；
3. 通过 tx_gain / tx_amp 控制接收端实测 SNR；
4. 在每个 tx_gain / tx_amp 下成对运行 Opus 和 JSCM；
5. 用 measured_snr_db 作为横坐标，对比 PESQ、STOI、SI-SDR、MCD、PER/frame-sync 等指标。
```

重要定义：

```text
本系统是 BLE-like SDR testbed，不是标准 BLE 协议栈。
Opus 使用 GFSK，尽量贴近 BLE LE 1M PHY 参数。
JSCM 使用 FM 直接调制 float 型连续数据，不是标准 BLE PHY。
两者共享 BLE-like 外层帧、RF 频点、采样率、发射功率控制和测试流程。
```

---

## 1. 总体工程目录

Codex 应按以下目录结构构建工程：

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
│   ├── radio/
│   │   ├── ble_like_usrp.yaml
│   │   ├── snr_sweep.yaml
│   │   ├── opus_gfsk.yaml
│   │   └── jscm_fm.yaml
│   └── experiments/
│       └── paired_opus_jscm_ble.yaml
│
├── data/
│   ├── raw/
│   │   └── LibriSpeech/                  # 不提交 git
│   ├── manifests/
│   │   ├── train_clean_100.jsonl
│   │   ├── dev_clean.jsonl
│   │   ├── test_clean.jsonl
│   │   └── test_clean_small.jsonl
│   └── cache/
│       └── stft/                         # 可选缓存，不提交 git
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
│       │   ├── gfsk_sim.py
│       │   └── fm_sim.py
│       │
│       ├── baselines/
│       │   ├── opus_codec.py
│       │   └── opus_link.py
│       │
│       ├── radio/
│       │   ├── ble_frame.py              # 统一 BLE-like 帧定义
│       │   ├── opus_frame.py             # Opus byte/bits 组帧和解帧
│       │   ├── jscm_float_frame.py       # JSCM float 组帧和解帧
│       │   ├── iq_io.py
│       │   ├── snr.py
│       │   ├── sync.py
│       │   ├── grc_runner.py             # subprocess 调用 generated GRC Python
│       │   └── usrp_config.py
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
│           ├── subprocess_utils.py
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
│   └── usrp/
│       ├── 00_generate_calib_pilot.py
│       ├── 01_capture_noise.py
│       ├── 02_run_calib_sweep.py
│       ├── 03_estimate_snr_lut.py
│       ├── 04_prepare_test_payloads.py
│       ├── 05_run_paired_snr_test.py
│       ├── 06_decode_opus_results.py
│       ├── 07_decode_jscm_results.py
│       ├── 08_compute_metrics.py
│       ├── 09_plot_results.py
│       └── 10_make_report.py
│
├── grc/
│   ├── README.md
│   ├── common/
│   │   ├── calib_tx.grc
│   │   ├── calib_rx.grc
│   │   ├── rx_iq_capture.grc
│   │   └── loopback_channel.grc
│   ├── opus_gfsk/
│   │   ├── opus_gfsk_loopback.grc
│   │   ├── opus_gfsk_tx.grc
│   │   ├── opus_gfsk_rx.grc
│   │   └── params_opus_gfsk.md
│   ├── jscm_fm/
│   │   ├── jscm_fm_loopback.grc
│   │   ├── jscm_fm_tx.grc
│   │   ├── jscm_fm_rx.grc
│   │   └── params_jscm_fm.md
│   └── generated/
│       ├── calib_tx.py
│       ├── calib_rx.py
│       ├── rx_iq_capture.py
│       ├── opus_gfsk_tx.py
│       ├── opus_gfsk_rx.py
│       ├── jscm_fm_tx.py
│       └── jscm_fm_rx.py
│
├── checkpoints/
│   └── conformer_jscm_base/
│
├── outputs/
│   ├── calibration/
│   ├── prepared/
│   │   ├── opus/
│   │   └── jscm/
│   ├── usrp_paired_ble/
│   ├── wav/
│   ├── iq/
│   ├── logs/
│   ├── metrics/
│   └── figures/
│
└── tests/
    ├── test_stft_reconstruction.py
    ├── test_power_norm.py
    ├── test_ble_frame.py
    ├── test_opus_frame.py
    ├── test_jscm_float_frame.py
    ├── test_awgn_channel.py
    ├── test_opus_codec.py
    ├── test_symbol_io.py
    └── test_snr_estimation.py
```

---

## 2. 数据集与输入规格

### 2.1 数据集划分

使用 LibriSpeech：

```text
Training:    LibriSpeech train-clean-100
Validation:  LibriSpeech dev-clean
Test:        LibriSpeech test-clean
Optional:    LibriSpeech test-other，用于鲁棒性测试
```

第一版不从 `train-clean-100` 内部再切验证集，直接使用官方 `dev-clean` 作为验证集。

### 2.2 每条训练样本大小

```text
sample_rate      = 16000 Hz
segment_samples  = 16384
segment_duration = 1.024 s
waveform.shape   = [1, 16384]
```

训练时：

```text
长度 >= 16384 samples：随机裁剪 16384 samples。
长度 <  16384 samples：zero-pad 到 16384 samples。
```

验证/测试：

```text
早期调试：center crop 16384 samples。
正式测试：完整 utterance 分块，最后一个 chunk padding，重构后裁回原长度。
```

### 2.3 STFT 特征规格

第一版使用 STFT real/imag，不使用 Mel。

```text
n_fft      = 512
win_length = 400      # 25 ms at 16 kHz
hop_length = 160      # 10 ms at 16 kHz
center     = false
window     = hann
feature    = real_imag
```

维度：

```text
F = 512 / 2 + 1 = 257
T = floor((16384 - 512) / 160) + 1 = 100
x.shape = [2, 257, 100]
x_batch.shape = [B, 2, 257, 100]
y_hat.shape = [B, 2, 257, 100]
```

---

## 3. Conformer-JSCM 模型与 JSCM 符号设置

### 3.1 基本链路

```text
waveform
→ STFT real/imag
→ Conformer-JSCM encoder
→ channel mapper
→ power normalization
→ complex channel symbols
→ BLE-like float frames
→ FM modulation
→ TX USRP 192.168.10.1
→ RF / cable / attenuator / air
→ RX USRP 192.168.10.2
→ FM demodulation
→ BLE-like float deframe
→ received complex symbols
→ Conformer-JSCM decoder
→ reconstructed STFT real/imag
→ iSTFT
→ reconstructed waveform
```

### 3.2 Base 模型规格

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
  num_complex_symbols: 1568    # BLE-like frame 对齐后的 Mid 档

power_norm:
  average_power: 1.0
```

### 3.3 JSCM 码率档位

为了和 BLE-like frame 的 56 个 float32 slot 对齐，JSCM 每帧承载：

```text
56 float32 real slots = 28 complex symbols
complex symbol z_k = Re_k + j Im_k
payload order = [Re0, Im0, Re1, Im1, ..., Re27, Im27]
```

每个 1.024 s segment 的三档设置：

| Rate | complex symbols / segment | radio frames / segment | complex symbols / frame | real dims | 近似压缩比 |
|---|---:|---:|---:|---:|---:|
| Low | 784 | 28 | 28 | 1568 | 1/32.8 |
| Mid | 1568 | 56 | 28 | 3136 | 1/16.4 |
| High | 3136 | 112 | 28 | 6272 | 1/8.2 |

默认主实验使用：

```text
JSCM Mid: 1568 complex symbols / 1.024 s segment = 56 radio frames
```

原因：Opus 20 ms frame 在 1.024 s 内约 51–52 个 Opus frame，与 JSCM Mid 的 56 个 radio frame 最接近，最适合第一版公平对比。

---

## 4. Opus + GFSK baseline 原理

### 4.1 基本链路

```text
waveform
→ 20 ms Opus frame
→ Opus encode
→ Opus payload bytes
→ BLE-like byte frame
→ bit stream
→ GFSK modulation
→ TX USRP 192.168.10.1
→ RF / cable / attenuator / air
→ RX USRP 192.168.10.2
→ GFSK demodulation
→ bit stream
→ BLE-like byte deframe
→ Opus decode
→ reconstructed waveform
```

### 4.2 Opus 参数

第一版建议：

```yaml
opus:
  sample_rate: 16000
  channels: 1
  application: voip
  frame_ms: 20
  bitrates: [6000, 9000, 12000, 16000, 24000]
  primary_bitrate: 12000
```

第一版每个 BLE-like radio frame 放一个 Opus encoded frame。若 payload 不足 224 bytes，右侧补零；CommonHeader 中记录真实 Opus payload byte 数。

---

## 5. 统一 BLE-like 帧结构

### 5.1 外层帧目标

Opus 与 JSCM 使用统一外层帧，payload 语义不同：

```text
Opus payload: bytes / bits，用 GFSK 数字调制。
JSCM payload: float32 slots，用 FM 直接调制连续值。
```

公平性原则：

```text
统一帧头结构；
统一帧时长定义；
统一 preamble/access/header/trailer 开销；
统一 RF 频点、采样率、带宽、TX/RX gain 设置；
最终用 measured_snr_db 和实际空口资源统计对比。
```

### 5.2 帧字段

```text
Frame = Preamble + Access Address + PHY Header + Common Header + App Payload + Trailer
```

建议长度：

| Field | byte-equivalent | bit-time | 用途 |
|---|---:|---:|---|
| Preamble | 8 bytes | 64 | 帧检测、粗同步 |
| Access Address | 4 bytes | 32 | 固定同步字 |
| PHY Header | 2 bytes | 16 | payload length / flags |
| Common Header | 24 bytes | 192 | stream id、frame id、rate id、sample offset |
| App Payload | 224 bytes-equivalent | 1792 | Opus bytes 或 JSCM float slots |
| Trailer | 3 bytes-equivalent | 24 | Opus CRC24 或 JSCM guard/pilot |

总时长，不含额外 guard：

```text
64 + 32 + 16 + 192 + 1792 + 24 = 2120 bit-times
BLE-like symbol_rate = 1e6 symbol/s
frame_duration ≈ 2.120 ms
```

若加入前后 guard，guard 必须对 Opus 和 JSCM 相同，并计入空口资源统计。

### 5.3 CommonHeader 结构

统一 24 bytes：

```c
struct CommonHeader {
    uint8_t  version;          // 初始为 1
    uint8_t  payload_type;     // 0=OPUS_BYTES, 1=JSCM_FLOAT32
    uint8_t  rate_id;          // Opus bitrate id 或 JSCM rate id
    uint8_t  flags;            // LAST_FRAME, HAS_PADDING 等

    uint32_t stream_id;        // utterance/session id
    uint32_t frame_idx;        // radio frame index
    uint32_t sample_offset;    // 原始语音起始 sample index

    uint16_t source_samples;   // 本帧覆盖的源语音 samples
    uint16_t payload_units;    // Opus: valid bytes; JSCM: valid float slots
    uint16_t header_crc16;     // CommonHeader 前 22 bytes 的 CRC16
    uint16_t reserved;
};
```

### 5.4 Opus payload

```text
App Payload = 224 bytes
payload_units = valid_opus_bytes
不足 224 bytes 时右侧补 0
Trailer = CRC24
```

接收端：

```text
CRC pass → Opus decode
CRC fail → frame erasure / packet loss concealment / mark lost
```

### 5.5 JSCM payload

```text
App Payload = 56 float32 slots
56 float32 = 224 bytes-equivalent
56 float32 = 28 complex symbols
payload_units = valid_float_slots
不足 56 float slots 时补 0.0
Trailer = guard / pilot，不对 payload 做 hard CRC
```

JSCM payload 不能使用 hard CRC 判断丢弃。因为 FM 后恢复的是连续值，不可能 bit-exact。只允许对 JSCM 的 CommonHeader 做 CRC 或可靠解码，payload 应保留连续退化特性。

### 5.6 JSCM float 的 byte-equivalent 时长对齐

这是关键实现点。

由于 JSCM 是直接 FM 调制 float，不是把 float32 的 32 个 bit 数字化发送，为了保证与 Opus 的 BLE-like frame 时长公平，必须定义：

```text
1 个 JSCM float32 slot 占用 32 个 BLE-like bit-time。
```

因此：

```text
56 float32 slots × 32 bit-time/slot = 1792 bit-time
= 224 bytes × 8 bit/byte
```

Python 生成 JSCM FM 输入序列时必须执行 sample-and-hold：

```text
BLE-like symbol_rate = 1e6
samples_per_symbol = 4
USRP samp_rate = 4e6

1 bit-time = 4 complex/baseband samples
1 float32 slot = 32 bit-times = 128 float samples before FM
```

JSCM frame 的 float waveform 生成规则：

```text
preamble/header/access/trailer: 每个 bit 映射为 -1.0/+1.0，并重复 4 samples。
payload float slot: 每个 float 值保持 32 bit-times，即重复 128 samples。
```

这样 Opus 和 JSCM 的 radio frame duration 一致。

---

## 6. BLE-like RF 参数

第一版采用 BLE LE 1M-like 参数：

```yaml
ble_like:
  phy: LE_1M_like
  center_freq: 2440e6
  channel_spacing: 2e6
  symbol_rate: 1e6
  samples_per_symbol: 4
  samp_rate: 4e6
```

建议先使用一个较干净的 data-channel-like 频点，例如：

```text
center_freq = 2440 MHz
```

后续可扩展为多频点：

```yaml
center_freq_list:
  - 2402e6
  - 2440e6
  - 2480e6
```

注意：实际空口发射必须符合实验室许可和当地无线电管理要求。早期优先使用同轴线 + 衰减器或屏蔽箱。

### 6.1 Opus/GFSK 参数

```yaml
gfsk:
  bit_rate: 1e6
  samples_per_symbol: 4
  bt: 0.5
  modulation_index: 0.5
  freq_deviation: 250e3
  sensitivity: 0.3926990817  # 2*pi*250e3/4e6
```

### 6.2 JSCM/FM 参数

```yaml
fm:
  samp_rate: 4e6
  max_deviation_hz: 250e3
  frequency_modulator_sensitivity: 0.3926990817  # 2*pi*250e3/4e6
  quadrature_demod_gain: 2.546479089             # 4e6/(2*pi*250e3)
  input_clip: [-1.0, 1.0]
  payload_float_hold_bit_times: 32
  samples_per_bit_time: 4
  samples_per_float_slot: 128
```

---

## 7. 配置文件模板

### 7.1 `configs/radio/ble_like_usrp.yaml`

```yaml
ble_like:
  phy: LE_1M_like
  center_freq: 2440e6
  channel_spacing: 2e6
  symbol_rate: 1e6
  samples_per_symbol: 4
  samp_rate: 4e6

usrp:
  tx_device_args: "addr=192.168.10.1"
  rx_device_args: "addr=192.168.10.2"
  tx_antenna: "TX/RX"
  rx_antenna: "RX2"
  tx_bw: 2e6
  rx_bw: 2e6
  rx_gain: 20
  clock_source: "internal"
  time_source: "internal"

safety:
  max_tx_baseband_abs: 0.7
  max_rx_iq_abs: 0.7
  agc: false
```

### 7.2 `configs/radio/snr_sweep.yaml`

```yaml
sweep:
  tx_gain_list: [0, 5, 10, 15, 20, 25, 30]
  tx_amp_list: [0.03, 0.05, 0.07, 0.10]
  target_snr_db: [-5, 0, 5, 10, 15, 20]

calibration:
  noise_capture_seconds: 2.0
  pilot_capture_seconds: 1.0
  snr_drift_tolerance_db: 1.0
  snr_lut_csv: outputs/calibration/snr_lut.csv
```

### 7.3 `configs/experiments/paired_opus_jscm_ble.yaml`

```yaml
experiment:
  name: paired_opus_jscm_ble
  repeats_per_snr: 3
  run_order_policy: alternating
  test_manifest: data/manifests/test_clean_small.jsonl

systems:
  opus:
    enabled: true
    bitrate: 12000
    frame_ms: 20
    modulation: gfsk

  jscm:
    enabled: true
    checkpoint: checkpoints/conformer_jscm_base/best.pt
    rate_id: mid
    complex_symbols_per_segment: 1568
    frames_per_segment: 56
    modulation: fm

outputs:
  root: outputs/usrp_paired_ble
```

---

## 8. GRC 图设计总则

GRC 只负责：

```text
文件输入
→ 基带调制 / USRP TX
→ USRP RX / 基带解调
→ 文件输出
```

Python 负责：

```text
组帧、解帧、SNR 标定、实验调度、Opus 编解码、JSCM 编解码、指标计算。
```

所有 GRC generated Python 必须支持命令行参数：

```text
--tx-device-args
--rx-device-args
--center-freq
--samp-rate
--tx-gain
--rx-gain
--tx-amp
--tx-bw
--rx-bw
--input-file
--output-file
--iq-output-file
--num-samps
--run-seconds
```

示例：

```bash
python grc/generated/opus_gfsk_tx.py \
  --tx-device-args "addr=192.168.10.1" \
  --center-freq 2440e6 \
  --samp-rate 4e6 \
  --tx-gain 10 \
  --tx-amp 0.05 \
  --input-file outputs/prepared/opus/tx_bits.bin

python grc/generated/opus_gfsk_rx.py \
  --rx-device-args "addr=192.168.10.2" \
  --center-freq 2440e6 \
  --samp-rate 4e6 \
  --rx-gain 20 \
  --output-file outputs/run/opus/rx_bits.bin \
  --iq-output-file outputs/run/opus/rx_iq_before_demod.c64
```

---

## 9. GRC 图 1：`common/calib_tx.grc`

### 9.1 作用

发送固定 complex64 pilot，用于建立：

```text
(tx_gain, tx_amp) → measured_snr_db
```

### 9.2 输入

```text
outputs/calibration/calib_pilot.c64
```

Python 生成：

```text
guard zeros + PN/BPSK complex pilot + constant-amplitude pilot + guard zeros
```

### 9.3 流图

```text
File Source complex64
    ↓
Multiply Const complex       # tx_amp
    ↓
File Sink complex64 optional # calib_tx_baseband.c64
    ↓
UHD: USRP Sink               # device addr=192.168.10.1
```

UHD Sink 参数：

```text
device_args = "addr=192.168.10.1"
center_freq = center_freq
gain        = tx_gain
samp_rate   = 4e6
bandwidth   = 2e6
antenna     = TX/RX
```

---

## 10. GRC 图 2：`common/calib_rx.grc`

### 10.1 作用

接收校准 pilot 或噪声，保存原始 IQ，由 Python 离线估计 SNR。

### 10.2 流图

```text
UHD: USRP Source             # device addr=192.168.10.2
    ↓
DC Blocker
    ↓
Low Pass Filter              # cutoff 1.0 MHz, transition 200 kHz, 可按实际调
    ↓
Head                         # num_samps = run_seconds * samp_rate
    ↓
File Sink complex64
```

UHD Source 参数：

```text
device_args = "addr=192.168.10.2"
center_freq = center_freq
gain        = rx_gain
samp_rate   = 4e6
bandwidth   = 2e6
antenna     = RX2
```

禁止使用 AGC。

输出：

```text
outputs/calibration/noise_iq.c64
outputs/calibration/calib_rx_iq_gainXX_ampYY.c64
```

---

## 11. GRC 图 3：`opus_gfsk/opus_gfsk_tx.grc`

### 11.1 输入

```text
outputs/prepared/opus/tx_bits.bin
```

`tx_bits.bin` 由 Python 生成，内容为完整 BLE-like frame bit stream：

```text
Preamble + Access Address + PHY Header + Common Header + Opus Payload + CRC24
```

### 11.2 流图

```text
File Source uchar / byte bits
    ↓
Unpack K Bits / Repack Bits if needed
    ↓
GFSK Mod
    ↓
Multiply Const complex          # tx_amp
    ↓
File Sink complex64 optional    # opus_tx_baseband.c64
    ↓
UHD: USRP Sink                  # addr=192.168.10.1
```

GFSK Mod 参数：

```text
samples_per_symbol = 4
bt = 0.5
sensitivity = 0.3926990817
```

注意：如果 GNU Radio 版本中 GFSK Mod 的 sensitivity 定义不同，必须用 loopback 和频谱图验证主频偏接近 ±250 kHz。

---

## 12. GRC 图 4：`opus_gfsk/opus_gfsk_rx.grc`

### 12.1 输出

```text
outputs/run/opus/rx_iq_before_demod.c64
outputs/run/opus/rx_bits.bin
```

### 12.2 流图

```text
UHD: USRP Source                # addr=192.168.10.2
    ↓
DC Blocker
    ↓
Low Pass Filter
    ↓
File Sink complex64             # rx_iq_before_demod.c64
    ↓
GFSK Demod
    ↓
Binary Slicer
    ↓
File Sink uchar                 # rx_bits.bin
```

第一版不要在 GRC 内做 deframe。Python 负责：

```text
access address search
frame alignment
header parse
CRC check
Opus payload extraction
Opus decode
```

---

## 13. GRC 图 5：`jscm_fm/jscm_fm_tx.grc`

### 13.1 输入

```text
outputs/prepared/jscm/tx_fm_input.f32
```

`tx_fm_input.f32` 由 Python 生成，采样率已经是 `4e6`，包含完整 BLE-like frame 的 FM 输入 float waveform：

```text
preamble/access/header/trailer bits: -1/+1 repeated 4 samples per bit-time
JSCM float payload: each float32 slot held for 32 bit-times = 128 samples
```

因此 GRC 中不再负责 JSCM payload slot 的重复扩展。

### 13.2 流图

```text
File Source float32
    ↓
Clip / Limit optional          # [-1.0, +1.0]
    ↓
Frequency Modulator            # sensitivity = 0.3926990817
    ↓
Multiply Const complex         # tx_amp
    ↓
File Sink complex64 optional   # jscm_tx_baseband_after_fm.c64
    ↓
UHD: USRP Sink                 # addr=192.168.10.1
```

---

## 14. GRC 图 6：`jscm_fm/jscm_fm_rx.grc`

### 14.1 输出

```text
outputs/run/jscm/rx_iq_before_demod.c64
outputs/run/jscm/rx_fm_output.f32
```

### 14.2 流图

```text
UHD: USRP Source               # addr=192.168.10.2
    ↓
DC Blocker
    ↓
Low Pass Filter
    ↓
File Sink complex64            # rx_iq_before_demod.c64
    ↓
Quadrature Demod / FM Demod    # gain = 2.546479089
    ↓
Low Pass Filter
    ↓
File Sink float32              # rx_fm_output.f32
```

第一版不要在 GRC 内做 float deframe。Python 负责：

```text
float preamble correlation
header bit recovery
CommonHeader CRC16 check
payload window extraction
每 128 samples 估计一个 float slot
float slots → complex symbols
JSCM decoder
```

JSCM payload float slot 恢复建议：

```python
# samples_per_float_slot = 128
slot_value = mean(rx_fm_output[start:end])
```

---

## 15. GRC 图 7：`common/rx_iq_capture.grc`

该图只抓原始 IQ，用于排查问题。

```text
UHD: USRP Source, addr=192.168.10.2
    ↓
DC Blocker
    ↓
Low Pass Filter
    ↓
Head
    ↓
File Sink complex64
```

用途：

```text
检查 2.4 GHz 干扰；
检查中心频偏；
检查 RX ADC 是否饱和；
检查 Opus/JSCM 同一 tx_gain 下接收功率是否接近；
离线画频谱图。
```

---

## 16. SNR 标定方法

### 16.1 原则

不要把 `tx_gain` 直接当成 SNR。必须用 RX 端实测：

```text
TX off → capture noise → noise_power
TX pilot on → capture pilot → measured_snr_db
```

最终曲线横坐标使用：

```text
measured_snr_db
```

不是：

```text
tx_gain
```

### 16.2 SNR 估计

Python 中实现：

```python
def estimate_pilot_snr(rx_pilot, tx_pilot):
    h = np.vdot(tx_pilot, rx_pilot) / (np.vdot(tx_pilot, tx_pilot) + 1e-12)
    err = rx_pilot - h * tx_pilot
    sig_pwr = np.mean(np.abs(h * tx_pilot) ** 2)
    noise_pwr = np.mean(np.abs(err) ** 2)
    return 10 * np.log10((sig_pwr + 1e-12) / (noise_pwr + 1e-12))
```

同时检查：

```python
max_abs_rx = np.max(np.abs(rx_iq))
valid = max_abs_rx < 0.7
```

若 RX 接近饱和，该点作废。

### 16.3 标定流程

```text
1. 固定 RX gain、center_freq、samp_rate、天线/同轴/衰减器。
2. TX off，运行 calib_rx，保存 noise_iq.c64。
3. 对每组 tx_gain / tx_amp：
   a. 运行 calib_rx 后台接收；
   b. 运行 calib_tx 发送 pilot；
   c. 保存 calib_rx_iq_gainXX_ampYY.c64；
   d. Python 估计 measured_snr_db。
4. 生成 outputs/calibration/snr_lut.csv。
5. 从 LUT 中选择最接近 target_snr_db 的 tx_gain / tx_amp。
```

输出 CSV：

```csv
tx_gain,tx_amp,measured_snr_db,rx_power,noise_power,max_abs_rx,valid
0,0.03,-7.2,...
5,0.03,-2.1,...
10,0.03,3.4,...
```

---

## 17. 成对对比测试流程

### 17.1 推荐顺序

必须在同一个 tx_gain / tx_amp 下成对运行 Opus 和 JSCM，然后再换下一个 tx_gain / tx_amp。

不要这样：

```text
先跑完所有 Opus SNR 点，再跑所有 JSCM SNR 点。
```

应该这样：

```text
for each selected_snr_point:
    for repeat in repeats:
        calib_pre
        run Opus and JSCM in alternating order
        calib_post
```

推荐交替顺序：

```text
repeat 0: Opus → JSCM
repeat 1: JSCM → Opus
repeat 2: Opus → JSCM
repeat 3: JSCM → Opus
```

如果：

```text
abs(calib_post_snr - calib_pre_snr) > 1.0 dB
```

则该 run 标记为 unstable，不用于最终主曲线。

### 17.2 单个 SNR 点的完整过程

```text
固定：
  center_freq = 2440 MHz
  samp_rate = 4 MS/s
  rx_gain = 20
  tx_gain = 当前点
  tx_amp = 当前点
  TX USRP = 192.168.10.1
  RX USRP = 192.168.10.2

Step 1: calib_pre
Step 2: run Opus + GFSK
Step 3: run JSCM + FM
Step 4: calib_post
Step 5: decode both
Step 6: compute metrics
Step 7: save all meta/log/files
```

---

## 18. Python 脚本设计

### 18.1 `scripts/usrp/00_generate_calib_pilot.py`

职责：生成校准 pilot。

输出：

```text
outputs/calibration/calib_pilot.c64
outputs/calibration/calib_pilot_meta.json
```

meta 示例：

```json
{
  "samp_rate": 4000000,
  "guard_samples": 8000,
  "pilot_start": 8000,
  "pilot_len": 4096,
  "pilot_type": "bpsk_pn"
}
```

### 18.2 `scripts/usrp/01_capture_noise.py`

职责：调用 `grc/generated/calib_rx.py`，在 TX off 时抓噪声。

输出：

```text
outputs/calibration/noise_iq.c64
outputs/calibration/noise_stats.json
```

### 18.3 `scripts/usrp/02_run_calib_sweep.py`

职责：扫描 `tx_gain_list × tx_amp_list`。

伪代码：

```python
for tx_gain in tx_gain_list:
    for tx_amp in tx_amp_list:
        rx_proc = start_calib_rx(tx_gain, tx_amp)
        sleep(rx_warmup)
        run_calib_tx(tx_gain, tx_amp)
        wait_tx_done()
        sleep(rx_tail)
        stop_rx(rx_proc)
```

### 18.4 `scripts/usrp/03_estimate_snr_lut.py`

职责：从校准 IQ 文件估计 SNR，生成 LUT。

输出：

```text
outputs/calibration/snr_lut.csv
outputs/calibration/selected_snr_points.yaml
```

### 18.5 `scripts/usrp/04_prepare_test_payloads.py`

职责：为 Opus 和 JSCM 生成待发送文件。

Opus 输出：

```text
outputs/prepared/opus/tx_bits.bin
outputs/prepared/opus/tx_frames_meta.jsonl
```

JSCM 输出：

```text
outputs/prepared/jscm/tx_symbols.c64
outputs/prepared/jscm/tx_float_frames.f32
outputs/prepared/jscm/tx_fm_input.f32
outputs/prepared/jscm/tx_frames_meta.jsonl
```

JSCM 生成规则：

```text
1. Conformer-JSCM encoder 输出 complex symbols。
2. 每 28 complex symbols 组成一个 radio frame payload。
3. 28 complex → 56 float slots。
4. 每个 float slot clip 到 [-1, 1]。
5. 每个 float slot hold 128 samples，生成 4 MS/s 的 tx_fm_input.f32。
```

### 18.6 `scripts/usrp/05_run_paired_snr_test.py`

核心脚本，负责成对运行。

伪代码：

```python
def run_paired_test(config):
    snr_points = load_selected_snr_points()

    for point in snr_points:
        tx_gain = point["tx_gain"]
        tx_amp = point["tx_amp"]

        for repeat in range(repeats_per_snr):
            order = ["opus", "jscm"] if repeat % 2 == 0 else ["jscm", "opus"]

            calib_pre = run_calibration_once(tx_gain, tx_amp, tag="pre")

            for system in order:
                if system == "opus":
                    run_opus_once(tx_gain, tx_amp, repeat)
                else:
                    run_jscm_once(tx_gain, tx_amp, repeat)

            calib_post = run_calibration_once(tx_gain, tx_amp, tag="post")

            drift = abs(calib_post["snr_db"] - calib_pre["snr_db"])
            save_pair_meta(drift=drift, unstable=drift > tolerance_db)
```

单次 Opus：

```python
def run_opus_once(tx_gain, tx_amp, repeat):
    rx_proc = start_grc_rx("opus_gfsk_rx.py")
    sleep(rx_warmup)
    run_grc_tx("opus_gfsk_tx.py", tx_gain=tx_gain, tx_amp=tx_amp)
    wait_tx_done()
    sleep(rx_tail)
    stop_rx(rx_proc)
```

单次 JSCM：

```python
def run_jscm_once(tx_gain, tx_amp, repeat):
    rx_proc = start_grc_rx("jscm_fm_rx.py")
    sleep(rx_warmup)
    run_grc_tx("jscm_fm_tx.py", tx_gain=tx_gain, tx_amp=tx_amp)
    wait_tx_done()
    sleep(rx_tail)
    stop_rx(rx_proc)
```

### 18.7 `scripts/usrp/06_decode_opus_results.py`

职责：

```text
rx_bits.bin
→ access address search
→ frame parse
→ CommonHeader parse
→ CRC24 check
→ Opus payload extraction
→ Opus decode
→ reconstructed wav
```

输出：

```text
outputs/.../opus/reconstructed/*.wav
outputs/.../opus/decode_stats.json
```

统计：

```json
{
  "num_frames": 1000,
  "crc_pass": 912,
  "crc_fail": 88,
  "packet_error_rate": 0.088,
  "opus_decode_fail": 12
}
```

### 18.8 `scripts/usrp/07_decode_jscm_results.py`

职责：

```text
rx_fm_output.f32
→ preamble correlation
→ access/header recovery
→ CommonHeader CRC16 check
→ payload sample window extraction
→ 128 samples average per float slot
→ 56 float slots per frame
→ 28 complex symbols per frame
→ concatenate segment symbols
→ JSCM decoder
→ iSTFT
→ reconstructed wav
```

输出：

```text
outputs/.../jscm/reconstructed/*.wav
outputs/.../jscm/decode_stats.json
```

统计：

```json
{
  "num_frames_detected": 980,
  "num_frames_expected": 1000,
  "frame_sync_rate": 0.98,
  "mean_pilot_snr_db": 4.7,
  "payload_float_rms": 0.42,
  "payload_clip_rate": 0.001
}
```

### 18.9 `scripts/usrp/08_compute_metrics.py`

职责：统一计算：

```text
PESQ
STOI
SI-SDR
MCD
Opus PER
JSCM frame_sync_rate
measured_snr_db_pre/post/mean
```

输出：

```text
outputs/usrp_paired_ble/metrics/per_utterance.csv
outputs/usrp_paired_ble/metrics/summary_by_snr.csv
```

字段示例：

```csv
system,target_snr_db,measured_snr_db,repeat,utt_id,pesq,stoi,sisdr,mcd,per,frame_sync_rate,unstable
opus,5,4.7,0,19-198-0000,...
jscm,5,4.6,0,19-198-0000,...
```

### 18.10 `scripts/usrp/09_plot_results.py`

绘图：

```text
PESQ vs measured_snr_db
STOI vs measured_snr_db
SI-SDR vs measured_snr_db
MCD vs measured_snr_db
Opus PER vs measured_snr_db
JSCM frame_sync_rate vs measured_snr_db
pre/post SNR drift
RX max_abs / clipping diagnostics
```

横坐标必须是 `measured_snr_db`。

### 18.11 `scripts/usrp/10_make_report.py`

输出：

```text
outputs/usrp_paired_ble/report.md
outputs/usrp_paired_ble/figures/*.png
outputs/usrp_paired_ble/metrics/*.csv
```

报告必须包含：

```text
硬件 IP
BLE-like RF 参数
SNR LUT
每个 SNR 点的 tx_gain / tx_amp
Opus/JSCM 运行顺序
pre/post SNR drift
语音指标曲线
异常 run 列表
```

---

## 19. 输出目录规范

每个 run 目录：

```text
outputs/usrp_paired_ble/
└── snr_target_005_gain10_amp005/
    └── repeat_00/
        ├── pair_meta.json
        ├── calib_pre/
        │   ├── rx_iq.c64
        │   └── snr.json
        ├── opus/
        │   ├── rx_iq_before_demod.c64
        │   ├── rx_bits.bin
        │   ├── reconstructed/
        │   ├── decode_stats.json
        │   ├── metrics.csv
        │   └── grc_log.txt
        ├── jscm/
        │   ├── rx_iq_before_demod.c64
        │   ├── rx_fm_output.f32
        │   ├── reconstructed/
        │   ├── decode_stats.json
        │   ├── metrics.csv
        │   └── grc_log.txt
        └── calib_post/
            ├── rx_iq.c64
            └── snr.json
```

`pair_meta.json` 示例：

```json
{
  "tx_usrp": "192.168.10.1",
  "rx_usrp": "192.168.10.2",
  "center_freq": 2440000000.0,
  "samp_rate": 4000000.0,
  "rx_gain": 20,
  "tx_gain": 10,
  "tx_amp": 0.05,
  "target_snr_db": 5,
  "measured_snr_pre_db": 4.7,
  "measured_snr_post_db": 4.5,
  "snr_drift_db": 0.2,
  "run_order": ["opus", "jscm"],
  "unstable": false
}
```

---

## 20. Codex 构建顺序与验收标准

### 阶段 1：基础工程与配置

实现：

```text
configs/radio/ble_like_usrp.yaml
configs/radio/snr_sweep.yaml
configs/experiments/paired_opus_jscm_ble.yaml
src/speech_jscm/radio/usrp_config.py
```

验收：

```text
能读取 TX/RX USRP IP：192.168.10.1 / 192.168.10.2。
能生成 UHD device_args。
```

### 阶段 2：BLE-like frame 模块

实现：

```text
src/speech_jscm/radio/ble_frame.py
src/speech_jscm/radio/opus_frame.py
src/speech_jscm/radio/jscm_float_frame.py
tests/test_ble_frame.py
tests/test_opus_frame.py
tests/test_jscm_float_frame.py
```

验收：

```text
CommonHeader pack/unpack 一致。
Opus frame bits 可恢复 payload。
JSCM float frame 可恢复 56 float slots。
JSCM tx_fm_input.f32 的每个 payload float slot 重复 128 samples。
```

### 阶段 3：GRC loopback

实现：

```text
grc/opus_gfsk/opus_gfsk_loopback.grc
grc/jscm_fm/jscm_fm_loopback.grc
```

验收：

```text
Opus bits loopback BER 接近 0。
JSCM float loopback payload MSE 很小。
```

### 阶段 4：校准 GRC 和 SNR LUT

实现：

```text
grc/common/calib_tx.grc
grc/common/calib_rx.grc
scripts/usrp/00_generate_calib_pilot.py
scripts/usrp/01_capture_noise.py
scripts/usrp/02_run_calib_sweep.py
scripts/usrp/03_estimate_snr_lut.py
```

验收：

```text
能生成 snr_lut.csv。
tx_gain 或 tx_amp 增大时 measured_snr_db 基本单调上升。
rx_iq max_abs < 0.7。
```

### 阶段 5：Opus USRP 链路

实现：

```text
grc/opus_gfsk/opus_gfsk_tx.grc
grc/opus_gfsk/opus_gfsk_rx.grc
scripts/usrp/06_decode_opus_results.py
```

验收：

```text
高 SNR 下 Opus frame CRC pass rate 高。
Opus reconstructed wav 可生成。
PESQ/STOI/SI-SDR 可计算。
```

### 阶段 6：JSCM USRP 链路

实现：

```text
grc/jscm_fm/jscm_fm_tx.grc
grc/jscm_fm/jscm_fm_rx.grc
scripts/usrp/07_decode_jscm_results.py
```

验收：

```text
高 SNR 下 JSCM frame sync rate 高。
能从 rx_fm_output.f32 恢复 complex symbols。
JSCM reconstructed wav 可生成。
```

### 阶段 7：成对 SNR sweep

实现：

```text
scripts/usrp/04_prepare_test_payloads.py
scripts/usrp/05_run_paired_snr_test.py
scripts/usrp/08_compute_metrics.py
scripts/usrp/09_plot_results.py
scripts/usrp/10_make_report.py
```

验收：

```text
每个 selected SNR 点执行：calib_pre → Opus/JSCM alternating → calib_post。
每个 SNR 点至少 3 次 repeat。
pre/post SNR drift > 1 dB 的 run 被标记 unstable。
输出 summary_by_snr.csv 和指标曲线。
```

---

## 21. 调试顺序

严格按此顺序：

```text
1. 检查双 USRP 连接：uhd_find_devices / ping。
2. 纯 Python 仿真：STFT → JSCM → AWGN → iSTFT。
3. Opus 纯 Python 编解码。
4. BLE-like frame 单元测试。
5. GRC loopback：Opus/GFSK。
6. GRC loopback：JSCM/FM。
7. calib_tx/rx 生成 SNR LUT。
8. Opus USRP 高 SNR 单点跑通。
9. JSCM USRP 高 SNR 单点跑通。
10. 成对 SNR sweep。
11. 指标汇总和报告生成。
```

不要跳过 loopback 和 calibration。否则 USRP 链路错误很难定位。

---

## 22. 关键工程约束

1. **GRC 不负责实验逻辑**：只做调制、USRP、解调和落盘。
2. **Python 负责所有实验调度**：SNR 标定、成对运行、解帧、指标。
3. **禁止 AGC**：RX gain 固定，AGC 会破坏 SNR sweep。
4. **JSCM payload 不做 hard CRC**：只对 header 做 CRC。
5. **JSCM float slot 必须占用 32 bit-times**：否则与 Opus byte-equivalent 帧时长不公平。
6. **每个 tx_gain 下必须成对跑 Opus/JSCM**：不要分开跑完两个系统。
7. **最终横坐标用 measured_snr_db**，不是 tx_gain。
8. **所有 run 必须保存 IQ 和 meta**，否则无法复现实验。
9. **TX/RX USRP IP 必须显式写入配置和日志**。
10. **实际发射必须遵守频谱法规**，优先同轴线 + 衰减器或屏蔽箱。

---

## 23. 最终验收目标

工程完成后，应能执行：

```bash
python scripts/usrp/00_generate_calib_pilot.py --config configs/radio/ble_like_usrp.yaml
python scripts/usrp/01_capture_noise.py --config configs/radio/ble_like_usrp.yaml
python scripts/usrp/02_run_calib_sweep.py --radio configs/radio/ble_like_usrp.yaml --sweep configs/radio/snr_sweep.yaml
python scripts/usrp/03_estimate_snr_lut.py --radio configs/radio/ble_like_usrp.yaml --sweep configs/radio/snr_sweep.yaml
python scripts/usrp/04_prepare_test_payloads.py --config configs/experiments/paired_opus_jscm_ble.yaml
python scripts/usrp/05_run_paired_snr_test.py --config configs/experiments/paired_opus_jscm_ble.yaml
python scripts/usrp/06_decode_opus_results.py --root outputs/usrp_paired_ble
python scripts/usrp/07_decode_jscm_results.py --root outputs/usrp_paired_ble
python scripts/usrp/08_compute_metrics.py --root outputs/usrp_paired_ble
python scripts/usrp/09_plot_results.py --root outputs/usrp_paired_ble
python scripts/usrp/10_make_report.py --root outputs/usrp_paired_ble
```

最终输出：

```text
outputs/usrp_paired_ble/report.md
outputs/usrp_paired_ble/metrics/per_utterance.csv
outputs/usrp_paired_ble/metrics/summary_by_snr.csv
outputs/usrp_paired_ble/figures/pesq_vs_snr.png
outputs/usrp_paired_ble/figures/stoi_vs_snr.png
outputs/usrp_paired_ble/figures/sisdr_vs_snr.png
outputs/usrp_paired_ble/figures/per_or_sync_vs_snr.png
outputs/usrp_paired_ble/figures/snr_drift.png
```

---

## 24. 结论

本工程采用：

```text
TX USRP: 192.168.10.1
RX USRP: 192.168.10.2
BLE-like LE 1M RF 参数
Opus + GFSK baseline
Conformer-JSCM + FM float payload
统一 BLE-like frame shell
成对、交替、重复的 SNR sweep
```

核心实验原则：

```text
同一个 tx_gain / tx_amp 下，Opus 和 JSCM 必须连续成对运行；
每个点前后校准 SNR；
最终用 measured_snr_db 作横坐标；
GRC 越简单越好，实验逻辑全部放在 Python。
```


---

# v3 Appendix A. 训练系统规格 Training Specification

本章用于指导 Codex 构建 **Conformer-JSCM 训练闭环**。训练系统必须先在纯 Python 仿真中稳定收敛，再进入 GRC loopback 和 USRP 实测。不要直接用未充分仿真的模型做空口实验。

## A.1 训练目标

训练目标是学习如下端到端映射：

```text
waveform
→ STFT real/imag feature
→ Conformer-JSCM encoder
→ power normalization
→ differentiable channel
→ Conformer-JSCM decoder
→ reconstructed STFT real/imag
→ iSTFT
→ reconstructed waveform
```

第一版训练只使用 AWGN 可微信道；第二版再加入 Rayleigh、CFO、幅度缩放、相位旋转和轻微符号时钟误差。

## A.2 训练入口和必须输出

Codex 必须实现：

```text
scripts/02_train_conformer_jscm.py
scripts/03_eval_jscm_sim.py
src/speech_jscm/channels/awgn.py
src/speech_jscm/losses/spectral_loss.py
src/speech_jscm/losses/sisdr_loss.py
src/speech_jscm/utils/checkpoint.py
src/speech_jscm/utils/logger.py
```

训练脚本必须支持：

```text
--config configs/train/train_conformer_jscm.yaml
--resume checkpoints/conformer_jscm_base/last.pt
--device cuda
--seed 1234
```

每次训练必须输出：

```text
checkpoints/conformer_jscm_base/last.pt
checkpoints/conformer_jscm_base/best.pt
outputs/train/conformer_jscm_base/train_log.csv
outputs/train/conformer_jscm_base/valid_log.csv
outputs/train/conformer_jscm_base/config_resolved.yaml
outputs/train/conformer_jscm_base/samples/epoch_xxxx_ref.wav
outputs/train/conformer_jscm_base/samples/epoch_xxxx_rec.wav
```

## A.3 推荐训练配置

`configs/train/train_conformer_jscm.yaml`：

```yaml
task: conformer_jscm_train

seed: 1234
device: cuda
amp: true

paths:
  train_manifest: data/manifests/train_clean_100.jsonl
  valid_manifest: data/manifests/dev_clean.jsonl
  checkpoint_dir: checkpoints/conformer_jscm_base
  output_dir: outputs/train/conformer_jscm_base

data:
  sample_rate: 16000
  segment_samples: 16384
  random_crop: true
  valid_crop: center
  num_workers: 4
  pin_memory: true

stft:
  n_fft: 512
  win_length: 400
  hop_length: 160
  center: false
  feature: real_imag

model:
  name: conformer_jscm
  d_model: 192
  num_heads: 4
  encoder_layers: 6
  decoder_layers: 8
  ffn_dim: 768
  conv_kernel_size: 31
  dropout: 0.1
  freq_bins: 257
  time_frames: 100
  complex_symbols_per_segment: 1568
  power_average: 1.0
  snr_conditioning: true

train_channel:
  type: awgn
  snr_db_min: -5
  snr_db_max: 20
  random_snr_per_batch: true
  random_snr_per_sample: false

eval_channel:
  type: awgn
  snr_db_list: [-5, 0, 5, 10, 15, 20]

loss:
  stft_l1_weight: 1.0
  mrstft_weight: 0.5
  sisdr_weight: 0.1
  waveform_l1_weight: 0.0

optimizer:
  name: adamw
  lr: 2.0e-4
  weight_decay: 1.0e-4
  betas: [0.9, 0.98]

scheduler:
  name: cosine
  warmup_steps: 5000
  min_lr: 1.0e-6

training:
  epochs: 100
  batch_size: 16
  grad_clip_norm: 1.0
  log_interval: 50
  valid_interval_epochs: 1
  save_sample_interval_epochs: 1
  save_last_every_epochs: 1
  save_best_metric: valid_loss
```

## A.4 损失函数

第一版使用：

```text
L = λ1 * L_stft_ri
  + λ2 * L_mrstft
  + λ3 * L_sisdr
```

其中：

```text
L_stft_ri: reconstructed real/imag STFT 与 target real/imag STFT 的 L1 loss
L_mrstft: multi-resolution STFT magnitude loss
L_sisdr: negative SI-SDR loss，作用于 waveform
```

建议实现细节：

```text
1. 训练主 loss 以 STFT real/imag L1 为主，保证重构精度。
2. MR-STFT loss 用于抑制频谱伪影。
3. SI-SDR loss 权重不要过高，否则可能破坏频谱稳定性。
4. 第一版不要加入 GAN 或 diffusion loss。
```

MR-STFT 推荐分辨率：

```yaml
mrstft:
  fft_sizes: [256, 512, 1024]
  hop_sizes: [64, 160, 256]
  win_lengths: [256, 400, 1024]
```

## A.5 可微信道层

`src/speech_jscm/channels/awgn.py` 必须提供：

```python
class AWGNChannel(nn.Module):
    def forward(self, x_complex, snr_db):
        """
        x_complex: complex tensor or real tensor [..., 2]
        snr_db: scalar or [B]
        return: noisy complex tensor, same shape
        """
```

功率归一化后，AWGN 噪声方差按以下逻辑计算：

```python
signal_power = mean(|x|^2)
snr_linear = 10 ** (snr_db / 10)
noise_power = signal_power / snr_linear
```

如果 complex 用 `[real, imag]` 表示，噪声每个实部/虚部维度的方差为：

```python
sigma2_per_real_dim = noise_power / 2
```

## A.6 训练验收标准

训练系统初步可用的最低验收：

```text
1. overfit 8 条训练语音时，loss 能明显下降；
2. 无信道或高 SNR = 30 dB 时，重构语音可听且无明显爆音；
3. AWGN eval 中，SNR 从 -5 到 20 dB 提升时，PESQ/STOI/SI-SDR 单调或近似单调提升；
4. best.pt 和 last.pt 可正常 resume；
5. valid sample wav 每个 epoch 能正常保存。
```

---

# v3 Appendix B. 纯 Python 仿真验收 Simulation Evaluation Specification

在连接 USRP 之前，必须完成三层仿真验收。任何一层失败，都不要进入下一层。

## B.1 Level 1：无信道 JSCM 自重构

命令：

```bash
python scripts/03_eval_jscm_sim.py \
  --config configs/train/train_conformer_jscm.yaml \
  --checkpoint checkpoints/conformer_jscm_base/best.pt \
  --channel none \
  --manifest data/manifests/test_clean_small.jsonl \
  --output-dir outputs/eval/jscm_no_channel
```

验收：

```text
1. 所有 wav 能成功重构；
2. 输出长度与原始长度一致，误差不超过 1 个 hop；
3. 没有 NaN/Inf；
4. 指标文件 summary.json 和 per_utterance.csv 正常生成。
```

## B.2 Level 2：AWGN 仿真

命令：

```bash
python scripts/03_eval_jscm_sim.py \
  --config configs/train/train_conformer_jscm.yaml \
  --checkpoint checkpoints/conformer_jscm_base/best.pt \
  --channel awgn \
  --snr-list -5 0 5 10 15 20 \
  --manifest data/manifests/test_clean_small.jsonl \
  --output-dir outputs/eval/jscm_awgn
```

输出：

```text
outputs/eval/jscm_awgn/metrics/per_utterance.csv
outputs/eval/jscm_awgn/metrics/summary_by_snr.csv
outputs/eval/jscm_awgn/wav/snr_XX/*.wav
```

## B.3 Level 3：Opus 纯 Python 基线

命令：

```bash
python scripts/04_eval_opus_sim.py \
  --manifest data/manifests/test_clean_small.jsonl \
  --bitrate-list 6000 9000 12000 16000 24000 \
  --frame-ms 20 \
  --output-dir outputs/eval/opus_sim
```

Opus baseline 必须支持两种 packet loss 处理：

```text
mode=plc: CRC fail 或丢包时调用 Opus packet loss concealment
mode=silence: CRC fail 或丢包时填充静音
```

第一版报告 `plc` 结果，`silence` 作为诊断结果保留。

## B.4 Level 4：GRC 软件 loopback

Opus/GFSK loopback：

```bash
python scripts/usrp/04_prepare_test_payloads.py --config configs/experiments/paired_opus_jscm_ble.yaml --system opus
python grc/generated/opus_gfsk_loopback.py --input-file outputs/prepared/opus/tx_bits.bin --output-file outputs/loopback/opus/rx_bits.bin
python scripts/usrp/06_decode_opus_results.py --root outputs/loopback/opus
```

JSCM/FM loopback：

```bash
python scripts/usrp/04_prepare_test_payloads.py --config configs/experiments/paired_opus_jscm_ble.yaml --system jscm
python grc/generated/jscm_fm_loopback.py --input-file outputs/prepared/jscm/tx_float_frames.f32 --output-file outputs/loopback/jscm/rx_float_frames.f32
python scripts/usrp/07_decode_jscm_results.py --root outputs/loopback/jscm
```

验收：

```text
Opus loopback: 高 SNR/无噪声下 BER 接近 0，CRC pass rate 接近 100%。
JSCM loopback: float payload MSE 很小，frame sync rate 接近 100%。
```

---

# v3 Appendix C. 双 USRP 同步、CFO/SCO 和接收补偿

两台 USRP 分别作为 TX/RX：

```text
TX USRP: addr=192.168.10.1
RX USRP: addr=192.168.10.2
```

若没有共享 10 MHz/PPS 参考，必须假设存在 CFO、采样时钟偏差和相位漂移。

## C.1 两种同步模式

### C.1.1 默认模式：internal clock + 软件同步

配置：

```yaml
sync:
  clock_source: internal
  time_source: internal
  cfo_estimation: preamble_pilot
  cfo_correction: software
  frame_timing: preamble_correlation
  save_sync_diagnostics: true
```

必须实现：

```text
1. 用 preamble/pilot 做 packet detection；
2. 用 pilot 估计 CFO；
3. 用复数旋转做 CFO correction；
4. 用相关峰确定 frame timing；
5. 每次 run 保存 estimated_cfo_hz、timing_offset_samples、pilot_corr_peak。
```

### C.1.2 可选模式：external 10 MHz/PPS

如果实验室有共享参考源，可配置：

```yaml
sync:
  clock_source: external
  time_source: external
```

但 Codex 第一版仍应保留软件 CFO correction，因为实际调试时仍可能有残余频偏。

## C.2 CFO 估计接口

`src/speech_jscm/radio/sync.py` 必须提供：

```python
def estimate_cfo_from_pilot(rx_iq: np.ndarray, tx_pilot: np.ndarray, samp_rate: float) -> float:
    """Return estimated CFO in Hz."""


def correct_cfo(rx_iq: np.ndarray, cfo_hz: float, samp_rate: float) -> np.ndarray:
    """Apply exp(-j*2*pi*cfo*t) correction."""


def find_frame_start(rx_seq: np.ndarray, preamble: np.ndarray) -> tuple[int, float]:
    """Return start index and normalized correlation peak."""
```

## C.3 推荐 CFO 估计方法

第一版可用 pilot 相位差估计：

```python
# rx_pilot roughly aligned with tx_pilot
z = rx_pilot * np.conj(tx_pilot)
phase = np.unwrap(np.angle(z))
slope = np.polyfit(np.arange(len(phase)), phase, 1)[0]
cfo_hz = slope * samp_rate / (2 * np.pi)
```

如果 pilot 很短，采用两段重复 pilot 更稳：

```text
pilot = PN repeated twice
```

利用两段相位差估计 CFO：

```python
phase_diff = angle(sum(conj(rx_first) * rx_second))
cfo_hz = phase_diff / (2*pi*repeat_len/samp_rate)
```

## C.4 每帧同步诊断字段

每个 run 的 `run_meta.json` 必须包含：

```json
{
  "estimated_cfo_hz": 0.0,
  "timing_offset_samples": 0,
  "pilot_corr_peak": 0.0,
  "frame_sync_rate": 0.0,
  "header_decode_rate": 0.0,
  "rx_iq_max_abs": 0.0,
  "rx_iq_rms": 0.0
}
```

## C.5 GRC 里的同步责任边界

第一版建议：

```text
GRC 只保存 rx_iq_before_demod.c64 和粗解调结果。
Python 做帧同步、CFO 校正、header decode、payload extract。
```

不要把复杂同步逻辑放进 GRC，除非 Python 离线版本已经充分验证。

---

# v3 Appendix D. Opus/JSCM 公平性与资源统计 Fairness Accounting

Opus 是数字 bitstream，JSCM 是连续值 payload。两者不能用同一个 “payload bytes” 直接比较。必须同时报告空口资源、源语音资源和系统指标。

## D.1 统一公平原则

每个 SNR 点必须保证：

```text
1. 同一 TX/RX USRP；
2. 同一 center_freq；
3. 同一 samp_rate；
4. 同一 RX gain 和 RX bandwidth；
5. 同一 tx_gain / tx_amp 设置；
6. 同一测试语音集合；
7. 同一帧 shell 和同步开销定义；
8. 同一 measured_snr_db 统计方法；
9. 每个 tx_gain 下 Opus/JSCM 成对、交替、重复运行。
```

## D.2 必须统计的资源字段

每个 run 输出 `resource_summary.json`：

```json
{
  "system": "opus_gfsk or jscm_fm",
  "source_duration_sec": 0.0,
  "num_radio_frames": 0,
  "rf_samp_rate": 4000000,
  "rf_samples_total": 0,
  "rf_duration_sec": 0.0,
  "center_freq_hz": 2440000000,
  "occupied_bw_est_hz": null,
  "tx_gain": 0,
  "tx_amp": 0.0,
  "rx_gain": 0,
  "measured_snr_db_mean": 0.0,
  "measured_snr_db_std": 0.0
}
```

Opus 额外字段：

```json
{
  "opus_bitrate": 12000,
  "opus_frame_ms": 20,
  "payload_bytes_total": 0,
  "crc_pass_rate": 0.0,
  "packet_erasure_rate": 0.0,
  "plc_mode": "plc"
}
```

JSCM 额外字段：

```json
{
  "jscm_rate_id": "mid",
  "complex_symbols_per_segment": 1568,
  "complex_symbols_per_sec": 0.0,
  "float_slots_total": 0,
  "frame_sync_rate": 0.0,
  "header_decode_rate": 0.0,
  "payload_clip_rate": 0.0
}
```

## D.3 帧时长公平

JSCM float32 payload 使用 **32 bit-time 等效占用**。也就是说：

```text
1 float32 slot 的空口时间 = 32 个 BLE-like bit 的空口时间
56 float32 slots = 1792 bit-times
28 complex symbols/frame = 56 float32 slots/frame
```

Opus payload：

```text
224 bytes = 1792 bits
```

因此在外层 payload 时长上：

```text
Opus 224-byte payload 与 JSCM 56-float32-slot payload 对齐。
```

注意：这只是 **airtime-equivalent accounting**，不代表 JSCM 的 float 被数字可靠传输。

## D.4 绘图横坐标

所有性能曲线横坐标必须使用：

```text
measured_snr_db_mean
```

而不是：

```text
tx_gain
```

如果 `SNR_post - SNR_pre` 超过阈值：

```yaml
snr_drift_tolerance_db: 1.0
```

该 run 必须标记：

```json
{"stable": false, "unstable_reason": "snr_drift"}
```

默认不纳入最终 summary，但保留在 raw results 中。

---

# v3 Appendix E. GRC Generated Python Runner 规范

Codex 不应把实验调度逻辑写进 GRC 生成文件。GRC 生成 Python 只作为 flowgraph 可执行文件，由 `grc_runner.py` 统一调用。

## E.1 必须实现的 runner

`src/speech_jscm/radio/grc_runner.py`：

```python
@dataclass
class GrcRunConfig:
    flowgraph: str
    input_file: str | None
    output_file: str | None
    center_freq: float
    samp_rate: float
    tx_gain: float | None = None
    rx_gain: float | None = None
    tx_amp: float | None = None
    num_samps: int | None = None
    run_seconds: float | None = None
    extra_args: dict | None = None


def run_flowgraph(cfg: GrcRunConfig, log_file: str) -> int:
    """Run a generated GRC Python flowgraph and save stdout/stderr."""


def run_rx_tx_pair(rx_cfg: GrcRunConfig, tx_cfg: GrcRunConfig, warmup_sec: float, tail_sec: float) -> dict:
    """Start RX, wait, run TX, wait tail, stop/collect RX."""
```

## E.2 RX 结束方式

第一版 RX flowgraph 必须使用以下二选一方式自动结束：

```text
1. Head block: --num-samps 控制总采样数；
2. run_seconds: flowgraph 内部定时 stop。
```

不建议第一版使用：

```text
无限流 + 手动 kill
```

原因：文件 sink 可能未 flush，run 结果不可复现。

## E.3 每个 GRC generated Python 的 CLI 参数

所有 generated Python 应尽量支持：

```text
--device-args
--center-freq
--samp-rate
--tx-gain
--rx-gain
--tx-amp
--rx-bw
--tx-bw
--input-file
--output-file
--num-samps
--run-seconds
```

TX 示例：

```bash
python grc/generated/opus_gfsk_tx.py \
  --device-args "addr=192.168.10.1" \
  --center-freq 2440e6 \
  --samp-rate 4e6 \
  --tx-gain 10 \
  --tx-amp 0.05 \
  --input-file outputs/prepared/opus/tx_bits.bin
```

RX 示例：

```bash
python grc/generated/opus_gfsk_rx.py \
  --device-args "addr=192.168.10.2" \
  --center-freq 2440e6 \
  --samp-rate 4e6 \
  --rx-gain 20 \
  --num-samps 12000000 \
  --output-file outputs/run/opus/rx_bits.bin
```

---

# v3 Appendix F. 小测试集、依赖环境和安全边界

## F.1 小测试集生成规则

`data/manifests/test_clean_small.jsonl` 用于 USRP 初期调试，不应直接使用完整 `test-clean`。

生成规则：

```yaml
small_test:
  source_manifest: data/manifests/test_clean.jsonl
  output_manifest: data/manifests/test_clean_small.jsonl
  num_utterances: 20
  max_duration_sec_per_utt: 5.0
  speaker_balanced: true
  seed: 1234
```

要求：

```text
1. 尽量覆盖不同 speaker；
2. 单条语音不超过 5 秒；
3. manifest 中保留原 utt_id、speaker_id、duration、wav path；
4. USRP 调试默认只使用 test_clean_small。
```

## F.2 依赖版本建议

`environment.yml` 建议固定：

```yaml
name: speech-jscm
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.10
  - pytorch
  - torchaudio
  - pytorch-cuda=12.1
  - numpy
  - scipy
  - pandas
  - pyyaml
  - soundfile
  - matplotlib
  - tqdm
  - pip
  - pip:
      - pesq
      - pystoi
      - opuslib
```

系统依赖需要在 README 中列出：

```text
GNU Radio 3.10.x
UHD 4.x
libopus / opus-tools
ffmpeg, optional
```

Codex 应在 `scripts/check_env.py` 中检查：

```text
python version
torch/torchaudio version
CUDA available
GNU Radio executable exists
grc/generated files exist
uhd_find_devices 能否找到 192.168.10.1 和 192.168.10.2
```

## F.3 RF 安全边界

默认使用同轴线 + 30–60 dB 衰减器或屏蔽箱。若使用天线，必须确认实验许可和当地法规。

配置中写入：

```yaml
safety:
  prefer_cabled_test: true
  default_tx_gain: 0
  default_tx_amp: 0.03
  max_tx_gain: 30
  max_tx_baseband_abs: 0.7
  max_rx_iq_abs: 0.7
  require_user_confirm_for_antenna: true
```

脚本必须检查：

```text
tx_gain <= max_tx_gain
max(abs(tx_baseband)) < max_tx_baseband_abs
max(abs(rx_iq)) < max_rx_iq_abs
```

如果 RX IQ 接近饱和：

```text
1. 降低 RX gain；
2. 增加外部衰减；
3. 降低 TX gain 或 tx_amp；
4. 不允许把该 run 用于最终结果。
```

---

# v3 Appendix G. Codex MVP 构建路径

Codex 应按 MVP 顺序实现，不要一次性生成全工程后再调试。

## MVP-1：数据与 STFT/JSCM 前向

实现：

```text
src/speech_jscm/data/*
src/speech_jscm/features/stft.py
src/speech_jscm/models/conformer_jscm.py
scripts/01_make_manifest.py
scripts/02_train_conformer_jscm.py 的最小 forward 版本
```

验收命令：

```bash
python scripts/01_make_manifest.py --root data/raw/LibriSpeech
python scripts/02_train_conformer_jscm.py --config configs/train/train_conformer_jscm.yaml --debug-overfit 8
```

验收标准：

```text
batch shape = [B, 2, 257, 100]
encoder output complex symbols shape = [B, 1568, 2]
decoder output shape = [B, 2, 257, 100]
loss 可以反向传播
```

## MVP-2：训练闭环和 AWGN 仿真

实现：

```text
AWGNChannel
losses
checkpoint
03_eval_jscm_sim.py
```

验收：

```bash
python scripts/02_train_conformer_jscm.py --config configs/train/train_conformer_jscm.yaml --debug-overfit 8
python scripts/03_eval_jscm_sim.py --checkpoint checkpoints/conformer_jscm_base/best.pt --channel awgn --snr-list 0 10 20
```

## MVP-3：Opus encode/decode 与 BLE-like frame

实现：

```text
src/speech_jscm/baselines/opus_codec.py
src/speech_jscm/radio/ble_frame.py
src/speech_jscm/radio/opus_frame.py
scripts/04_eval_opus_sim.py
```

验收：

```text
Opus encode/decode 能恢复 wav；
BLE-like frame pack/unpack 单元测试通过；
CRC pass/fail 行为正确。
```

## MVP-4：JSCM float frame pack/unpack

实现：

```text
src/speech_jscm/radio/jscm_float_frame.py
scripts/05_export_jscm_symbols.py
scripts/06_recover_jscm_symbols.py
```

验收：

```text
complex symbols → float slots → frame → deframe → complex symbols 误差为 0；
最后一帧 padding 能正确恢复；
header decode 能识别 stream_id/frame_idx/sample_offset。
```

## MVP-5：GRC loopback

实现：

```text
grc/opus_gfsk/opus_gfsk_loopback.grc
grc/jscm_fm/jscm_fm_loopback.grc
```

验收：

```text
Opus loopback BER 接近 0；
JSCM loopback float MSE 很小；
loopback 输出可以进入 decode 脚本。
```

## MVP-6：双 USRP 校准链路

实现：

```text
grc/common/calib_tx.grc
grc/common/calib_rx.grc
scripts/usrp/00_generate_calib_pilot.py
scripts/usrp/01_capture_noise.py
scripts/usrp/02_run_calib_sweep.py
scripts/usrp/03_estimate_snr_lut.py
```

验收：

```text
uhd_find_devices 找到 192.168.10.1 和 192.168.10.2；
snr_lut.csv 能生成；
measured_snr_db 随 tx_gain/tx_amp 增大而上升；
rx_iq 不饱和。
```

## MVP-7：Opus USRP 高 SNR 单点

实现：

```text
grc/opus_gfsk/opus_gfsk_tx.grc
grc/opus_gfsk/opus_gfsk_rx.grc
scripts/usrp/06_decode_opus_results.py
```

验收：

```text
高 SNR 下 CRC pass rate 高；
Opus reconstructed wav 可生成；
PESQ/STOI/SI-SDR 可计算。
```

## MVP-8：JSCM USRP 高 SNR 单点

实现：

```text
grc/jscm_fm/jscm_fm_tx.grc
grc/jscm_fm/jscm_fm_rx.grc
scripts/usrp/07_decode_jscm_results.py
```

验收：

```text
高 SNR 下 frame sync rate 高；
header decode rate 高；
JSCM reconstructed wav 可生成；
保存 estimated_cfo_hz 和 pilot_corr_peak。
```

## MVP-9：成对 SNR sweep 与报告

实现：

```text
scripts/usrp/04_prepare_test_payloads.py
scripts/usrp/05_run_paired_snr_test.py
scripts/usrp/08_compute_metrics.py
scripts/usrp/09_plot_results.py
scripts/usrp/10_make_report.py
```

验收：

```text
每个 selected SNR 点执行 calib_pre → alternating Opus/JSCM → calib_post；
每个 SNR 点至少 3 次 repeat；
unstable run 自动标记；
summary_by_snr.csv 和 report.md 正常生成。
```

---

# v3 Appendix H. 最终交付清单

Codex 完成工程后，仓库至少应包含：

```text
1. 可训练 Conformer-JSCM 的 Python 训练系统；
2. 可运行 AWGN 仿真评估的 JSCM 脚本；
3. 可运行 Opus baseline 的纯 Python 仿真脚本；
4. BLE-like Opus byte frame 和 JSCM float frame 实现；
5. GRC loopback、calib、Opus TX/RX、JSCM TX/RX flowgraph；
6. 双 USRP SNR LUT 标定脚本；
7. Opus/JSCM 成对、交替、重复的 SNR sweep 脚本；
8. 统一指标计算、绘图和 Markdown 报告生成脚本；
9. 单元测试覆盖 STFT、power norm、AWGN、frame pack/unpack、SNR estimation、CFO correction；
10. README 中给出从 MVP-1 到 MVP-9 的命令序列。
```

最终用户应能按以下顺序运行：

```bash
# 1. 环境和 USRP 检查
python scripts/check_env.py

# 2. 数据 manifest
python scripts/01_make_manifest.py --root data/raw/LibriSpeech

# 3. 训练 JSCM
python scripts/02_train_conformer_jscm.py --config configs/train/train_conformer_jscm.yaml

# 4. 仿真评估
python scripts/03_eval_jscm_sim.py --checkpoint checkpoints/conformer_jscm_base/best.pt --channel awgn --snr-list -5 0 5 10 15 20
python scripts/04_eval_opus_sim.py --manifest data/manifests/test_clean_small.jsonl --bitrate-list 6000 9000 12000 16000 24000

# 5. USRP SNR 标定
python scripts/usrp/00_generate_calib_pilot.py --config configs/radio/ble_like_usrp.yaml
python scripts/usrp/01_capture_noise.py --config configs/radio/ble_like_usrp.yaml
python scripts/usrp/02_run_calib_sweep.py --radio configs/radio/ble_like_usrp.yaml --sweep configs/radio/snr_sweep.yaml
python scripts/usrp/03_estimate_snr_lut.py --radio configs/radio/ble_like_usrp.yaml --sweep configs/radio/snr_sweep.yaml

# 6. 准备 payload 并成对测试
python scripts/usrp/04_prepare_test_payloads.py --config configs/experiments/paired_opus_jscm_ble.yaml
python scripts/usrp/05_run_paired_snr_test.py --config configs/experiments/paired_opus_jscm_ble.yaml

# 7. 解码、指标、绘图和报告
python scripts/usrp/06_decode_opus_results.py --root outputs/usrp_paired_ble
python scripts/usrp/07_decode_jscm_results.py --root outputs/usrp_paired_ble
python scripts/usrp/08_compute_metrics.py --root outputs/usrp_paired_ble
python scripts/usrp/09_plot_results.py --root outputs/usrp_paired_ble
python scripts/usrp/10_make_report.py --root outputs/usrp_paired_ble
```

v3 的核心约束：

```text
先训练和仿真，再 GRC loopback，再 USRP 单点，最后成对 sweep。
不要跳过 SNR calibration。
不要在 RX 链路使用 AGC。
不要用 tx_gain 代替 measured_snr_db。
不要把 JSCM payload 当作数字 CRC 包处理。
不要把 Opus 和 JSCM 分开跑完整 SNR sweep。
```
