# QA Evaluation Report

## Summary

- Total questions: 10
- Passed: 9
- Failed: 1
- Pass rate: 90.00%
- Average keyword score: 71.38%
- Average latency: 11.23s

## Dataset Coverage

- Comparative - Cross Series: 1
- Comparative - Same Series: 1
- Configuration/Usage: 1
- Feature Availability: 1
- Operating Environment: 1
- Single Source - ECU-700: 1
- Single Source - ECU-800: 1
- Single Source - ECU-800 Enhanced: 1
- Storage Comparison: 1
- Technical Specification: 1

## Per-question Results

| ID | Status | Score | Category | Latency | Question |
|---:|---|---:|---|---:|---|
| 1 | PASS | 75% | Single Source - ECU-700 | 8.62s | What is the maximum operating temperature for the ECU-750? |
| 2 | PASS | 100% | Single Source - ECU-800 | 8.60s | How much RAM does the ECU-850 have? |
| 3 | PASS | 62% | Single Source - ECU-800 Enhanced | 10.96s | What are the AI capabilities of the ECU-850b? |
| 4 | PASS | 77% | Comparative - Same Series | 12.29s | What are the differences between ECU-850 and ECU-850b? |
| 5 | PASS | 67% | Comparative - Cross Series | 10.90s | Compare the CAN bus capabilities of ECU-750 and ECU-850. |
| 6 | PASS | 50% | Technical Specification | 7.75s | What is the power consumption of the ECU-850b under load? |
| 7 | PASS | 81% | Feature Availability | 17.04s | Which ECU models support Over-the-Air (OTA) updates? |
| 8 | FAIL | 35% | Storage Comparison | 15.30s | How does the storage capacity compare across all ECU models? |
| 9 | PASS | 67% | Operating Environment | 13.11s | Which ECU can operate in the harshest temperature conditions? |
| 10 | PASS | 100% | Configuration/Usage | 7.78s | How do you enable the NPU on the ECU-850b? |

## Detailed Results

### Q1: PASS

**Category:** Single Source - ECU-700

**Question:** What is the maximum operating temperature for the ECU-750?

**Expected answer:**

The maximum operating temperature for the ECU-750 is +85°C. The ECU-750 operates in a temperature range of -40°C to +85°C.

**Actual answer:**

The maximum operating temperature for the ECU-750 is **+85°C** as specified in the operating temperature range of -40°C to +85°C (source: ECU-700_Series_Manual.md).

**Score:** 75.00%

**Latency:** 8.62s

**Confidence:** 0.95

**Needs review:** False

**Sources:**

- ECU-700_Series_Manual.md

**Evaluation reason:**

Matched 6/8 expected keywords. Matched: maximum, operating, temperature, ecu-750, range, -40°c

### Q2: PASS

**Category:** Single Source - ECU-800

**Question:** How much RAM does the ECU-850 have?

**Expected answer:**

The ECU-850 has 2 GB of LPDDR4 RAM.

**Actual answer:**

The ECU-850 has **2 GB LPDDR4** RAM. (Source: ECU-800_Series_Base.md)

**Score:** 100.00%

**Latency:** 8.60s

**Confidence:** 0.95

**Needs review:** False

**Sources:**

- ECU-800_Series_Base.md

**Evaluation reason:**

Matched 4/4 expected keywords. Matched: ecu-850, 2, lpddr4, ram.

### Q3: PASS

**Category:** Single Source - ECU-800 Enhanced

**Question:** What are the AI capabilities of the ECU-850b?

**Expected answer:**

The ECU-850b features a dedicated Neural Processing Unit (NPU) capable of 5 TOPS (Tera Operations Per Second) for AI acceleration. This makes it suitable for edge AI workloads.

**Actual answer:**

According to the retrieved documentation (source: **ECU-800_Series_Plus.md**), the ECU-850b is an AI-enhanced variant designed for edge AI workloads. Its AI capabilities include:
- A dedicated Neural Processing Unit (NPU) capable of **5 TOPS**.
- Upgraded **4 GB LPDDR4 RAM** to support larger models.
- **Dual-core ARM Cortex-A53 @ 1.5 GHz** (higher clock speed than the base ECU-850).
- A driver command to enable the NPU in performance mode (`me-driver-ctl --enable-npu --mode=performance`).

**Score:** 62.50%

**Latency:** 10.96s

**Confidence:** 0.95

**Needs review:** False

**Sources:**

- ECU-800_Series_Plus.md

**Evaluation reason:**

Matched 10/16 expected keywords. Matched: ecu-850b, dedicated, neural, processing, unit, npu, 5, tops, edge, workloads.

### Q4: PASS

**Category:** Comparative - Same Series

**Question:** What are the differences between ECU-850 and ECU-850b?

**Expected answer:**

The ECU-850b is an enhanced variant of the ECU-850 with three key upgrades: Dedicated AI Accelerator (NPU capable of 5 TOPS), Increased Memory (4 GB LPDDR4 RAM vs 2 GB), Higher Clock Speed (Cortex-A53 cores at 1.5 GHz vs 1.2 GHz).

**Actual answer:**

Based on the retrieved evidence, the differences between the ECU-850 and ECU-850b are as follows:

- **Dedicated AI Accelerator:** The ECU-850b adds a Neural Processing Unit (NPU) capable of 5 TOPS, which the base ECU-850 does not have.
- **Increased Memory:** The ECU-850b has upgraded RAM to **4 GB LPDDR4** (from the base ECU-850's unspecified RAM).
- **Higher Clock Speed:** The Cortex-A53 cores in the ECU-850b are clocked at **1.5 GHz**, which is higher than the base ECU-850's clock speed.

Sources:  
- ECU-800_Series_Base.md (covers ECU-850 as baseline)  
- ECU-800_Series_Plus.md (lists the key differentiators in section "Key Differentiators from ECU-850")

**Score:** 76.92%

**Latency:** 12.29s

**Confidence:** 0.95

**Needs review:** False

**Sources:**

- ECU-800_Series_Base.md
- ECU-800_Series_Plus.md

**Evaluation reason:**

Matched 20/26 expected keywords. Matched: ecu-850b, ecu-850, key, dedicated, accelerator, npu, 5, tops, increased, memory, 4, lpddr4

### Q5: PASS

**Category:** Comparative - Cross Series

**Question:** Compare the CAN bus capabilities of ECU-750 and ECU-850.

**Expected answer:**

The ECU-750 (700 series) has a single channel CAN FD interface with speeds up to 1 Mbps, while the ECU-850 (800 series) features a dual channel CAN FD interface with speeds up to 2 Mbps per channel. The ECU-850 offers significantly better CAN bus performance and redundancy.

**Actual answer:**

Based on the retrieved documentation, the ECU-750 and ECU-850 have different CAN bus capabilities:

- **ECU-750**: Single Channel, CAN FD compatible up to **1 Mbps** (source: ECU-700_Series_Manual.md).
- **ECU-850**: Dual Channel, CAN FD up to **2 Mbps** per channel (source: ECU-800_Series_Base.md).

The ECU-850 therefore offers two CAN channels and a higher data rate per channel compared to the single-channel, lower-speed ECU-750.

**Score:** 66.67%

**Latency:** 10.90s

**Confidence:** 0.95

**Needs review:** False

**Sources:**

- ECU-700_Series_Manual.md
- ECU-800_Series_Base.md

**Evaluation reason:**

Matched 14/21 expected keywords. Matched: ecu-750, 700, series, single, channel, can, 1, mbps, ecu-850, 800, dual, 2

### Q6: PASS

**Category:** Technical Specification

**Question:** What is the power consumption of the ECU-850b under load?

**Expected answer:**

The ECU-850b consumes 1.7A under load and 550mA when idle.

**Actual answer:**

The power consumption of the ECU-850b under load is **1.7A** (source: ECU-800_Series_Plus.md).

**Score:** 50.00%

**Latency:** 7.75s

**Confidence:** 0.95

**Needs review:** False

**Sources:**

- ECU-800_Series_Plus.md

**Evaluation reason:**

Matched 4/8 expected keywords. Matched: ecu-850b, 1.7a, under, load

### Q7: PASS

**Category:** Feature Availability

**Question:** Which ECU models support Over-the-Air (OTA) updates?

**Expected answer:**

OTA updates are supported by the ECU-800 Series (ECU-850 and ECU-850b). The ECU-700 Series (ECU-750) does not support OTA updates, as explicitly stated in its documentation.

**Actual answer:**

Based on the retrieved documentation, the **ECU-850** model (from the ECU-800 Series) explicitly supports Over‑the‑Air (OTA) updates, as listed in the "Key Features" section of `ECU-800_Series_Base.md`. The **ECU-750** (ECU-700 Series) does not support OTA updates, as stated in `ECU-700_Series_Manual.md`. For the **ECU-850b** variant, the documentation does not contain enough information to confirm OTA support. No other models (e.g., other ECU-700 or ECU-800 series models) are mentioned in the evidence.
source: ECU-700_Series_Manual.md, ECU-800_Series_Base.md, ECU-800_Series_Plus.md

**Score:** 81.25%

**Latency:** 17.04s

**Confidence:** 0.45

**Needs review:** True

**Sources:**

- ECU-700_Series_Manual.md
- ECU-800_Series_Base.md
- ECU-800_Series_Plus.md

**Evaluation reason:**

Matched 13/16 expected keywords. Matched: ota, updates, ecu-800, series, ecu-850, ecu-850b, ecu-700, ecu-750, does, not, support, explicitly

### Q8: FAIL

**Category:** Storage Comparison

**Question:** How does the storage capacity compare across all ECU models?

**Expected answer:**

Storage capacity varies significantly across the series: ECU-750 (700 series): 2 MB Internal Flash; ECU-850 (800 series): 16 GB eMMC; ECU-850b (800 series): 32 GB eMMC. The 800 series offers substantially more storage than the legacy 700 series.

**Actual answer:**

Based on the retrieved evidence, the documentation does not contain enough information to compare storage capacity across all ECU models. The provided documents only cover the ECU-850 and ECU-850b models from the ECU-800 series, with no evidence for the ECU-750 or other models. For the ECU-850b, the evidence notes an upgrade to **4 GB LPDDR4** RAM (ECU-800_Series_Plus.md), but the base ECU-850's storage capacity is not specified in the retrieved excerpts. Thus, a complete comparison is not possible.
source: ECU-800_Series_Base.md, ECU-800_Series_Plus.md

**Score:** 34.78%

**Latency:** 15.30s

**Confidence:** 0.45

**Needs review:** True

**Sources:**

- ECU-800_Series_Base.md
- ECU-800_Series_Plus.md

**Evaluation reason:**

Matched 8/23 expected keywords. Matched: storage, capacity, across, series, ecu-750, ecu-850, 800, ecu-850b

### Q9: PASS

**Category:** Operating Environment

**Question:** Which ECU can operate in the harshest temperature conditions?

**Expected answer:**

The ECU-850 and ECU-850b (800 series) can operate in the harshest temperature conditions, with an operating range of -40°C to +105°C. The ECU-750 (700 series) has a more limited range of -40°C to +85°C, making it 20°C less tolerant at the high end.

**Actual answer:**

Based on the retrieved evidence, both the **ECU-850** and **ECU-850b** can operate in the harshest temperature conditions, with a specified operating range of **-40°C to +105°C** (source: `ECU-800_Series_Base.md` and `ECU-800_Series_Plus.md`). The **ECU-750** has a narrower range of **-40°C to +85°C** (source: `ECU-700_Series_Manual.md`). Therefore, the ECU-850 and ECU-850b are equally capable of enduring more extreme temperatures than the ECU-750.

**Score:** 66.67%

**Latency:** 13.11s

**Confidence:** 0.95

**Needs review:** False

**Sources:**

- ECU-700_Series_Manual.md
- ECU-800_Series_Base.md
- ECU-800_Series_Plus.md

**Evaluation reason:**

Matched 16/24 expected keywords. Matched: ecu-850, ecu-850b, 800, series, can, operate, harshest, temperature, conditions, operating, range, -40°c

### Q10: PASS

**Category:** Configuration/Usage

**Question:** How do you enable the NPU on the ECU-850b?

**Expected answer:**

To enable the NPU on the ECU-850b, use the following driver command: me-driver-ctl --enable-npu --mode=performance

**Actual answer:**

To enable the NPU on the ECU-850b, use the following driver command:

```bash
me-driver-ctl --enable-npu --mode=performance
```

**Source:** ECU-800_Series_Plus.md (section "Example: NPU Configuration Snippet")

**Score:** 100.00%

**Latency:** 7.78s

**Confidence:** 0.95

**Needs review:** False

**Sources:**

- ECU-800_Series_Plus.md

**Evaluation reason:**

Matched 11/11 expected keywords. Matched: enable, npu, ecu-850b, use, following, driver, command, me-driver-ctl, --enable-npu, --mode, performance
