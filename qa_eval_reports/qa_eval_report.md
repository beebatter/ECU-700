# QA Evaluation Report

## Summary

- Total questions: 10
- Passed: 10
- Failed: 0
- Pass rate: 100.00%
- Average keyword score: 68.64%
- Average latency: 21.12s

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
| 1 | PASS | 75% | Single Source - ECU-700 | 18.20s | What is the maximum operating temperature for the ECU-750? |
| 2 | PASS | 75% | Single Source - ECU-800 | 15.43s | How much RAM does the ECU-850 have? |
| 3 | PASS | 62% | Single Source - ECU-800 Enhanced | 32.29s | What are the AI capabilities of the ECU-850b? |
| 4 | PASS | 73% | Comparative - Same Series | 24.58s | What are the differences between ECU-850 and ECU-850b? |
| 5 | PASS | 62% | Comparative - Cross Series | 23.33s | Compare the CAN bus capabilities of ECU-750 and ECU-850. |
| 6 | PASS | 50% | Technical Specification | 13.42s | What is the power consumption of the ECU-850b under load? |
| 7 | PASS | 75% | Feature Availability | 22.98s | Which ECU models support Over-the-Air (OTA) updates? |
| 8 | PASS | 70% | Storage Comparison | 20.28s | How does the storage capacity compare across all ECU models? |
| 9 | PASS | 62% | Operating Environment | 25.15s | Which ECU can operate in the harshest temperature conditions? |
| 10 | PASS | 82% | Configuration/Usage | 15.52s | How do you enable the NPU on the ECU-850b? |

## Detailed Results

### Q1: PASS

**Category:** Single Source - ECU-700

**Question:** What is the maximum operating temperature for the ECU-750?

**Expected answer:**

The maximum operating temperature for the ECU-750 is +85°C. The ECU-750 operates in a temperature range of -40°C to +85°C.

**Actual answer:**

The maximum operating temperature for the ECU-750 is **+85°C**, with an operating temperature range of **-40°C to +85°C**. This information is from the *ECU-700_Series_Manual.md*.
source: ECU-700_Series_Manual.md

**Score:** 75.00%

**Latency:** 18.20s

**Confidence:** 0.8500000000000001

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

The ECU-850 has **2 GB LPDDR4** RAM (source: ECU-800_Series_Base.md, section: ECU-850 Technical Specifications).

**Score:** 75.00%

**Latency:** 15.43s

**Confidence:** 0.85

**Needs review:** False

**Sources:**

- ECU-800_Series_Base.md

**Evaluation reason:**

Matched 3/4 expected keywords. Matched: ecu-850, 2, lpddr4

### Q3: PASS

**Category:** Single Source - ECU-800 Enhanced

**Question:** What are the AI capabilities of the ECU-850b?

**Expected answer:**

The ECU-850b features a dedicated Neural Processing Unit (NPU) capable of 5 TOPS (Tera Operations Per Second) for AI acceleration. This makes it suitable for edge AI workloads.

**Actual answer:**

Based on the retrieved documentation, the ECU-850b has a dedicated **Neural Processing Unit (NPU)** that is a **5 TOPS AI Accelerator** ([ECU-800_Series_Plus.md](ECU-800_Series_Plus.md)). This model is described as an "AI Enhanced" variant designed for edge AI workloads. Its key AI differentiator from the base ECU-850 is the dedicated AI accelerator. The documentation does not contain enough information about additional AI capabilities beyond the NPU.
source: ECU-800_Series_Plus.md

**Score:** 62.50%

**Latency:** 32.29s

**Confidence:** 0.35

**Needs review:** True

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

Based on the retrieved evidence, the key differences between the ECU-850 and ECU-850b are:

- **Dedicated AI Accelerator:** The ECU-850b includes a Neural Processing Unit (NPU) capable of 5 TOPS, whereas the base ECU-850's AI/ML acceleration is noted as a listed feature with a reference to the ECU-850b for an enhanced NPU (ECU-800_Series_Base.md, ECU-800_Series_Plus.md).
- **Memory:** The ECU-850b is upgraded to **4 GB LPDDR4** RAM to support larger models (ECU-800_Series_Plus.md).
- **Clock Speed:** The ARM Cortex-A53 cores in the ECU-850b are clocked at **1.5 GHz** (ECU-800_Series_Plus.md). Equivalent specifications for the base ECU-850 are not provided in the evidence.

All other features of the ECU-850 are included in the ECU-850b (ECU-800_Series_Plus.md). No additional numerical differences (e.g., power consumption, processor details for ECU-850) are available in the retrieved documentation.
source: ECU-800_Series_Base.md, ECU-800_Series_Plus.md

**Score:** 73.08%

**Latency:** 24.58s

**Confidence:** 0.8685278936324992

**Needs review:** False

**Sources:**

- ECU-800_Series_Base.md
- ECU-800_Series_Plus.md

**Evaluation reason:**

Matched 19/26 expected keywords. Matched: ecu-850b, enhanced, ecu-850, key, dedicated, accelerator, npu, 5, tops, memory, 4, lpddr4

### Q5: PASS

**Category:** Comparative - Cross Series

**Question:** Compare the CAN bus capabilities of ECU-750 and ECU-850.

**Expected answer:**

The ECU-750 (700 series) has a single channel CAN FD interface with speeds up to 1 Mbps, while the ECU-850 (800 series) features a dual channel CAN FD interface with speeds up to 2 Mbps per channel. The ECU-850 offers significantly better CAN bus performance and redundancy.

**Actual answer:**

Based on the retrieved documentation:

- **ECU-750**: Single Channel, CAN FD compatible up to 1 Mbps (source: ECU-700_Series_Manual.md)
- **ECU-850**: Dual Channel, CAN FD up to 2 Mbps per channel (source: ECU-800_Series_Base.md)

The ECU-850 offers two CAN FD channels with higher per-channel speed (2 Mbps) compared to the single channel on the ECU-750 (1 Mbps).

**Score:** 61.90%

**Latency:** 23.33s

**Confidence:** 0.85

**Needs review:** False

**Sources:**

- ECU-700_Series_Manual.md
- ECU-800_Series_Base.md

**Evaluation reason:**

Matched 13/21 expected keywords. Matched: ecu-750, 700, series, single, channel, can, 1, mbps, ecu-850, 800, dual, 2

### Q6: PASS

**Category:** Technical Specification

**Question:** What is the power consumption of the ECU-850b under load?

**Expected answer:**

The ECU-850b consumes 1.7A under load and 550mA when idle.

**Actual answer:**

The power consumption of the ECU-850b under load is **1.7A**. This is documented in the "Full Technical Specifications: ECU-850b" section of *ECU-800_Series_Plus.md*.
source: ECU-800_Series_Plus.md

**Score:** 50.00%

**Latency:** 13.42s

**Confidence:** 0.85

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

Based on the retrieved documentation, the **ECU-850** supports Over-the-Air (OTA) updates. The ECU-800 Series Base documentation lists "Over-the-Air (OTA) Update Capability" as a key feature for the ECU-850 model (sources: ECU-800_Series_Base.md).

The **ECU-750** does not support OTA updates; the ECU-700 Series Manual states: "Over-the-Air (OTA) updates are not supported on this hardware platform" (source: ECU-700_Series_Manual.md).

For the **ECU-850b**, the documentation does not contain enough information about OTA support.

**Score:** 75.00%

**Latency:** 22.98s

**Confidence:** 0.35

**Needs review:** True

**Sources:**

- ECU-700_Series_Manual.md
- ECU-800_Series_Base.md

**Evaluation reason:**

Matched 12/16 expected keywords. Matched: ota, updates, supported, ecu-800, series, ecu-850, ecu-850b, ecu-700, ecu-750, does, not, support

### Q8: PASS

**Category:** Storage Comparison

**Question:** How does the storage capacity compare across all ECU models?

**Expected answer:**

Storage capacity varies significantly across the series: ECU-750 (700 series): 2 MB Internal Flash; ECU-850 (800 series): 16 GB eMMC; ECU-850b (800 series): 32 GB eMMC. The 800 series offers substantially more storage than the legacy 700 series.

**Actual answer:**

Based on the retrieved evidence, storage capacity varies across ECU models as follows:

- **ECU-750**: 2 MB Internal Flash (source: ECU-700_Series_Manual.md)
- **ECU-850**: 16 GB eMMC (source: ECU-800_Series_Base.md)
- **ECU-850b**: 32 GB eMMC (source: ECU-800_Series_Plus.md)

No other ECU models are covered in the documentation, so a broader comparison is not possible.

**Score:** 69.57%

**Latency:** 20.28s

**Confidence:** 0.876564626634121

**Needs review:** False

**Sources:**

- ECU-700_Series_Manual.md
- ECU-800_Series_Base.md
- ECU-800_Series_Plus.md

**Evaluation reason:**

Matched 16/23 expected keywords. Matched: storage, capacity, varies, across, series, ecu-750, 700, 2, internal, flash, ecu-850, 800

### Q9: PASS

**Category:** Operating Environment

**Question:** Which ECU can operate in the harshest temperature conditions?

**Expected answer:**

The ECU-850 and ECU-850b (800 series) can operate in the harshest temperature conditions, with an operating range of -40°C to +105°C. The ECU-750 (700 series) has a more limited range of -40°C to +85°C, making it 20°C less tolerant at the high end.

**Actual answer:**

Based on the retrieved documentation, the ECU models that can operate in the harshest temperature conditions are the **ECU-850** and **ECU-850b**, both with an operating temperature range of **-40°C to +105°C**.

This compares to the ECU-750, which has a range of -40°C to +85°C.

**Sources:**
- ECU-850: ECU-800_Series_Base.md
- ECU-850b: ECU-800_Series_Plus.md
- ECU-750: ECU-700_Series_Manual.md

**Score:** 62.50%

**Latency:** 25.15s

**Confidence:** 0.9370350303649904

**Needs review:** False

**Sources:**

- ECU-700_Series_Manual.md
- ECU-800_Series_Base.md
- ECU-800_Series_Plus.md

**Evaluation reason:**

Matched 15/24 expected keywords. Matched: ecu-850, ecu-850b, 800, series, can, operate, harshest, temperature, conditions, operating, range, -40°c

### Q10: PASS

**Category:** Configuration/Usage

**Question:** How do you enable the NPU on the ECU-850b?

**Expected answer:**

To enable the NPU on the ECU-850b, use the following driver command: me-driver-ctl --enable-npu --mode=performance

**Actual answer:**

To enable the NPU on the ECU-850b, run the command:

```bash
me-driver-ctl --enable-npu --mode=performance
```

This is documented in the "Example: NPU Configuration Snippet" section of ECU-800_Series_Plus.md.
source: ECU-800_Series_Plus.md

**Score:** 81.82%

**Latency:** 15.52s

**Confidence:** 0.85

**Needs review:** False

**Sources:**

- ECU-800_Series_Plus.md

**Evaluation reason:**

Matched 9/11 expected keywords. Matched: enable, npu, ecu-850b, driver, command, me-driver-ctl, --enable-npu, --mode, performance
