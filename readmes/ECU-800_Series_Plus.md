# ME Engineering Specs: ECU-800 Series Addendum

## Model Variant: ECU-850b (AI Enhanced)

This document outlines the specifications for the **ECU-850b**, an enhanced variant of the baseline ECU-850. It is part of the **ECU-800 Series** and is designed for edge AI workloads.

## Key Differentiators from ECU-850

The ECU-850b includes all features of the base ECU-850, with the following key upgrades:

- **Dedicated AI Accelerator:** A Neural Processing Unit (NPU) capable of 5 TOPS.
- **Increased Memory:** The RAM is upgraded to **4 GB LPDDR4** to support larger models.
- **Higher Clock Speed:** The Cortex-A53 cores are clocked at 1.5 GHz.

## Full Technical Specifications: ECU-850b

| Feature               | Specification                                 |
| --------------------- | --------------------------------------------- |
| **Processor**         | Dual-core ARM Cortex-A53 @ 1.5 GHz            |
| **NPU**               | 5 TOPS AI Accelerator                         |
| **Memory (RAM)**      | **4 GB** LPDDR4                               |
| **Storage**           | 32 GB eMMC                                    |
| **CAN Interface**     | Dual Channel, CAN FD up to 2 Mbps per channel |
| **Ethernet**          | 1x 100BASE-T1                                 |
| **Power Consumption** | Idle: 550mA, Under Load: 1.7A                 |
| **Operating Temp.**   | -40°C to +105°C                               |

## Example: NPU Configuration Snippet
To enable the NPU, use the following driver command:
```bash
me-driver-ctl --enable-npu --mode=performance
```