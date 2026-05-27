# ME Corporation - Internal Engineering Document

## Product Manual: ECU-700 Series (Legacy)

**Document ID:** DOC-700-REV1.2

**1. Introduction**
The ECU-700 Series represents a line of robust and reliable Electronic Control Units designed for core automotive functions. This document covers the specifications for the flagship model, the **ECU-750**. This series is now in legacy support and is succeeded by the ECU-800 series.

**2. Technical Specifications: ECU-750**
The ECU-750 is built for efficiency and durability in harsh environments.

| Feature                 | Specification                          |
| ----------------------- | -------------------------------------- |
| **Processor**           | 32-bit Cortex-M4 @ 120 MHz             |
| **Memory (RAM)**        | 512 KB SRAM                            |
| **Storage**             | 2 MB Internal Flash                    |
**CAN Interface**       | Single Channel, CAN FD compatible up to **1 Mbps** |
| **Operating Voltage**   | 9V - 16V                               |
| **Power Consumption**   | Typical: 150mA, Peak: 500mA            |
| **Operating Temperature** | -40°C to **+85°C**                     |
| **Connectors**          | 1x Main Automotive Connector, 1x JTAG  |

**3. Diagnostics**
The ECU-750 provides basic diagnostic trouble codes (DTCs) via the CAN bus interface. Refer to document DOC-700-DIAG for a full list of codes. Over-the-Air (OTA) updates are not supported on this hardware platform.

**4. Safety and Compliance**
The ECU-750 is certified for ISO 26262 ASIL-B.
