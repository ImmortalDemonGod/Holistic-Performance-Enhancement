<!--  File: docs/6_scheduling/FocusPredictor_TechSpec_v1.1.md  -->

---

## **Real-Time Focus Predictor - Biosensor Stack Design & Implementation**

**Document ID:** `FOCUSPRED-TECH-SPEC-V1.1-GOLD`
**Revision:** `2025-05-23`
**Lead Maintainer:** `[Your Name/Team Name]`
**Status:** `Draft for Implementation`

---

### **Table of Contents**
1.  Introduction
    1.1. Purpose and Rationale for Real-Time Focus Prediction
    1.2. Choice of Biosignals (EEG, HRV, EDA)
    1.3. Target User and Application Context
2.  System Architecture
    2.1. High-Level Overview
    2.2. Key Components
3.  Hardware Design & Assembly (per sensor module)
    3.1. EEG-A: MindWave Mobile (Commercial, Used)
    3.2. EEG-B: ADS1292R DIY Rig (2-Channel, High-Resolution)
    3.3. HRV: Polar H10 Chest Strap (Commercial)
    3.4. EDA: Grove GSR DIY Rig (Modified, with ACC)
4.  Software Stack
    4.1. Data Acquisition, Synchronization, and Drivers
    4.2. Signal Processing Pipeline (Preprocessing & Feature Extraction)
    4.3. Sensor Fusion Engine (Machine Learning Model)
    4.4. Focus Score API
    4.5. Dependency Management and Versioning
    4.6. Error Handling and Logging Strategy
5.  Validation Protocol
    5.1. Hardware & Signal Quality Validation
    5.2. Feature Validation
    5.3. Fused Focus Score Validation
    5.4. Individual Calibration & Onboarding
6.  Safety & Ethical Considerations
    6.1. Electrical Safety
    6.2. Data Privacy, Security, and User Control
    6.3. User Burden and Comfort
    6.4. Algorithmic Bias and Fair Usage
7.  Budget & Timeline
    7.1. Hardware Budget
    7.2. Development & Implementation Timeline (MVP)
    7.3. Resource Assumptions & Timeline Caveats
8.  Future Considerations & Scalability
9.  Appendices (Placeholders)
    9.1. Appendix A: Circuit Diagram - EEG-B (ADS1292R DIY)
    9.2. Appendix B: Circuit Diagram - EDA Grove GSR Modification
    9.3. Appendix C: LSL Stream Configuration Details
    9.4. Appendix D: Detailed Feature List

---

### **1. Introduction**

**1.1. Purpose and Rationale for Real-Time Focus Prediction**
The primary objective of this system is to design, implement, and validate a non-invasive, multi-modal biosensor stack capable of providing a continuous, objective measure of an individual's cognitive focus or attentional state in real time. This real-time focus score is envisioned as a critical input for adaptive systems, with a primary application within the **Holistic Performance Enhancement (HPE) framework's task scheduler**.

By accurately understanding a user's momentary cognitive capacity, the integrated system aims to:
*   **Optimize Task Allocation:** Dynamically match the cognitive demands of tasks to the user's current focus level.
*   **Enhance Productivity & Deep Work:** Facilitate longer, uninterrupted periods of high-quality work by minimizing ill-timed interruptions or task switches.
*   **Prevent Cognitive Overload & Burnout:** Proactively suggest breaks or schedule less demanding activities when focus wanes or physiological strain is detected.
*   **Improve Task Completion Time (TTC) Estimation:** Incorporate real-time focus into ETA models for more accurate and dynamic project planning.
*   **Offer Personalized Insights:** Enable users to understand their individual work rhythms, peak performance times, and the impact of various lifestyle factors (sleep, exercise, stress) on their cognitive performance.
*   **Reduce Context-Switching Costs:** By better timing tasks and breaks, minimize the cognitive overhead associated with frequent task interruptions.

The existing literature and our prior research (see `FOCUSPRED-LITREVIEW-V1.0`) suggest that such adaptive, physiology-aware systems can yield significant improvements in task throughput (estimated 12-18%), reduce self-reported fatigue (estimated ~20%), and lead to more sustainable work patterns.

**1.2. Choice of Biosignals (EEG, HRV, EDA)**
A multi-modal approach combining Electroencephalography (EEG), Heart Rate Variability (HRV), and Electrodermal Activity (EDA) has been selected. This combination offers a holistic view of cognitive and physiological state, leveraging the unique strengths of each signal:

*   **EEG (Electroencephalography):**
    *   **Principle:** Directly measures the electrical activity of the brain via scalp electrodes.
    *   **Relevance to Focus:** Changes in EEG frequency band power (particularly Alpha, Theta, and Beta bands) and their ratios are well-established neural correlates of attention, mental workload, vigilance, and cognitive engagement. Frontal lobe activity is especially relevant for executive functions.
    *   **Contribution:** Provides high temporal resolution insights into central nervous system activity related to cognitive processing.

*   **HRV (Heart Rate Variability):**
    *   **Principle:** Quantifies the variation in time intervals between consecutive heartbeats, reflecting autonomic nervous system (ANS) regulation.
    *   **Relevance to Focus:** Lower HRV (especially metrics like RMSSD and HF power, indicative of reduced parasympathetic activity or increased sympathetic drive) is often associated with increased mental effort, stress, or fatigue. Higher resting HRV is generally linked to better self-regulation and cognitive readiness.
    *   **Contribution:** Offers a robust measure of overall physiological state, accumulated strain, and readiness for cognitive tasks.

*   **EDA (Electrodermal Activity / Galvanic Skin Response):**
    *   **Principle:** Measures changes in the electrical conductance of the skin, primarily due to eccrine sweat gland activity, which is innervated by the sympathetic nervous system.
    *   **Relevance to Focus:** Increases in tonic skin conductance level (SCL) and the frequency/amplitude of phasic skin conductance responses (SCRs) are sensitive to cognitive load, emotional arousal, and orienting responses. These can signal shifts in attention, mental effort, or the onset of fatigue-related stress.
    *   **Contribution:** Provides a sensitive measure of sympathetic arousal and emotional engagement, complementing EEG and HRV.

**Synergy:** Fusing these three modalities is anticipated to yield a more accurate, robust, and nuanced assessment of cognitive focus than any single signal alone. EEG provides direct neural correlates, HRV offers insight into systemic physiological regulation, and EDA captures arousal dynamics. This triangulation can help disambiguate states (e.g., differentiate stressed-but-focused from relaxed-and-unfocused) and improve resilience to artifacts in any single modality.

**1.3. Target User and Application Context**
The initial target user for this system is an individual knowledge worker, researcher, or developer (such as the primary user of the HPE framework) engaged in cognitively demanding tasks (e.g., programming, data analysis, writing, focused learning). The system is designed for use in typical office or home-office environments. While the sensor stack aims for minimal obtrusiveness, it is best suited for periods of dedicated desk-based work.

---

### **2. System Architecture**

**2.1. High-Level Overview**
The system employs a modular pipeline architecture, as depicted below. Raw data from each biosensor is acquired, time-stamped, and streamed. These streams are then synchronized, processed to extract relevant features, and finally fused by a machine learning model to produce a continuous, real-time focus score.

```mermaid
graph TD
    subgraph Sensor_Modules ["Sensor Modules (Wearable/DIY)"]
        A[EEG-A: MindWave Mobile BT<br>(1-ch Raw EEG @ 512Hz, eSense @ 1Hz)]
        B[EEG-B: ADS1292R DIY Rig<br>(2-ch Raw EEG @ 250Hz)]
        C[HRV: Polar H10 BLE<br>(RR-Intervals @ ~1Hz, Raw ECG @ 130Hz)]
        D[EDA: Grove GSR DIY Rig<br>(GSR @ 4Hz, ACC @ 50Hz)]
    end

    subgraph Data_Acquisition_Transport ["Data Acquisition & Transport Layer"]
        LSL[Lab Streaming Layer (LSL)<br>Timestamping & Network Streaming]
    end

    subgraph Central_Processing_Unit ["Central Processing Unit (Host PC/System)"]
        LSLInlets[LSL Inlets & Time Synchronization Node]
        Preprocessing[Signal Preprocessing & Artifact Handling<br>(Filtering, Cleaning per Modality)]
        FeatureExtraction[Feature Extraction Node<br>(EEG Bands, HRV Metrics, EDA Tonic/Phasic)]
        SensorFusion[Sensor Fusion Engine<br>(XGBoost Static Probabilities + Kalman Filter Temporal Smoothing)]
        FocusAPI[Focus Score API<br>(JSON Output: Score, Confidence, Status<br>via WebSocket & REST)]
    end

    subgraph Consuming_Applications ["Consuming Applications"]
        HPE_Scheduler[HPE Task Scheduler<br>(Adaptive Planning, ETA Prediction)]
        Dashboard[Real-Time Focus Dashboard<br>(Visualization, User Feedback)]
    end

    A --> LSL
    B --> LSL
    C --> LSL
    D --> LSL

    LSL --> LSLInlets
    LSLInlets --> Preprocessing
    Preprocessing --> FeatureExtraction
    FeatureExtraction --> SensorFusion
    SensorFusion --> FocusAPI

    FocusAPI --> HPE_Scheduler
    FocusAPI --> Dashboard
```

**2.2. Key Components**

*   **Sensor Modules:** The four distinct hardware units (MindWave, ADS1292R rig, Polar H10, Grove GSR rig) responsible for transducing physiological signals into electrical data.
*   **Lab Streaming Layer (LSL):** A crucial middleware that provides a unified platform for:
    *   High-precision timestamping of data samples at their source.
    *   Network streaming of multimodal data.
    *   Automatic discovery of data streams.
    *   Buffering and synchronization primitives for consuming applications.
*   **LSL Inlets & Time Synchronization Node:** A software component on the host system that subscribes to all LSL streams, aligns samples based on their LSL timestamps, and resamples/windows data for consistent input to the processing pipeline.
*   **Signal Preprocessing & Artifact Handling:** Modules dedicated to cleaning raw signals from each modality, including filtering (band-pass, notch), baseline correction, and techniques for detecting and mitigating artifacts (e.g., blinks in EEG, motion in EDA, ectopic beats in HRV).
*   **Feature Extraction Node:** Computes a set of validated physiological features from the preprocessed signals that are known correlates of cognitive focus, workload, or fatigue.
*   **Sensor Fusion Engine:** The core machine learning component. It ingests features from all modalities and employs a trained model (e.g., stacked XGBoost for instantaneous probability estimation + Kalman filter for temporal smoothing and state tracking) to output a continuous, unified focus score (e.g., scaled 0-100).
*   **Focus Score API:** A standardized software interface (WebSocket for real-time push, REST for polled requests) that makes the focus score, confidence levels, and sensor status available to consuming applications like the HPE scheduler or a visualization dashboard.

---

### **3. Hardware Design & Assembly (per sensor module)**

This section details the hardware specifications, bill of materials (BOM), assembly instructions, firmware outlines, and validation steps for each of the four sensor modules.

**3.1. EEG-A: MindWave Mobile (Commercial, Used Unit)**
*   **Purpose:** Serves as a readily available, single-channel frontal EEG source. Provides both raw EEG data and NeuroSky's proprietary eSense™ "Attention" and "Meditation" indices, useful for baseline comparisons and as a potential fallback.
*   **Specifications:**
    *   Sensor Type: Single dry electrode, typically placed at FP1 (forehead). Reference and ground electrodes are usually on the ear clip.
    *   Data Output:
        *   Raw EEG: Up to 512Hz sampling rate, typically 12-bit resolution.
        *   eSense™ Metrics: "Attention" and "Meditation" scores (0-100 scale), updated at 1Hz.
        *   EEG Power Bands: Processed power in Delta, Theta, Alpha (Low/High), Beta (Low/High), Gamma (Low/Mid) bands, updated at 1Hz.
    *   Connectivity: Bluetooth Classic (original MindWave Mobile) or Bluetooth Low Energy (MindWave Mobile 2).
    *   Target Cost: ~$75 (used market).
*   **Data Access Method:**
    *   Utilizes the NeuroSky ThinkGear Socket Protocol (TGSP).
    *   Recommended Python Libraries:
        *   **BrainFlow:** Provides direct support for MindWave Mobile (Board ID for Mobile 2 via BLE is `22`; older versions may require different IDs or serial port emulation). Handles connection, data parsing, and LSL streaming.
        *   `mindwave-py` or `PyThinkGear`: Community-developed libraries for interacting with TGSP.
        *   Custom scripts using `bluetooth-serial` (for BT Classic) or `bleak` (for BLE) to communicate with the ThinkGear Connector (TGC) or directly with the headset.
    *   Enabling Raw Data: A JSON command like `{"enableRawOutput": true, "format": "Json"}` must be sent to the TGSP stream after connection to receive the raw 512Hz EEG waveform.
*   **Rationale:** Low cost, minimal setup (non-invasive dry sensor), provides immediate access to both raw EEG and processed attention metrics, extensive use in consumer and some research applications.
*   **Assembly:** No physical assembly required beyond ensuring the device is charged and fits the user's head securely for good electrode contact.

**3.2. EEG-B: ADS1292R DIY Rig (2-Channel, High-Resolution)**
*   **Purpose:** Provides two channels of high-resolution (24-bit), research-quality frontal EEG data, enabling more sophisticated analyses like frontal asymmetry and more accurate spectral power estimation compared to EEG-A.
*   **Full Bill of Materials (BOM) (Estimated Total Cost ≈ $85):**
    *   1x ADS1292R Breakout Board (e.g., CJMCU-1292, ProtoCentral AdsEEG-02): ~$59
    *   1x ESP32-S3 DevKit-C-1 (or similar ESP32 with sufficient GPIOs, SPI, and ADC capabilities): ~$7
    *   1x 1000mAh LiPo Battery with JST-PH connector: ~$7
    *   1x MCP1700-3302E/TO LDO Voltage Regulator (3.3V, low noise) (Optional, for cleaner power to AFE): ~$1
    *   1x SPDT Slide Switch (for power): ~$1
    *   6x Gold Cup Ag/AgCl Electrodes (1m shielded lead, 1.5mm touch-proof DIN connector): ~$9 (e.g., 2x active for FP1/FP2, 2x linked reference for A1/A2 or Cz, 1x DRL, 1x spare).
    *   Electrode Paste (e.g., Ten20 Conductive Paste) & Skin Preparation Gel (e.g., NuPrep Abrasive Gel): ~$5
    *   3D Printed Headband/Enclosure Materials (PETG/PLA): ~$4
    *   Miscellaneous: Dupont wires, heat shrink tubing, perfboard or small prototyping PCB, connectors: ~$2
*   **Circuit Diagram & Assembly Instructions:**
    *   **(Placeholder: Refer to Appendix A for detailed schematic and PCB layout if developed).**
    *   **Power Circuitry:** LiPo battery → Power Switch → (Optional MCP1700 LDO) → Common 3.3V power rail for ESP32 and ADS1292R. Ensure robust grounding and power supply decoupling capacitors for both ICs as per their datasheets.
    *   **SPI Interface (ADS1292R to ESP32):**
        *   `CS` (Chip Select): Connect to a configurable ESP32 GPIO pin (e.g., GPIO10).
        *   `SCLK` (Serial Clock): Connect to ESP32's SPI Clock pin (e.g., GPIO12 for VSPI).
        *   `DOUT` (Data Out / MISO): Connect to ESP32's SPI MISO pin (e.g., GPIO13).
        *   `DIN` (Data In / MOSI): Connect to ESP32's SPI MOSI pin (e.g., GPIO11).
        *   `DRDY` (Data Ready Interrupt): Connect to an interrupt-capable ESP32 GPIO pin (e.g., GPIO3).
        *   `START` (Start Conversion): Connect to an ESP32 GPIO pin (e.g., GPIO46).
        *   `PWDN/RESET`: Tie according to ADS1292R datasheet for normal operation (usually pulled high or controlled by ESP32 GPIO).
    *   **Electrode Connections (ADS1292R):**
        *   Channel 1 (e.g., FP1): Electrode → IN1P. Reference electrode (e.g., linked A1+A2 or Cz) → IN1N.
        *   Channel 2 (e.g., FP2): Electrode → IN2P. Reference electrode (can share with Ch1) → IN2N.
        *   DRL (Driven Right Leg): DRLOUT pin → DRL electrode (e.g., contralateral mastoid or forehead CbZ).
    *   **Assembly Notes:**
        *   Mount components on perfboard or a custom-designed PCB for better noise performance.
        *   Use shielded cables for electrode leads if possible, or twist pairs.
        *   Enclose the electronics in a 3D printed case. The headband should allow secure and comfortable placement of electrodes according to the 10-20 system (FP1, FP2, reference, DRL).
*   **Firmware (ESP32 Sketch Outline - Arduino Core with LSL_ESP32 Library):**
    ```cpp
    #include "ADS1292R.h"       // Assuming a suitable ADS1292R driver library
    #include <WiFi.h>          // Or BluetoothSerial.h for BLE
    #include "lsl_esp32.h"     // Lab Streaming Layer for ESP32

    // Define SPI pins and other control pins
    #define ADS_CS_PIN    10
    #define ADS_DRDY_PIN  3
    #define ADS_START_PIN 46
    // #define ADS_PWDN_PIN  XX // Optional reset/power-down control

    ADS1292R ads_sensor(SPI, ADS_CS_PIN, ADS_START_PIN, ADS_DRDY_PIN); // SPI bus, CS, START, DRDY
    LSLOutlet lsl_outlet;

    // ADS1292R V_REF depends on configuration (internal 2.42V or 4.033V, or external)
    const float V_REF_MICROVOLTS = 2420000.0f; // Example: 2.42V internal reference
    const float PGA_GAIN = 6.0f;              // Example: Gain of 6x
    const float ADC_CONVERSION_FACTOR = V_REF_MICROVOLTS / (8388607.0f * PGA_GAIN); // (2^23 - 1) for 24-bit signed

    void setup() {
        Serial.begin(115200);
        // Initialize WiFi or BLE for LSL streaming
        // ... (WiFi.begin(ssid, password); or BLEDevice::init("ESP32_EEG"); etc.)

        ads_sensor.begin(); // Initialize ADS1292R
        ads_sensor.setSampleRate(ADS1292R_250SPS);
        ads_sensor.setGain(ADS1292R_GAIN_6); // Set PGA gain (e.g., 6x)
        ads_sensor.enableInternalReference();
        ads_sensor.enableDRL();
        ads_sensor.enableLeadOffDetection(ADS1292R_LEADOFF_CURRENT_6NA, ADS1292R_LEADOFF_COMP_TH_95); // Optional
        ads_sensor.startDataConversionContinuous(); // Or use START pin for single conversions

        // LSL Stream Info: name, type, channel_count, nominal_srate, channel_format, source_id
        lsl_outlet.begin("DIY_EEG_ADS1292R", "EEG", 2, 250.0, lsl_channel_format_t::cft_float32, "ads1292r_esp32_s3_01");
        // Add channel labels if library supports or manually in consumer
        // Example: char* labels[] = {"FP1", "FP2"}; lsl_outlet.set_channel_labels(labels);
    }

    void loop() {
        if (ads_sensor.isDataAvailableDRDY()) { // Check DRDY pin or interrupt flag
            int32_t raw_data_ch1 = ads_sensor.readDataChannel1();
            int32_t raw_data_ch2 = ads_sensor.readDataChannel2();

            float sample_uv[2];
            sample_uv[0] = static_cast<float>(raw_data_ch1) * ADC_CONVERSION_FACTOR;
            sample_uv[1] = static_cast<float>(raw_data_ch2) * ADC_CONVERSION_FACTOR;

            lsl_outlet.push_sample(sample_uv);
        }
        // Handle WiFi/BLE connection maintenance
    }
    ```
*   **Validation Steps (Critical for DIY):**
    1.  **Noise Floor Test (Shorted Inputs):** Connect IN1P to IN1N and IN2P to IN2N (with appropriate bias resistors if required by AFE). Stream data. The RMS noise should be < 5 µV, ideally < 2 µV, with a flat Power Spectral Density (PSD).
    2.  **Alpha Modulation (Eyes-Open/Eyes-Closed):** Place electrodes (e.g., FP1-Ref, FP2-Ref, or O1/O2-Ref for stronger alpha). Record data during alternating 10-30 second blocks of eyes open and eyes closed. The PSD of the eyes-closed segments should show a clear peak in the alpha band (8-12 Hz), typically ≥3-5 dB (or 2-3x power) higher than the eyes-open segments.
    3.  **Blink Artifact Test:** With frontal electrodes, deliberate eye blinks should produce characteristic large-amplitude (~100-300 µV), sharp, positive-going (or negative, depending on reference) deflections.
    4.  **Jaw Clench Artifact Test:** Clenching the jaw should produce high-frequency (>30 Hz) electromyographic (EMG) artifacts, confirming good electrode coupling and system responsiveness to muscle activity.

**3.3. HRV: Polar H10 Chest Strap (Commercial)**
*   **Purpose:** Provides ECG-accurate R-R intervals for robust Heart Rate Variability analysis and continuous heart rate monitoring.
*   **Specifications:**
    *   Sensor Type: Single-lead ECG via chest strap.
    *   Data Output:
        *   Heart Rate (HR): Typically updated every 1 second.
        *   R-R Intervals (RRIs) / Peak-to-Peak Intervals (PPIs): Streamed in packets, usually multiple intervals per packet, allowing for precise HRV calculation. Units are typically 1/1024th of a second.
        *   Raw ECG: Can be streamed at 130Hz via Polar Measurement Data (PMD) service (may require specific commands/SDK).
        *   Accelerometer (ACC): 3-axis, up to 200Hz via PMD service.
    *   Connectivity: Bluetooth Low Energy (BLE), ANT+.
    *   Target Cost: ~$99.
*   **Real-Time Python Access:**
    *   **`bleak` library (Recommended for direct BLE control):**
        *   Scan for and connect to the Polar H10 by its MAC address.
        *   Discover services and characteristics.
        *   Subscribe to notifications on the standard Heart Rate Service UUID (`0000180d-0000-1000-8000-00805f9b34fb`) and its Heart Rate Measurement Characteristic UUID (`00002a37-0000-1000-8000-00805f9b34fb`).
        *   The notification callback receives a `bytearray`. Parse it according to the Bluetooth GATT specification for Heart Rate Measurement (Flags byte, HR value, optional Energy Expended, optional RR-Intervals).
    *   **BrainFlow Library:**
        *   Supports Polar H10 (verify the exact Board ID, e.g., `BoardIds.POLAR_H10_BOARD` or a specific RR/ECG variant if available in current BrainFlow).
        *   Simplifies connection, data parsing, and LSL streaming. May handle enabling PMD for ECG/ACC if supported for the board ID.
    *   **`polarpy` Library (Linux-focused):**
        *   A higher-level Python wrapper for interacting with Polar devices, including enabling advanced data streams like ECG and ACC via the PMD service.
*   **Data Format (RR-Intervals):**
    *   The HR Measurement characteristic payload (if RRIs are present, indicated by a flag in the first byte) contains one or more 16-bit unsigned integers. Each integer represents an R-R interval in units of 1/1024 seconds. These must be converted to milliseconds or seconds for standard HRV analysis.
*   **Rationale:** Considered a gold-standard consumer device for HRV accuracy, comfortable for extended wear, widely validated against clinical ECG, and well-supported by open-source Python tools.

**3.4. EDA: Grove GSR DIY Rig (Modified, with ACC)**
*   **Purpose:** Measures changes in skin conductance (galvanic skin response) reflecting sympathetic nervous system arousal, with accelerometer data for motion artifact detection.
*   **Full Bill of Materials (BOM) (Estimated Total Cost ≈ $22 + ACC module if separate):**
    *   1x Grove GSR Sensor module (Seeed Studio SKU 101020052 or similar): ~$8
    *   1x ESP32-S3 DevKit-C-1: ~$7
    *   2x Reusable Ag/AgCl cup electrodes or disposable snap ECG/TENS electrodes with conductive gel: ~$2-5 (for finger or palmar placement).
    *   1x **3 kΩ precision resistor** (e.g., 1% tolerance, 0805 or through-hole): ~\$0.20 (This replaces a stock resistor in the Grove module's voltage divider to optimize for skin conductance range).
    *   Components for a **2-pole RC low-pass filter** (e.g., Sallen-Key topology with op-amp if needed, or passive RC; cutoff ~2Hz): Resistors (e.g., 2x 10kΩ) and capacitors (e.g., 2x 7.5uF non-polarized for ~2Hz, or adjust values) ~\$1.
    *   1x MPU-6050 (or similar I2C 3-axis accelerometer + gyroscope module, if not on ESP32 board): ~$3.
    *   1x Small LiPo battery (e.g., 150-500mAh) + charger circuit: ~$5.
    *   Wires, perfboard, small enclosure: ~$2.
*   **Circuit Modifications & Analysis for Conductance Calculation:**
    *   **(Placeholder: Refer to Appendix B for detailed schematic of Grove GSR modification and connection to ESP32).**
    *   The standard Grove GSR module typically uses a voltage divider where one resistor is fixed (`R_fixed_grove`, e.g., 10kΩ in some schematics) and the other is the skin resistance (`R_skin`). The output voltage (`V_out_grove`) is measured across `R_skin` (or `R_fixed_grove`).
    *   **Modification:** The "shunt mod" often refers to changing `R_fixed_grove` to a different value (e.g., our target `R_mod = 3kΩ`) to better suit the typical range of `R_skin` (100kΩ to 2MΩ). Let's assume `V_out_grove` is measured across `R_mod`.
        `V_out_grove = V_cc * R_mod / (R_mod + R_skin)`
    *   From this, `R_skin = R_mod * (V_cc / V_out_grove - 1)`.
    *   Skin Conductance `G_skin = 1 / R_skin = 1 / (R_mod * (V_cc / V_out_grove - 1))`.
    *   `V_out_grove` is read by the ESP32 ADC: `V_out_grove_mV = adc_raw * (ADC_V_REF_mV / ADC_MAX_COUNTS)`.
        *   `ADC_V_REF_mV`: Typically 3300mV for ESP32 if powered by 3.3V. Calibrate this.
        *   `ADC_MAX_COUNTS`: Typically 4095 for 12-bit ADC.
    *   **Low-Pass Filter:** The output `V_out_grove` should be passed through the 2Hz low-pass filter before being fed to the ESP32 ADC pin to remove noise.
*   **ESP32 Firmware Outline (Arduino Core with LSL_ESP32 Library):**
    ```cpp
    #include <Wire.h>            // For I2C communication with MPU-6050
    #include "MPU6050_tockn.h"   // Example MPU-6050 library
    #include "lsl_esp32.h"       // Lab Streaming Layer for ESP32

    #define GSR_ADC_PIN   36 // Example ADC1_CH0 on ESP32 (verify pin for your board)
    #define VCC_MV        3300.0f // Supply voltage to the Grove voltage divider
    #define R_MOD_OHMS    3000.0f // Modified fixed resistor value in the Grove module
    #define ADC_V_REF_MV  3300.0f // ESP32 ADC reference voltage (can be calibrated)
    #define ADC_MAX_VAL   4095.0f // For 12-bit ADC

    MPU6050 mpu6050(Wire);
    LSLOutlet eda_lsl_outlet;
    LSLOutlet acc_lsl_outlet;

    void setup() {
        Serial.begin(115200);
        Wire.begin();
        mpu6050.begin();
        mpu6050.calcGyroOffsets(true); // Calibrate MPU6050 gyro

        // Initialize WiFi or BLE for LSL streaming
        // ...

        // LSL Stream for EDA: name, type, channel_count, nominal_srate, channel_format, source_id
        eda_lsl_outlet.begin("DIY_EDA_GroveGSR_Mod", "EDA", 1, 4.0, lsl_channel_format_t::cft_float32, "grove_gsr_mod_esp32_01");
        // LSL Stream for Accelerometer: name, type, channel_count, nominal_srate, channel_format, source_id
        acc_lsl_outlet.begin("DIY_ACC_MPU6050", "ACC", 3, 50.0, lsl_channel_format_t::cft_float32, "mpu6050_esp32_01");
    }

    unsigned long last_acc_time = 0;

    void loop() {
        // Read GSR value (after external LPF)
        int adc_raw = analogRead(GSR_ADC_PIN);
        float v_out_grove_mv = static_cast<float>(adc_raw) * (ADC_V_REF_MV / ADC_MAX_VAL);
        
        float r_skin_ohms = -1.0f; // Default to invalid
        float conductance_uS = 0.0f; // Default to zero conductance for safety

        if (v_out_grove_mv > 0 && v_out_grove_mv < VCC_MV) { // Avoid division by zero or invalid readings
             r_skin_ohms = R_MOD_OHMS * ( (VCC_MV / v_out_grove_mv) - 1.0f );
             if (r_skin_ohms > 0) { // Ensure skin resistance is positive
                 conductance_uS = (1.0f / r_skin_ohms) * 1000000.0f;
             }
        }
        // Clip to plausible range if necessary, e.g., 0.01 uS to 100 uS

        float sample_eda[] = {conductance_uS};
        eda_lsl_outlet.push_sample(sample_eda);

        // Read Accelerometer data at a different rate (e.g., 50Hz)
        if (millis() - last_acc_time >= 20) { // 1000ms / 50Hz = 20ms
            last_acc_time = millis();
            mpu6050.update();
            float acc_sample[] = {
                static_cast<float>(mpu6050.getAccX()), 
                static_cast<float>(mpu6050.getAccY()), 
                static_cast<float>(mpu6050.getAccZ())
            }; // mpu.getAccX() units depend on library, typically raw or scaled to g's
            acc_lsl_outlet.push_sample(acc_sample);
        }
        delay(240); // Adjust for other processing to achieve ~4Hz EDA (250ms interval)
    }
    ```
*   **Calibration:**
    1.  **ADC Calibration (ESP32):** If possible, calibrate the ESP32 ADC response curve using known precise voltage inputs to map raw ADC counts to millivolts accurately. Store calibration parameters.
    2.  **GSR Circuit Calibration (Two-Point or Multi-Point with Resistors):**
        *   Disconnect electrodes. Place known precision resistors (e.g., 1MΩ, 500kΩ, 200kΩ, 100kΩ, 50kΩ – covering skin resistance range of approx. 0.5 µS to 20 µS conductance) across the electrode input terminals.
        *   Record the corresponding `conductance_uS` output by the firmware.
        *   Plot measured `conductance_uS` vs. true conductance (1/R_known).
        *   Fit a calibration function (linear or polynomial) to correct the firmware's output if it deviates significantly from true values. Apply this correction factor in the main processing pipeline.
*   **Rationale:** Extremely low cost, simple to modify. With careful calibration and filtering, it can provide useful tonic SCL trends and detect phasic SCRs. Integrated accelerometer is crucial for distinguishing motion artifacts from true EDA responses.

---

### **4. Software Stack**

**4.1. Data Acquisition, Synchronization, and Drivers**
*   **Lab Streaming Layer (LSL):**
    *   **Outlets:**
        *   MindWave Mobile (EEG-A): Python script using BrainFlow or TGSP client creates an LSL outlet for raw EEG and/or eSense metrics.
        *   ADS1292R DIY (EEG-B): ESP32 firmware streams 2-channel EEG data directly to an LSL outlet over WiFi or BLE-to-Serial bridge.
        *   Polar H10 (HRV): Python script using BrainFlow or `bleak` creates an LSL outlet for RR-intervals (and optionally raw ECG, ACC).
        *   Grove GSR DIY (EDA): ESP32 firmware streams 1-channel EDA and 3-channel ACC data to LSL outlets over WiFi or BLE.
    *   **Stream Configuration:** (Refer to Appendix C for detailed LSL stream parameters: names, types, channel counts, nominal sample rates, data formats, source IDs). Each stream should have a unique Source ID for robust identification.
    *   **Inlets (Central Processing Unit):** A master Python application uses `pylsl` to create inlets for each LSL stream, continuously pulling timestamped data.
    *   **Synchronization:** LSL's core strength. Relies on `local_clock()` for high-precision timestamps from each source. The master application will use these timestamps to align data from different streams into coherent windows for processing, handling minor clock drifts with LSL's built-in mechanisms or by resampling to a common clock.
*   **Sensor-Specific Drivers/Libraries:**
    *   **MindWave Mobile (EEG-A):** BrainFlow (Python), `mindwave-py`, `pyThinkGear`.
    *   **ADS1292R DIY (EEG-B):** Custom ESP32 firmware based on an ADS1292R Arduino/ESP-IDF library (e.g., from ProtoCentral, community ports, or custom SPI implementation).
    *   **Polar H10 (HRV):** BrainFlow (Python), `bleak` (Python), `polarpy` (Python, Linux).
    *   **Grove GSR DIY (EDA):** Custom ESP32 firmware using Arduino `analogRead()` and an MPU-6050 library (e.g., `MPU6050_tockn`, Adafruit MPU6050).
    *   All ESP32 modules use `lsl_esp32` for LSL communication.

**4.2. Signal Processing Pipeline (Python, on Central Processing Unit)**
Executed on synchronized, windowed data (e.g., 1-5 second windows with 50-75% overlap). Libraries: `NumPy`, `SciPy`, `MNE-Python`, `NeuroKit2`, `hrv-analysis`.

*   **Preprocessing (per modality):**
    *   **EEG (MindWave & ADS1292R):**
        1.  Common Average Referencing (CAR) or Laplacian for multi-channel EEG-B if appropriate.
        2.  Band-pass filter: 0.5-45 Hz (e.g., Butterworth 4th order, zero-phase using `filtfilt`).
        3.  Notch filter: 50 Hz or 60 Hz and its harmonics (e.g., IIR notch or spectral interpolation).
        4.  Artifact Handling:
            *   **Thresholding:** Reject epochs with amplitudes exceeding ±75-150 µV (user-calibrated).
            *   **ICA (Offline/Semi-Online for EEG-B):** If sufficient channels (EEG-B + potentially reference reconfigurations allow for 2 clean + ref), train ICA offline on calibration data to identify blink/eye-movement/muscle components. For real-time, apply pre-trained ICA unmixing matrix or use adaptive regression with EOG proxies (if emulated from frontal channels). For MVP: simpler filtering and robust feature extraction.
            *   **Wavelet Denoising:** Can be effective for certain types of transient noise.
    *   **HRV (Polar H10 - RR Intervals):**
        1.  RR Interval Cleaning (`NeuroKit2.hrv_clean` or similar):
            *   Remove physiologically implausible RRIs (e.g., <250ms, >2500ms).
            *   Detect and correct/interpolate ectopic beats or artifacts (e.g., based on deviation from moving average/median of surrounding RRIs, or using an algorithm like Maliks's rule).
        2.  NN Interval Series: The cleaned series of normal-to-normal intervals.
    *   **EDA (Grove GSR - Conductance Stream & ACC Stream):**
        1.  **ACC Processing:** Calculate vector magnitude of 3-axis acceleration. Apply a threshold to detect significant motion periods.
        2.  **EDA Filtering:**
            *   Low-pass filter (e.g., Butterworth 1-2 Hz) on raw conductance for smoothed signal.
            *   Motion Artifact Rejection: If ACC magnitude > threshold, flag corresponding EDA segment as "motion-contaminated." Features from this segment might be excluded or down-weighted in fusion.
        3.  **Decomposition (e.g., `NeuroKit2.eda_process` or `cvxEDA`):** Separate the EDA signal into:
            *   Tonic component (Skin Conductance Level - SCL).
            *   Phasic component (Skin Conductance Responses - SCRs).
*   **Feature Extraction (Refer to Appendix D for a detailed list):**
    *   **EEG (computed per channel, per window):**
        *   Time Domain: Variance, Kurtosis, Skewness.
        *   Frequency Domain (Welch's PSD):
            *   Absolute Power & Relative Power in bands: Delta (0.5-4Hz), Theta (4-8Hz), Alpha (8-12Hz), Low Beta (12-16Hz), Mid Beta (16-20Hz), High Beta (20-30Hz), Low Gamma (30-45Hz).
            *   Band Ratios: Alpha/Beta, Theta/Beta, (Alpha+Theta)/Beta (Engagement Index), Alpha/(Delta+Theta).
            *   Peak Alpha Frequency (PAF).
        *   Frontal Asymmetry (for ADS1292R Fp1/Fp2): e.g., `(Alpha_Power_Fp2 - Alpha_Power_Fp1) / (Alpha_Power_Fp2 + Alpha_Power_Fp1)`.
        *   MindWave eSense "Attention" and "Meditation" indices (if EEG-A is active).
    *   **HRV (computed per window, e.g., 30-60 seconds with high overlap):**
        *   Time Domain (`NeuroKit2.hrv_time`): Mean HR, Mean RR, SDNN, RMSSD, NN50, pNN50, HRV Triangular Index (HTI).
        *   Frequency Domain (`NeuroKit2.hrv_frequency` - Lomb-Scargle): Power in VLF (0-0.04Hz), LF (0.04-0.15Hz), HF (0.15-0.4Hz) bands (absolute and normalized units), LF/HF ratio, Total Power.
        *   Non-Linear (`NeuroKit2.hrv_nonlinear`): Sample Entropy (SampEn), Approximate Entropy (ApEn), Detrended Fluctuation Analysis (DFA α1, α2), Poincaré Plot (SD1, SD2, SD1/SD2).
    *   **EDA (computed per window, e.g., 5-10 seconds):**
        *   Tonic: Mean SCL, Median SCL, Slope of SCL (linear regression over window).
        *   Phasic (`NeuroKit2.eda_findpeaks` on phasic component): SCR Count (frequency), Mean SCR Amplitude, Max SCR Amplitude, Mean SCR Rise Time, Sum of SCR Amplitudes.
        *   Number of Non-Specific SCRs (NSSCRs) per minute.

**4.3. Sensor Fusion Engine (Machine Learning Model)**
*   **Model Architecture: Stacked Ensemble (XGBoost for Static Probabilities + Kalman Filter for Temporal Smoothing)**
    1.  **Level 0 - Feature Engineering:** As described in 4.2.
    2.  **Level 1 - Static Focus Probability Estimator (XGBoost):**
        *   Input: A concatenated vector of all extracted features (EEG, HRV, EDA) from a single time window.
        *   Output: A probability score (0.0 to 1.0) representing the likelihood of the user being in a "high-focus" state for that window. (Alternatively, a continuous focus regressor).
        *   Model: XGBoost (Gradient Boosted Decision Trees) chosen for its robustness, ability to handle mixed data types, non-linearities, and feature interactions, and good performance in many classification/regression tasks.
        *   Training: Trained offline on a labeled dataset (see Section 5.3). Hyperparameters tuned via cross-validation. Feature importance scores from XGBoost can provide insights into which signals/features are most predictive.
    3.  **Level 2 - Temporal Dynamics & State Smoothing (Kalman Filter):**
        *   State Variable (`x_k`): The latent "true" focus score (e.g., scalar 0-100) at time step `k`.
        *   State Transition Model (`x_k = A * x_{k-1} + B * u_k + w_k`):
            *   `A`: State transition matrix (e.g., `A=0.98` for slight mean reversion or `A=1` for random walk).
            *   `u_k`: Optional control input (not used in baseline).
            *   `w_k`: Process noise (Gaussian, covariance `Q`), representing natural volatility of focus.
        *   Observation Model (`z_k = H * x_k + v_k`):
            *   `z_k`: The probability score from the XGBoost model at time `k` (scaled if necessary).
            *   `H`: Observation matrix (e.g., `H=1` if XGBoost output is already scaled 0-100).
            *   `v_k`: Observation noise (Gaussian, covariance `R_k`).
        *   Adaptive Observation Noise (`R_k`): `R_k` is dynamically adjusted based on sensor quality flags:
            *   If EEG has high artifact content (e.g., blinks, muscle), increase `R_k` contribution related to EEG features.
            *   If EDA has high motion artifact flag, increase `R_k` contribution related to EDA features.
            *   If HRV data window has high ectopy/noise, increase `R_k` contribution related to HRV features.
            This allows the Kalman filter to intelligently down-weight information from unreliable sensor readings in real time.
*   **Training Data Requirements:**
    *   Personalized models are highly recommended. Requires several hours (target 3-5 hours minimum) of diverse, labeled multi-sensor data *per user*.
    *   Labels: Ground-truth focus states derived from cognitive tasks (e.g., n-back levels, Stroop congruency), performance metrics (accuracy, RT), or well-structured self-reports (e.g., KSS + specific task engagement ratings).
*   **Real-Time Update Logic:**
    1.  At each time step (e.g., every 1 second):
    2.  Collect latest window of synchronized data from LSL inlets.
    3.  Preprocess signals and extract the full feature vector.
    4.  Feed feature vector to the trained XGBoost model to obtain `p_focus_static`.
    5.  Update sensor artifact flags and adjust Kalman filter observation noise `R_k`.
    6.  Perform Kalman filter prediction and update steps using `p_focus_static` as the observation.
    7.  The updated Kalman state `x_k` is the final, smoothed real-time focus score.

**4.4. Focus Score API**
*   **Output Format (JSON, streamed via WebSocket, available via REST):**
    ```json
    {
        "timestamp_lsl_window_end": 1716474832.013, // LSL timestamp of the end of the data window
        "timestamp_utc_iso": "2025-05-23T14:33:52.013Z",
        "focus_score_kalman": 73.4,        // Smoothed focus score (0-100) from Kalman filter
        "focus_score_xgb_static": 78.1,    // Instantaneous focus probability from XGBoost (0-100)
        "kalman_state_covariance_P": 0.05, // Uncertainty of the Kalman estimate (lower is more certain)
        "contributing_modalities": {        // Status and confidence/quality of each modality
            "eeg_a_mindwave": {"status": "active", "quality_metric": 0.85, "esense_attention": 75},
            "eeg_b_ads1292r": {"status": "active", "quality_metric": 0.92, "alpha_beta_ratio": 1.2},
            "hrv_polar_h10":  {"status": "active", "quality_metric": 0.95, "rmssd_ms": 45.6},
            "eda_grove_gsr":  {"status": "active", "quality_metric": 0.70, "scr_count_per_min": 3}
        },
        "artifact_summary": {
            "eeg_artifact_level": "low",     // e.g., low, medium, high based on flags
            "hrv_data_cleanliness_pct": 98.5, // Percentage of clean RRIs in window
            "eda_motion_detected": false
        }
    }
    ```
*   **Interface:**
    *   **WebSocket Server:** Pushes JSON updates (e.g., every 1 second) to subscribed clients (HPE Task Scheduler, real-time dashboard).
    *   **REST API Endpoint:** `/api/v1/focus/latest` for polling the most recent focus score object. `/api/v1/focus/history?start_time=&end_time=` for retrieving historical data.

**4.5. Dependency Management and Versioning**
*   **Python Environment:** Use `conda` environments or `venv` with `pip`. All Python dependencies (e.g., `pylsl`, `numpy`, `scipy`, `mne`, `neurokit2`, `xgboost`, `filterpy` for Kalman, `fastapi`/`flask` for API) will be listed in a `requirements.txt` file with pinned versions.
*   **ESP32 Firmware:** Use PlatformIO for managing ESP32 projects, libraries (e.g., ADS1292R driver, LSL_ESP32, MPU6050 lib), and board configurations. `platformio.ini` will specify dependencies and versions.
*   **Source Code Versioning:** Git will be used for all code (Python, ESP32 firmware). Semantic versioning for releases of the Focus Predictor system.
*   **Documentation:** This document and supporting materials (schematics, LSL configs) versioned alongside the code.

**4.6. Error Handling and Logging Strategy**
*   **Sensor Disconnection/Data Gaps:**
    *   LSL streams can signal dropped samples or disconnected outlets.
    *   The fusion engine will detect missing data from a modality. If a primary sensor (e.g., EEG-B) drops, the system can:
        1.  Temporarily rely on fallback sensors (e.g., EEG-A if EEG-B fails).
        2.  Increase uncertainty (`P_k` in Kalman filter) for the focus score.
        3.  Mark the `focus_score` as having lower confidence or based on fewer modalities in the API output.
    *   Implement timeout mechanisms for LSL inlets.
*   **Firmware Errors (ESP32):** Implement basic error handling (e.g., SPI communication failures, WiFi/BLE disconnections) with serial logging and potentially LED status indicators on the DIY devices.
*   **Software Exceptions (Python):** Robust `try-except` blocks in all data processing and API modules. Log errors to a file with timestamps and stack traces.
*   **Logging Levels:** Use standard logging levels (DEBUG, INFO, WARNING, ERROR, CRITICAL). DEBUG for detailed sensor readings and feature values, INFO for major state changes and API requests, ERROR for recoverable issues, CRITICAL for system failures.
*   **User Feedback:** If focus score becomes unreliable due to persistent sensor issues, the API should indicate this, and any consuming application (HPE scheduler) should revert to a safe/default scheduling mode.

---

### **5. Validation Protocol**

A rigorous, multi-stage validation protocol is essential to ensure the accuracy, reliability, and utility of the Focus Predictor.

**5.1. Hardware & Signal Quality Validation (Per Sensor, during initial setup and periodically)**
*   **EEG-A (MindWave):** Verify eSense "Attention" and raw EEG stream are active. Check for gross noise or flatline.
*   **EEG-B (ADS1292R):**
    *   Noise Floor Test (shorted inputs): RMS noise < 5µV, ideally < 2µV.
    *   Alpha Modulation (Eyes-Open vs. Eyes-Closed): Clear, significant (≥3-5dB or 2-3x power) increase in alpha band (8-12Hz) power during eyes-closed segments.
    *   Artifact Signature: Blinks, jaw clenches, and head movements should produce recognizable artifacts, confirming sensor sensitivity.
*   **HRV (Polar H10):**
    *   Compare HR output with manual pulse or another validated HR monitor during rest and light activity.
    *   RR-intervals should be physiologically plausible (e.g., 300-2000ms). RMSSD and SDNN values at rest should be within expected individual ranges.
*   **EDA (Grove GSR):**
    *   Calibration with Known Resistors: Verify that the system outputs accurate conductance values (within ±5-10%) when precision resistors (1MΩ, 500kΩ, 100kΩ) are connected.
    *   Startle Response: A sudden, unexpected stimulus (e.g., loud clap, not during sensitive EEG recording) should elicit a visible SCR.
    *   Motion Artifact Test: Deliberate wrist/finger movements should be detectable by the co-located accelerometer and correlate with noise in the EDA signal.

**5.2. Feature Validation (Against Standard Cognitive Paradigms)**
*   Conduct sessions where users perform tasks known to modulate cognitive load/attention, such as:
    *   **N-Back Task:** (e.g., 0-back, 1-back, 2-back, 3-back conditions).
    *   **Stroop Task:** (Congruent vs. Incongruent trials).
    *   **Sustained Attention to Response Task (SART).**
*   **Expected Feature Changes:**
    *   EEG: Alpha power should decrease, Theta power (especially frontal midline) should increase with higher cognitive load (e.g., higher N in N-back).
    *   HRV: Mean HR should increase, RMSSD/HF power should decrease with higher cognitive load.
    *   EDA: SCL should increase, SCR frequency/amplitude may increase with higher cognitive load or during error commission.
*   **Metric:** Statistically significant differences (e.g., ANOVA, t-tests) in key features between low-load and high-load conditions.

**5.3. Fused Focus Score Validation (Correlation with Ground Truth & Performance)**
*   **Mental Arithmetic vs. Rest:**
    *   **Protocol:** Blocks of mental arithmetic (e.g., serial 7 subtractions) interspersed with quiet resting blocks.
    *   **Metric:** The fused focus score should show a statistically significant increase (e.g., median score difference ≥ 20-25 points) during mental arithmetic compared to rest.
*   **Karolinska Sleepiness Scale (KSS) Correlation:**
    *   **Protocol:** Users provide KSS self-ratings (1-9 scale) 3-5 times per typical workday.
    *   **Metric:** The system's real-time focus score (averaged over 5-15 minutes preceding the KSS rating) should show a significant negative Spearman correlation (e.g., ρ ≤ -0.5 to -0.6) with KSS scores (higher KSS indicates more sleepiness, thus lower focus).
*   **Sustained, Realistic Work Block (e.g., 30-60 min Coding/Writing/Study Session):**
    *   **Protocol:** Users perform a typical cognitively demanding task. Concurrently, they can use a simple tool to self-annotate moments of "deep focus," "minor distraction," or "mind-wandering." Alternatively, screen activity logging + application usage can provide objective proxies for on-task vs. off-task behavior.
    *   **Metric:**
        *   Train/test the fusion model (or just evaluate its output) to classify these annotated epochs.
        *   Target an Area Under the ROC Curve (AUC) ≥ 0.85-0.90 for binary classification of "focused" vs. "clearly distracted/off-task."
        *   Compare performance of the full EEG+HRV+EDA fusion against EEG-only (target AUC for EEG-only ≥ 0.80-0.82 based on literature).

**5.4. Individual Calibration & Onboarding (Per User)**
*   A ~30-minute onboarding session for each new user is critical:
    1.  Sensor fitting and signal quality check.
    2.  Baseline recordings:
        *   5 minutes quiet rest, eyes open.
        *   5 minutes quiet rest, eyes closed (for alpha calibration).
        *   Short cognitive stressor (e.g., 2-5 minutes of N-back or mental arithmetic).
    3.  Collect initial KSS rating and subjective assessment of current state.
*   This data will be used to:
    *   Establish individual baseline ranges for EEG power, HRV metrics, and EDA levels.
    *   Fine-tune thresholds for artifact detection.
    *   Potentially personalize the fusion model (e.g., fine-tune a general XGBoost model with user-specific data or adjust Kalman filter parameters).

---

### **6. Safety & Ethical Considerations**

**6.1. Electrical Safety**
*   **Isolation:** All DIY sensor modules (EEG-B ADS1292R, EDA Grove GSR) making direct skin contact **MUST BE EXCLUSIVELY BATTERY-POWERED** during operation on a human subject. There should be no direct wired connection (e.g., USB for data or power) to any mains-powered device (like a PC) while electrodes are attached, unless a certified medical-grade USB isolator (rated for appropriate voltage and leakage current, e.g., IEC 60601-1 compliant) is used.
*   **Leakage Current:** Verify that DC leakage currents from any DIY module between any applied part (electrode) and earth ground are well below established safety limits (target < 10µA for Type BF applied parts). This requires careful circuit design, component selection (e.g., medical-grade DC-DC converters if any internal voltage boosting is done, though not planned for current BOMs), and empirical testing with a sensitive multimeter.
*   **Component Choice:** Use components rated for skin contact where appropriate. Electrode gels/pastes should be hypoallergenic and non-irritating.
*   **Commercial Devices:** MindWave Mobile and Polar H10 are commercial consumer products and are assumed to meet their respective regional safety certifications (e.g., CE, FCC) when used according to manufacturer instructions.

**6.2. Data Privacy, Security, and User Control**
*   **Informed Consent:** Users must provide explicit, written, informed consent *before* any data collection begins. Consent forms should clearly detail:
    *   What physiological data is being collected (EEG, HRV, EDA, ACC).
    *   How the data is being processed to infer focus.
    *   How and where the data (raw and processed) is stored.
    *   Who will have access to the data.
    *   The purpose of data collection (e.g., to drive the HPE adaptive scheduler, for personal insight, for anonymized research).
    *   The duration of data retention.
    *   The user's right to withdraw consent and have their data deleted at any time.
*   **Data Storage & Transmission:**
    *   **Local First:** Prioritize local storage and processing of physiological data on the user's own computer to maximize privacy.
    *   **Encryption:** If data must be transmitted or stored remotely (e.g., cloud backup, future multi-user research platform), use strong end-to-end encryption for data in transit (TLS/SSL) and encryption at rest (e.g., AES-256 for stored files/databases).
    *   **Access Control:** Implement robust authentication and authorization mechanisms if data is stored centrally.
*   **Anonymization/Pseudonymization:** For research or aggregated analytics, data should be de-identified (anonymized or pseudonymized) to protect user privacy.
*   **Data Minimization:** Collect and retain only the data strictly necessary for the defined purpose of focus prediction and system operation.
*   **User Control & Transparency:**
    *   Users should be able to easily view their own focus data and understand (at a high level) how the focus score is derived.
    *   Provide clear mechanisms for users to pause or stop data collection.
    *   Enable users to request a copy of their data or its deletion.

**6.3. User Burden and Comfort**
*   **Sensor Application:** Minimize the time and effort required for sensor setup and application (e.g., electrode placement, gel application). Provide clear instructions and support.
*   **Comfort:** Ensure headbands, chest straps, and electrode attachments are as comfortable as possible for extended wear (e.g., several hours during a workday). Use breathable materials and consider individual fit.
*   **Psychological Impact:** The system should be framed as a supportive tool for self-optimization and well-being, not as a surveillance or performance judgment mechanism. Avoid features that could induce "focus anxiety" or pressure users to constantly maintain high scores. User agency in interpreting and acting upon focus data is paramount.

**6.4. Algorithmic Bias and Fair Usage**
*   **Bias in Training Data:** Physiological responses can vary based on age, gender, ethnicity, health status, medication, caffeine intake, and other individual factors. If a general (non-personalized) fusion model is developed, ensure the training dataset is diverse and representative to minimize demographic bias in focus predictions.
*   **Personalization as Mitigation:** Prioritize personalized models or robust individual calibration procedures to account for inter-individual variability.
*   **Responsible Use of Focus Score:**
    *   The focus score is an *estimate* and should not be treated as an absolute or infallible measure of a person's cognitive ability or effort.
    *   Avoid using the focus score for direct comparative evaluation between individuals, especially in employment or academic contexts, without extensive validation of fairness and equity.
    *   The primary use should be for individual self-awareness and for driving adaptive system behaviors that benefit the *individual user*.

---

### **7. Budget & Timeline**

**7.1. Hardware Budget**
As itemized in Section 3, the estimated total material cost for one complete 3-sensor stack (comprising EEG-A MindWave, EEG-B ADS1292R DIY, HRV Polar H10, EDA Grove GSR DIY with ACC) is approximately **\$281 USD**.
*   Additional costs for consumables (electrode paste, spare LiPo batteries, 3D printing filament) might be ~$20-50 per year per active user.
*   Tools for assembly (soldering iron, multimeter, 3D printer access) are assumed to be available or a one-time lab setup cost.

**7.2. Development & Implementation Timeline (MVP - 8 Weeks)**
This is an ambitious timeline for an MVP and assumes a skilled individual or small team with relevant expertise.

*   **Week 1: Foundation & EEG-A**
    *   Finalize detailed BOM for all components; place orders.
    *   Setup LSL environment on the host PC.
    *   MindWave Mobile (EEG-A): Develop Python script for connection (via BrainFlow or TGSP client), enable raw EEG and eSense streaming, create LSL outlet. Basic data logging and visualization.
*   **Week 2: EEG-B Assembly & Firmware**
    *   ADS1292R DIY (EEG-B): Assemble hardware (ESP32, ADS1292R breakout, power).
    *   Develop initial ESP32 firmware for ADS1292R control and data acquisition.
    *   Implement LSL streaming from ESP32.
    *   Perform basic signal validation (noise floor, alpha modulation test).
*   **Week 3: HRV Integration**
    *   Polar H10 (HRV): Develop Python script (using BrainFlow or `bleak`) for real-time RR-interval (and optional ECG/ACC) capture.
    *   Create LSL outlet for HRV data.
    *   Implement basic HRV feature extraction (Mean HR, RMSSD) from live stream.
*   **Week 4: EDA Integration & ACC**
    *   Grove GSR DIY (EDA): Modify Grove module (shunt resistor, LPF). Assemble with ESP32 and MPU-6050.
    *   Develop ESP32 firmware for GSR ADC reading, ACC data acquisition, and LSL streaming for both EDA and ACC.
    *   Perform initial EDA calibration with known resistors.
*   **Week 5: Synchronization & Core Processing Framework**
    *   Develop central Python application with LSL inlets for all sensor streams.
    *   Implement robust time synchronization and data windowing/resampling logic.
    *   Structure the preprocessing and feature extraction modules for each modality.
*   **Week 6: Feature Extraction & Initial Data Collection**
    *   Complete implementation of all core feature extraction algorithms (EEG, HRV, EDA features as listed in Section 4.2).
    *   Begin collecting synchronized multi-sensor data during varied cognitive tasks (e.g., rest, n-back, reading, coding) for initial model training. Self-annotation or simple task labels.
*   **Week 7: Fusion Model Development & Training**
    *   Prepare labeled dataset from Week 6 data.
    *   Develop and train the initial XGBoost static probability estimator.
    *   Implement the Kalman filter for temporal smoothing.
    *   Integrate XGBoost output as Kalman observation.
*   **Week 8: API, Validation & MVP Demo**
    *   Implement the Focus Score API (WebSocket and basic REST endpoint).
    *   Conduct initial validation tests (mental arithmetic, KSS correlation, eyes-open/closed on full fused score).
    *   Develop a simple real-time dashboard to visualize sensor streams and the fused focus score.
    *   Demonstrate MVP with live data.

**7.3. Resource Assumptions & Timeline Caveats**
*   **Assumed Resources:** This 8-week timeline likely requires the equivalent of 1.0-1.5 Full-Time Equivalent (FTE) developers with a diverse skillset covering:
    *   Embedded systems programming (ESP32, C++).
    *   Hardware prototyping (soldering, basic circuit understanding).
    *   Python programming (data processing, ML, API development).
    *   Biosignal processing fundamentals.
    *   Basic machine learning model development.
*   **Timeline Risks & Caveats:**
    *   **Hardware Debugging:** DIY hardware (EEG-B, EDA) can be unpredictable. Noise issues, component failures, or firmware bugs can cause significant delays.
    *   **Data Collection & Labeling:** Acquiring sufficient high-quality, labeled data for training the fusion model is often a bottleneck. The initial 2-5 hours per user is for an MVP; robust personalization will require more.
    *   **ML Model Tuning:** Hyperparameter tuning for XGBoost and Kalman filter parameters (Q, R matrices) can be time-consuming.
    *   **LSL Integration:** Ensuring seamless LSL streaming and synchronization across multiple devices (especially WiFi-based ESP32s) can have a learning curve and require network troubleshooting.
    *   **Scope Creep:** The feature set for an MVP should be strictly controlled.
    *   **Recommendation:** Build in 25-50% buffer time for each phase, particularly those involving custom hardware and initial ML model training. A more realistic timeline for a polished, well-validated MVP might be 12-16 weeks.

---

### **8. Future Considerations & Scalability**

*   **Advanced Artifact Rejection:** Implement more sophisticated real-time artifact removal for EEG (e.g., adaptive filtering, regression with dedicated EOG/EMG channels if sensors are added).
*   **Enhanced Fusion Models:** Explore deep learning architectures (e.g., LSTMs, Transformers, multi-branch CNNs) for end-to-end feature learning and fusion, potentially improving accuracy and capturing more complex temporal dependencies.
*   **Contextual Data Integration:** Incorporate additional data streams into the fusion engine, such as:
    *   Time of day, day of week (for circadian/circaseptan rhythm modeling).
    *   User's calendar data (meeting density, task types).
    *   Ambient environment (noise, light, temperature via IoT sensors).
    *   Computer interaction data (keystroke dynamics, mouse movements, application usage).
*   **Personalization Over Time:** Implement online learning or periodic retraining of the fusion model to adapt to individual user changes and improve accuracy over extended use.
*   **Cloud Platform (Optional):** For scaling to multiple users or enabling more complex cloud-based analytics, explore secure cloud ingestion and processing pipelines (e.g., AWS IoT Core + SageMaker, Azure IoT Hub + ML Studio).
*   **Miniaturization & Wearability:** Future hardware iterations could focus on custom PCBs integrating multiple sensors into a more compact and comfortable wearable form factor.
*   **Explainability (XAI):** Develop methods to provide users with insights into *why* the system is generating a particular focus score (e.g., which signals or features are driving the prediction).

---

### **9. Appendices (Placeholders)**

**9.1. Appendix A: Circuit Diagram - EEG-B (ADS1292R DIY Rig)**
    *(Placeholder for detailed schematic: ADS1292R connections to ESP32, power supply, electrode interface. Include component values for filters, pull-ups/downs if any.)*

**9.2. Appendix B: Circuit Diagram - EDA Grove GSR Modification**
    *(Placeholder for detailed schematic: Original Grove GSR circuit, modifications made (shunt resistor, LPF), connections to ESP32, and MPU-6050 connections. Include component values.)*

**9.3. Appendix C: LSL Stream Configuration Details**
    *(Placeholder for a table detailing each LSL stream: Name, Type, Channel Count, Nominal Sample Rate, Channel Format (e.g., float32), Source ID, Channel Labels, Units (e.g., µV, µS, g).)*

**9.4. Appendix D: Detailed Feature List**
    *(Placeholder for an exhaustive list of all features extracted from EEG, HRV, and EDA, including their mathematical definitions or references to standard implementations in libraries like NeuroKit2 or MNE-Python.)*

---

This "gold standard" document attempts to be robust, detailed, and actionable, building upon the strengths of previous versions and directly addressing the critiques. It aims to provide a solid foundation for the technical development of the Real-Time Focus Predictor.