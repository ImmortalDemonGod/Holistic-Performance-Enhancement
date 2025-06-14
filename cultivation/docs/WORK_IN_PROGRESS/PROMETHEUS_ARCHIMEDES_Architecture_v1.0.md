# **PROMETHEUS_ARCHIMEDES: A Hybrid Verifiable Reasoning Architecture**

**Document Version:** 1.0
**Date Created:** 2025-06-12
**Status:** Canonical Vision Blueprint
**Point of Contact:** The Architect

## **Abstract**

This document specifies the long-term architectural vision for a hybrid, verifiable reasoning engine designed to tackle complex, abstract problems, with a path toward Artificial General Intelligence (AGI). This architecture, codenamed **ARCHIMEDES**, is composed of two primary subsystems: a high-performance, intuitive "System 1" engine called **Prometheus**, and a "System 2" cognitive supervisor provided by a foundation Large Language Model (LLM).

The Prometheus engine itself is a hybrid, combining a powerful end-to-end neural model for fast, holistic pattern recognition (`JARC-Reactor`) with a deliberate, symbolic-neural reasoner (the `george` planner) that composes verified, single-purpose neural circuits from a foundational library (built with the `simplest_arc_agi` framework). This allows it to solve a wide range of tasks with both speed and explainable, logical precision.

The ARCHIMEDES framework integrates this Prometheus co-processor with a foundation LLM. The LLM provides world knowledge, grounding for Prometheus's abstract symbols, and, most critically, an **autonomous learning loop**. By identifying failures in the Prometheus engine, the LLM can generate synthetic training data and orchestrate the creation of new skills, enabling the system to learn and improve on its own. This document details the architecture of both Prometheus and ARCHIMEDES, specifies their component interactions, and defines their core interfaces, laying the blueprint for a system with a plausible path to AGI.

---

## **Part I: The Prometheus Architecture (The Core Reasoning Engine)**

Prometheus is designed to be the ultimate solver for well-defined, abstract domains like the ARC challenge. It operates on a dual-process model inspired by human cognition to achieve a balance of speed, power, and explainability.

### **2.1. The "System 1 / System 2" Cognitive Model**

Prometheus explicitly separates fast, intuitive pattern-matching from slow, deliberate, step-by-step reasoning. This allows it to allocate the right kind of cognitive resource to a problem, maximizing both efficiency and accuracy.

*   **System 1 (The Intuitive Engine):** A powerful, end-to-end model trained for holistic pattern recognition. It provides a fast, "best guess" answer to any given task. It is highly capable for problems it has seen patterns for, but its internal reasoning is opaque.
*   **System 2 (The Deliberate Reasoner):** A symbolic planner that constructs solutions by composing a library of smaller, verifiable, single-purpose functions. It is slower and more resource-intensive, but its reasoning process is transparent and trustworthy, making it ideal for novel or multi-step logical problems.

An orchestrator module uses a **confidence check** to intelligently decide which system's output to trust, escalating to the more rigorous System 2 only when System 1 is uncertain.

### **2.2. System 1: The Perceptual & Heuristic Engine (`JARC-Reactor`)**

The role of System 1 is fulfilled by the `JARC-Reactor` architecture, pushed to its performance limits.

*   **Core Technology:** A sophisticated Encoder-Decoder Transformer model, highly optimized for ARC tasks. Its key innovation is the **`ContextEncoderModule`**, which allows it to perform in-context, few-shot learning by generating a task-specific "hint" from a single demonstration pair.
*   **Role in Prometheus:** It serves as the first-pass solver. Given a new task, it makes an immediate, holistic prediction based on its vast training on similar patterns.
*   **Critical Modification (Weakness Mitigation):** The "black box" nature of the Transformer is mitigated by modifying its output layer to produce a **confidence score** alongside its prediction. This score, which can be derived from the entropy of the final logits or via Monte Carlo dropout during inference, allows the system to assess its own certainty. A high-confidence prediction can be trusted and returned immediately; a low-confidence prediction signals the need for more rigorous, deliberate reasoning and triggers an escalation to System 2.

### **2.3. System 2: The Deliberate Reasoner (`george` + `simplest_arc_agi`)**

System 2 is a composite architecture that combines a high-level planner with a library of callable, verifiable skills. It is designed for transparency and logical precision.

*   **2.3.1. The Foundational Library of Verifiable Skills (`simplest_arc_agi`)**
    *   **Role:** This is the **System 2 Toolbox**. The `simplest_arc_agi` framework is used as a "skill factory" to generate and validate a library of fundamental neural circuits.
    *   **Implementation:** It is used to train hundreds of small, specialized models on atomic, fundamental concepts relevant to abstract reasoning (e.g., `rotate_90`, `count_objects`, `detect_symmetry`, `bitwise_xor`). Using mechanistic interpretability techniques, these "circuits" are verified to ensure they have learned the correct function. The resulting collection of verified, single-purpose neural models is stored in the `circuits.db`, which includes rich metadata about each circuit's function, performance, and interface.

*   **2.3.2. The Symbolic-Neural Planner & Composer (`george`)**
    *   **Role:** This is the **System 2 Orchestrator**. The conceptual architecture of `george` is realized as a high-level planner. When activated, it receives a problem that System 1 failed to solve confidently.
    *   **Implementation:** Its core task is to formulate a multi-step, symbolic plan to solve the problem. It does this by:
        1.  **Decomposing** the problem into a sequence of logical steps based on an analysis of the task's demonstration pairs.
        2.  **Querying** the Foundational Library (`circuits.db`) to find the appropriate neural circuit for each step.
        3.  **Assembling** these circuits into an executable program (a plan).
        4.  **Executing** the program, managing the flow of data between steps and using its `ScratchpadMemory` for intermediate results, to arrive at a final answer.
    *   This creates a powerful symbiosis: the `george` planner is no longer limited by a small set of hardcoded functions, and the `simplest_arc_agi` library is no longer just a collection of isolated models. Together, they form a dynamic, extensible, and verifiable reasoning engine.

### **2.4. The Prometheus Workflow Diagram**

The complete workflow integrates these components into a cohesive, hierarchical decision-making process.

```mermaid
graph TD
    A[Task Input] --> B{System 1: JARC-Reactor Engine};
    B --> C{"Confidence Check"};
    C -- High Confidence --> D[Output Final Answer];
    C -- Low Confidence --> E{Escalate to System 2};

    subgraph System 2
        E --> F[george Planner: Decompose Problem & Formulate Plan];
        F -- Formulates Plan by... --> G[Querying circuits.db];
        G --> H{Foundational Library<br/>(simplest_arc_agi circuits)};
        H -- Returns Circuits --> F;
        F -- Assembled Plan --> I[george Composer: Execute Program];
    end
    
    I --> K[Output Verified Answer & Trace];
    D --> L[End]
    K --> L

    style D fill:#d4f8d4,stroke:#333
    style K fill:#d4f8d4,stroke:#333
    style B fill:#fff0e6,stroke:#333
    style C fill:#fde0dc,stroke:#333
    style H fill:#e6e6fa,stroke:#333
```

---

## **Part II: The ARCHIMEDES Architecture (The Path to AGI)**

While Prometheus is a powerful specialist, it is not a general intelligence. It lacks world knowledge, the ability to operate outside its domain, and the capacity to learn new skills autonomously. The ARCHIMEDES architecture solves these problems by integrating Prometheus with a foundation Large Language Model (LLM).

### **3.1. Plugging the Gaps: The Role of the Foundation LLM**

The LLM acts as the **Executive and World Modeler**, providing the capabilities that Prometheus lacks:

1.  **Generality:** The LLM can understand and process information from any domain (natural language, code, images, etc.). It serves as the primary interface to the outside world, translating user requests or unstructured problems into a format that the Prometheus engine can understand.
2.  **Grounding:** The LLM possesses a vast, implicit world model. It provides the semantic grounding for the abstract symbols and operations within Prometheus. It "knows" what `rotate_90` means in the real world, allowing it to apply Prometheus's logical reasoning to grounded problems.
3.  **Autonomous Learning:** The LLM acts as the supervisor in a self-improvement loop, enabling the system to learn new skills without human intervention.

### **3.2. The Autonomous Learning Loop**

This is the most critical feature of the ARCHIMEDES architecture. It allows the system to identify its own weaknesses and proactively fix them, creating a path to recursively self-improving intelligence.

**The Loop:**

1.  **Failure Identification & Explanation:** The Prometheus engine encounters a task it cannot solve. Its planner reports a specific failure: "My plan requires the skill `calculate_convex_hull`, but this circuit is not in my library."
2.  **Conceptual Understanding & Data Generation (LLM):** The LLM receives this error. Because of its world knowledge, it understands the concept of a "convex hull." It then uses its generative capabilities to create a new, high-quality synthetic training dataset containing thousands of simple shapes and their corresponding correct convex hulls, formatted for the `simplest_arc_agi` training framework.
3.  **Supervised Skill Acquisition (Prometheus):** The LLM triggers the `simplest_arc_agi` training pipeline, using the newly generated dataset to train a new, specialized `convex_hull_c1` neural circuit until it achieves mastery.
4.  **Verification & Integration (LLM):** The LLM supervises the verification of the new circuit. Once the circuit meets a performance threshold, the LLM adds it to the Foundational Library (`circuits.db`), complete with a new semantic description and interface specification.
5.  **Re-attempt & Generalization:** The system can now re-attempt the original failed task. More importantly, the `george` planner can now use the `convex_hull` skill in any future plan, permanently expanding the system's capabilities.

### **3.3. The ARCHIMEDES Architecture Diagram**

This diagram illustrates the symbiotic relationship between the LLM Executive and the Prometheus Co-Processor.

```mermaid
graph TD
    subgraph ARCHIMEDES_Framework
        LLM[Foundation LLM<br/>(Executive & World Modeler)]
        Prometheus[Prometheus Engine<br/>(Logical Co-Processor)]

        subgraph Prometheus
            System1[System 1: JARC-Reactor]
            System2[System 2: george + circuits.db]
        end
    end

    UserInput[User Request / New Problem] --> LLM
    LLM -- Formulated Task --> Prometheus
    Prometheus -- Result / Confidence --> LLM
    Prometheus -- Failure Report --> Autonomous_Learning_Loop
    LLM -- Final Answer --> UserOutput

    subgraph Autonomous_Learning_Loop
        Prometheus -.->|"1. Failure: Skill 'X' Missing"| LLM
        LLM -- "2. Generate Training Data for 'X'" --> SyntheticData[Synthetic Data]
        SyntheticData -- "3. Trigger Training" --> Training[simplest_arc_agi Pipeline]
        Training -- "4. New Verified Circuit" --> LLM
        LLM -- "5. Integrate into Library" --> System2
    end

    style Autonomous_Learning_Loop fill:#f0f0f0,stroke:#666,stroke-dasharray: 5 5
```

---

## **Part III: Interface Specifications (Initial Draft)**

To make this architecture concrete, we must define the "APIs" and data structures that allow these core components to interact. Versioning is a key principle to ensure long-term stability and backward compatibility.

### **4.1. Orchestrator Call Signatures (Python-like)**

The top-level solver would interact with Prometheus via a well-defined orchestrator interface.

```python
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class Solution:
    answer: Any
    status: str  # 'Success', 'Failure: Plan Formulation', 'Failure: Execution'
    engine_used: str
    confidence: float
    plan_executed: Optional[Plan] = None  # The Plan object that was run
    execution_trace: Optional[Dict] = None  # The detailed trace from the composer

class PrometheusOrchestrator:
    def solve(self, task: ArcTask, confidence_threshold: float = 0.9) -> Solution:
        """
        Attempts to solve an ARC task using the hybrid engine.

        Returns:
            A Solution object containing the answer and rich metadata
            about the solving process.
        """
        # 1. System 1 attempt
        system1_prediction, confidence = self.jarc_reactor.predict_with_confidence(task)

        # 2. Confidence check and escalation
        if confidence >= confidence_threshold:
            return Solution(
                answer=system1_prediction,
                method="System 1 (Heuristic)",
                confidence=confidence,
                trace=None
            )
        else:
            # 3. System 2 attempt
            plan = self.george_planner.formulate_plan(task)
            if plan and plan.is_executable:
                system2_prediction, trace = self.circuit_composer.execute(plan)
                return Solution(
                    answer=system2_prediction,
                    method="System 2 (Compositional)",
                    confidence=1.0,  # Assumed high confidence due to verifiable steps
                    trace=trace
                )
            else:
                # Fallback if S2 cannot formulate a valid plan
                return Solution(
                    answer=system1_prediction, # Fallback to S1's best guess
                    method="Failed (Plan Formulation)",
                    confidence=0.0,
                    trace=plan.error_log if plan else None
                )
```

### **4.2. `CircuitDatabase` Query API (Python-like)**

The `george` planner needs a robust API to query the Foundational Library for skills. Versioning is critical here.

```python
# from cultivation.systems.arc_reactor.simplest_arc_agi.database.circuit_database import CircuitDatabase

@dataclass
class CircuitRecord:
    circuit_id: str
    version: str # e.g., "1.0", "1.1"
    description: str
    tags: List[str]
    model_checkpoint_path: str
    interface_spec: dict # JSON schema defining input/output tensors
    fidelity_score: float

class CircuitLibrary:
    def __init__(self, db_path: str):
        # ... connection logic ...
        pass

    def search_circuits(
        self,
        description: str | None = None,
        tags: list[str] | None = None,
        version: str | None = "latest" # Allows querying for specific or latest versions
    ) -> list[CircuitRecord]:
        """
        Searches the library for verified neural circuits matching criteria.
        'description' uses semantic search on circuit interpretations.
        Returns a list of candidate circuits, ranked by relevance.
        """
        # ... SQL query building and execution logic ...
        pass
```

### **4.3. `Plan` Data Structure (YAML Example)**

A plan generated by the `george` planner is a Directed Acyclic Graph (DAG) of operations, represented here in a human-readable YAML format. Note the specification of a circuit version.

```yaml
# Example plan structure for a task.

plan_id: "plan-arc-task-123"
description: "Find largest blue object and rotate it."
steps:
  - id: "s1_find_blue"
    # Call a specific version of a circuit to ensure reproducibility
    operation: "find_objects_by_color:v1.2"
    inputs:
      grid: "task.input_grid"  # Global input
      color: "blue"            # Literal value
    outputs: ["blue_objects"]  # Intermediate variable

  - id: "s2_get_largest_blue"
    operation: "get_largest_object:v1.0"
    inputs:
      object_list: "s1_find_blue.blue_objects" # Output from a previous step
    outputs: ["largest_blue"]

  - id: "s3_rotate_largest"
    operation: "rotate_180:v1.0"
    inputs:
      object_to_transform: "s2_get_largest_blue.largest_blue"
    outputs: ["rotated_object"]
    
output_source: "s3_rotate_largest.rotated_object" # Final output node
```

This structure defines a verifiable program where each `operation` is a versioned call to a trusted neural circuit from the library, ensuring the stability and reproducibility of complex reasoning chains.