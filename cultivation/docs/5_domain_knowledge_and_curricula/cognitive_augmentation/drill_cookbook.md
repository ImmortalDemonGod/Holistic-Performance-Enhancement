# Mentat-OS: The Drill Cookbook v1.0

**Document Version:** 1.0
**Date Created:** 2025-06-11
**Status:** Definitive Proposal
**Parent Blueprint:** [`mentat_os_blueprint.md`](./mentat_os_blueprint.md)

## 0. Preamble: The Philosophy of Deliberate Cognitive Practice

This document serves as the central repository and operational guide for all training drills within the Mentat-OS. It translates the theoretical architecture of the five cognitive layers into concrete, repeatable, and measurable exercises.

The philosophy behind this cookbook is **deliberate practice**. Each drill is designed not for passive completion, but for focused, high-intensity engagement aimed at pushing the boundaries of a specific cognitive skill. The drills are structured with clear objectives, progression ladders, and success criteria to enable systematic improvement and data-driven adaptation.

This cookbook is a living document. Drills will be refined, added, or even retired based on their measured effectiveness in improving the Key Performance Indicators (KPIs) outlined in the Mentat-OS Blueprint.

---

## 1. General Training Protocol & Scheduling

*   **Daily Commitment:** A standard daily practice block is **20 minutes**, integrated into a suitable time slot (e.g., "Flex-Slot #1" or a dedicated "Cognitive Practice" block).
*   **Structure of Daily Block (Example):**
    *   **Warm-up (2 min):** Focused breathing, mental state check.
    *   **Core Drill 1 (7 min):** E.g., Working Memory Span (D3).
    *   **Core Drill 2 (5 min):** E.g., Rapid Recode (D1).
    *   **Rotating Skill Drill (5 min):** Cycle through one of D2, D4, D5, D6, D7, or D8 daily.
    *   **Log & Reflect (1 min):** Log scores and a one-sentence reflection in the daily log.
*   **Tools:**
    *   A simple timer is essential for all timed drills.
    *   A dedicated physical or digital notebook for logging results.
    *   The `mentat_autograder.py` script for automated scoring and tracking.
    *   An AI assistant (e.g., ChatGPT, Claude) for generating stimuli and providing feedback where noted.

---

## 2. Drill Library: By Cognitive Layer

##### **Layer 1: The Intuition Engine - "To Synthesize"**

###### **Drill ID: D1 - Rapid Recode**

*   **Objective:** To rapidly compress complex information into concise, salient headlines, training the mind to quickly grasp the essence of data.
*   **KPI:** `RapidRecode-Recall` (% of headlines recalled correctly after a delay).
*   **Procedure:**
    1.  Select a source of dense information (e.g., the abstract of a new scientific paper, a news article, a business report).
    2.  Set a timer for **60 seconds per item**.
    3.  Read the item and immediately formulate and **speak aloud** a 3-5 word headline that captures its core message (e.g., "AI model beats benchmark via new method").
    4.  Repeat for 3-5 items.
    5.  After a 5-minute delay (e.g., do another drill), attempt to recall the exact headline for each item from memory.
*   **Progression Ladder:**
    *   **Level 1:** 3 items, 60s/item.
    *   **Level 2:** 5 items, 45s/item.
    *   **Level 3:** 7 items, 30s/item.
*   **HPE Integration:** The `mentat_autograder.py` script can score this by comparing your recalled headlines to a saved log. The `literature/notes/` directory is an excellent source of daily items.

###### **Drill ID: D2 - Forced Analogy**

*   **Objective:** To train the brain's ability to find deep structural similarities between seemingly unrelated concepts, fostering creative problem-solving.
*   **KPI:** `Analogy-Quality-Score` (self-rated 1-5 based on a rubric assessing novelty, structural soundness, and actionability).
*   **Procedure:**
    1.  Use a script or tool to randomly select two concepts from your `cultivation` knowledge base (e.g., from flashcard tags or document titles). Example pair: "Logistic Growth Model" and "Pytest-Error-Fixing-Framework."
    2.  Set a timer for **3 minutes**.
    3.  Generate and articulate **three plausible analogies** or structural connections between the two concepts.
        *   *Example Connection:* "The logistic model has a 'carrying capacity' (K) that limits growth, which is analogous to the `pytest-fixer`'s 'max retries' parameter that limits the search for a solution. Both prevent infinite, unproductive processes."
*   **Progression Ladder:**
    *   **Level 1:** Use two concepts from the *same* domain (e.g., two different running training blocks).
    *   **Level 2:** Use two concepts from *adjacent* domains (e.g., Running and Strength Training).
    *   **Level 3:** Use two concepts from *distant* domains (e.g., RNA Modeling and Social Dynamics).
*   **HPE Integration:** The concepts can be pulled directly from the `data/` and `docs/` directories. Logged analogies can be added to the KCV "Think Tank" knowledge graph.

---

##### **Layer 2: The Cognitive Core - "To Compute & Verify"**

###### **Drill ID: D3 - Working-Memory Span & Manipulation**

*   **Objective:** To expand the capacity and flexibility of working memory (your mental "RAM").
*   **KPI:** `WM-Span` (maximum number of items correctly recalled and/or manipulated).
*   **Procedure:**
    1.  Use an AI assistant or a simple script to generate a sequence of N random digits (e.g., `7 1 9 4 2 ...`).
    2.  **Recall Task:** Display the sequence for N seconds, then hide it. Recall the sequence in order.
    3.  **Manipulation Task:** Display the sequence for N*2 seconds. Recall the sequence, but with a transformation applied to each digit (e.g., "add 3 to each digit," or "reverse the order").
*   **Progression Ladder:**
    *   **Level 1:** Simple recall, starting with N=7. Increase N by 1 each time you succeed 3 times in a row.
    *   **Level 2:** Manipulation (reverse order), starting with N=5.
    *   **Level 3:** Manipulation (arithmetic), starting with N=4.
*   **HPE Integration:** The `mentat_autograder.py` can generate the stimuli and score the recall accuracy.

###### **Drill ID: D4 - Mental Algorithm Trace**

*   **Objective:** To train the ability to mentally simulate and track the state of simple algorithms.
*   **KPI:** `Trace-Accuracy` (% of steps correctly tracked).
*   **Procedure:**
    1.  Select a short, simple algorithm (e.g., Bubble Sort, finding the max value in a list, a basic loop with a conditional).
    2.  Select a small input dataset (e.g., an unsorted list of 8 single-digit numbers).
    3.  **Mentally execute the algorithm step-by-step.** Use finger-tapping or a similar kinesthetic cue to track the "program counter" or loop index. Verbally state the contents of key variables at the end of each major step or loop.
    4.  Compare your final state and output with the correct result (which can be computed by an AI or script).
*   **Progression Ladder:**
    *   **Level 1:** Linear search on 10 elements.
    *   **Level 2:** Bubble sort on 8 elements.
    *   **Level 3:** Mentally trace a simple recursive function (e.g., factorial) to a depth of 3-4 calls.
*   **HPE Integration:** A script can generate the algorithm and input, and provide the correct trace for comparison. This directly supports the Software Engineering domain.

###### **Drill ID: D5 - Mental Arithmetic (Abacus/Soroban Method)**

*   **Objective:** To develop high-speed, accurate mental calculation abilities.
*   **KPI:** `Math-SPS` (two-digit products per minute with ≤2% error).
*   **Procedure:**
    1.  Use a mental arithmetic training app or a script that generates problems.
    2.  Practice visualizing an abacus or soroban to perform calculations.
    3.  **Drill Type 1 (Speed):** Complete as many problems of a certain type (e.g., 2-digit + 2-digit addition) as possible in 60 seconds.
    4.  **Drill Type 2 (Complexity):** Attempt more complex problems (e.g., 3-digit x 2-digit multiplication) without a time limit, focusing on accuracy.
*   **Progression Ladder:**
    *   **Level 1:** 2-digit addition/subtraction.
    *   **Level 2:** 2-digit x 1-digit multiplication.
    *   **Level 3:** 2-digit x 2-digit multiplication.
*   **HPE Integration:** `mentat_autograder.py` can generate problems and check answers, logging `Math-SPS` scores to the KPI dashboard.

---

##### **Layer 3: The Somatic Interface - "To Sense & Ground"**

###### **Drill ID: D6 - Interoceptive Focus**

*   **Objective:** To increase sensitivity to internal physiological signals ("gut feelings").
*   **KPI:** `Somatic-Insight-Rate` (% of logged decisions where somatic state correctly predicted outcome).
*   **Procedure:**
    1.  Set a timer for **3 minutes**.
    2.  Sit quietly and perform a body scan, paying non-judgmental attention to physical sensations: heart rate, breathing rhythm, muscle tension (jaw, shoulders, gut), temperature, etc.
    3.  Verbally label the sensations as they arise (e.g., "noticing warmth in chest," "awareness of heart beating fast").
    4.  At the end, summarize the overall state in one or two words (e.g., "Calm and Centered," "Agitated and Tense").
*   **Progression Ladder:**
    *   **Level 1:** Perform the drill in a quiet, distraction-free environment.
    *   **Level 2:** Perform the drill in a slightly distracting environment (e.g., with background noise).
    *   **Level 3:** Perform a 60-second "spot check" of your somatic state in the middle of a complex work task.
*   **HPE Integration:** This drill is best performed just before a session tracked by the **Focus Predictor**. The subjective state ("Calm and Centered") can be correlated with objective biometric data (HRV, EDA) to build a personalized mind-body dictionary.

---

##### **Layer 4: The Social Dynamics Engine - "To Relate & Persuade"**

###### **Drill ID: D7 - Argument Reframing**

*   **Objective:** To practice tailoring communication to different audiences and motivations, a core skill of influence.
*   **KPI:** `Persuasion-Success-Rate` (in mock scenarios).
*   **Procedure:**
    1.  Take a single, clear argument or proposal (e.g., "We should adopt `pytest-fixer` for our CI/CD pipeline").
    2.  Set a timer for **5 minutes**.
    3.  Write or speak three distinct versions of the argument, each framed for a different audience:
        *   **Audience 1 (The Engineer):** Focus on technical benefits, efficiency gains, and code quality.
        *   **Audience 2 (The Manager):** Focus on ROI, time saved, reduced bug-fixing costs, and team productivity.
        *   **Audience 3 (The Skeptic):** Proactively address potential risks, failure modes, and implementation costs, and present mitigation strategies.
*   **Progression Ladder:**
    *   **Level 1:** Reframe a simple, familiar argument.
    *   **Level 2:** Reframe a complex, nuanced technical argument.
    *   **Level 3:** Reframe an emotionally charged or controversial argument.
*   **HPE Integration:** Can be paired with the **SVEP** initiative. The reframed arguments can be stored as communication templates.

---

##### **Layer 5: The Ethical Governor - "To Govern & Decide"**

###### **Drill ID: D8 - Value-Conflict Resolution**

*   **Objective:** To practice making decisions under ethical or value-based conflict, strengthening the "governor" faculty.
*   **KPI:** `Ethical-Conflicts-Resolved` (count of scenarios analyzed).
*   **Procedure:**
    1.  Use an AI assistant to generate a realistic dilemma that creates a conflict between two of your stated core values from your `personal_constitution.md`.
        *   *Example Prompt:* "Generate a scenario for a research scientist where the 'pursuit of knowledge' conflicts with the 'do no harm' principle."
    2.  Set a timer for **5 minutes**.
    3.  Write down a structured decision following the Decision Journal format:
        *   **Decision:** Clearly state the choice to be made.
        *   **Values in Conflict:** Identify the core values at stake.
        *   **Stakeholder Analysis:** List the parties affected and the potential consequences for each.
        *   **Resolution Principle:** Articulate a higher-order principle or a compromise that resolves the conflict.
        *   **Final Action:** State the chosen course of action.
*   **Progression Ladder:**
    *   **Level 1:** Simple, clear-cut dilemmas.
    *   **Level 2:** Dilemmas with significant ambiguity or unknown second-order effects.
    *   **Level 3:** Time-pressured dilemmas requiring a rapid but principled decision.
*   **HPE Integration:** The outputs of this drill directly populate the "Decision Journal," creating a long-term record of your ethical reasoning and refining your `personal_constitution.md`.

---

#### **3. Conclusion: The Path to the Hybrid Cortex**

This Drill Cookbook provides the practical, day-to-day training regimen for building the Mentat-OS. Consistent practice of these drills, combined with automated scoring and integration into the broader "Cultivation" project, will systematically enhance the five core cognitive layers. This is the operational path from a baseline human mind to a "Hybrid Cortex" capable of extraordinary, synergistic performance with its AI partner.
