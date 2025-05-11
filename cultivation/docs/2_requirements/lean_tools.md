DeepSeek-Prover-V2 at a glance

Feature	Details
Release	30 April 2025 (arXiv & GitHub)  ￼
Model sizes	7 B (single-GPU friendly) and 671 B MoE (built on DeepSeek-V3)  ￼
Context length	Up to 32 k tokens (7 B variant)  ￼
Training recipe	Recursive sub-goal decomposition + RL from Lean 4 feedback. Cold-start proofs are harvested with DeepSeek-V3, then the prover is fine-tuned and finally RL-tuned on binary success signals.  ￼
Headline results	88.9 % pass rate on miniF2F-test (↑ 25 pp over the previous SOTA of 63.5 % by DeepSeek-Prover-V1.5)  ￼ ￼
Extra benchmarks	49 / 658 PutnamBench problems proved; new ProverBench dataset of 325 tasks (15 from AIME 24-25) released.  ￼
Licence	DeepSeek Open Model Licence – permissive for research & commercial use, with standard responsible-use clauses.  ￼


⸻

What’s new and why it matters
	1.	Massive quality jump. The jump from 63 % → 88 % on miniF2F is the largest single-release gain the Lean ecosystem has seen, putting the model well beyond specialised tree-search systems like ProofAug (66 %). This is mostly due to the sub-goal RL loop, which blends informal chain-of-thought with formally-checked Lean tactics.  ￼
	2.	Two usable tiers.
	•	7 B: weights on HuggingFace, one-GPU inference, 32 k context – perfect for local experimentation or fine-tuning on your speciality domains.
	•	671 B: MoE giant that inherits DeepSeek-V3 routing; remote-inference only for most users, but it shows how far scaling helps formal reasoning.  ￼
	3.	ProverBench. The authors note that existing test sets (miniF2F, ProofNet) saturate quickly; ProverBench injects fresh AIME-level problems plus textbook calculus/analysis, giving a harder target for future work.  ￼
	4.	Better alignment with Lean 4. V2 ships native Lean 4 syntax hints, configurable set_options, and longer heart-beat budgets, reducing “almost-proved” failures that plagued V1.x.

⸻

Quick-start (7 B variant)

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, json

model_id = "deepseek-ai/DeepSeek-Prover-V2-7B"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)

lean_problem = """
import Mathlib
open Real

/-- Show √2 is irrational -/
theorem sqrt2_irrational : ¬ ∃ (a b : ℕ), b ≠ 0 ∧ (a^2) = 2*(b^2) := by
  sorry
"""

prompt = f"""
Provide a high-level proof plan, then fill in Lean 4 code:

```lean
{lean_problem}
```"""
inputs = tok.apply_chat_template([{"role":"user","content": prompt}],
                                 add_generation_prompt=True,
                                 return_tensors="pt").to(model.device)
out = model.generate(inputs, max_new_tokens=2048)
print(tok.decode(out[0], skip_special_tokens=True))

The model will first outline a sub-goal plan (e.g., “assume √2 = a/b, WLOG coprime…”) and then emit Lean tactics that discharge each sub-goal.

⸻

How this compares to other provers you’ve used

Model	MiniF2F-test ↑	ProofNet ↑	Strategy
GPT-4 (0-shot Lean)	23 %	≈ 5 %	plain decoding
DeepSeek-Prover-V1	52 %	–	synthetic data only
DeepSeek-Prover-V1.5	63.5 %	25 %	RL + tree-search
ProofAug (MIT)	66 %	–	data augmentation
DeepSeek-Prover-V2 (7 B)	88.9 %	not reported	sub-goal RL

Numbers from original papers or leaderboard snapshots.  ￼

⸻

Why you (specifically) might care
	•	Lean Copilot integration. Because V2 exposes a chat interface and Lean-aware ε-tactic suggestions, wiring it into your Lean Copilot prototype should be straightforward—just swap out the backend call and extend your aggregation logic to parse the model’s proof plans.
	•	Circuit-extraction research. The paper’s ablation study hints that sub-goal decomposition creates tight activation clusters around tactic categories—useful fodder for the circuit-extraction framework you sketched on 04 Mar.
	•	Set-theoretic combinatorics project. Formalising your Venn-intersection lemmas in Lean and letting V2 attack them could benchmark automated discovery vs manual proofs.
	•	LoRA fine-tuning. The 7 B weights accept standard LoRA adapters; you can cheaply specialise on domain-specific math (e.g., survival-analysis measure theory) without touching the monster 671 B model.

⸻

Caveats & next steps
	•	Licensing: Commercial use is fine, but you must avoid generating disallowed content and must disclose model usage per the licence.  ￼
	•	Hardware: 671 B requires multi-node inference or services like DeepSeek Cloud; context windows beyond 32 k are not yet supported.
	•	Benchmark saturation: MiniF2F may no longer differentiate models; expect future papers (and your evaluations) to pivot to harder suites like ProverBench or IMO-Grand.

Feel free to tell me if you’d like code for Lean Copilot integration, a deeper dive into the RL algorithm, or guidance on fine-tuning workflows.