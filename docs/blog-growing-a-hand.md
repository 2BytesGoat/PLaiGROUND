# Growing a Hand

Status: idea  
Updated: 2026-10-27

## Hook

> "Suddenly growing a hand will probably lead to you not using it at all, even if in theory that hand would make you a 1.25x human."

That is the blog post.

A new sense is not a plug-in. A competent agent already has a working solution, and a new input has no established credit assignment, no learned calibration, and no reason to be trusted. It gets ignored — not because it is useless, but because the old policy already knows how to move.

## Thesis

Adding a sensory channel to an existing policy is not additive. The policy has to *co-adapt* with the new signal. If the old solution still works, the optimizer has little pressure to discover what the new signal means.

This is the difference between:

- **bolting a hand onto an adult** — the brain already has a body model; the hand is unexplained extra data;
- **growing up with the hand** — the hand is part of the feedback loop from the beginning.

## Evidence from the Dragon Jump experiments

### 1. The 754 baseline

The classic 7×7 multi-hot observation reached:

- **1.633 s** canonical 1-1
- **60/60** 1-1 starts
- later, a 200k multilevel continuation covered all 242 starts across 1-1/1-2/1-3

### 2. Sensors added to a solved policy are ignored

The 794 observation adds 20 head-mounted ray distances and 20 hit flags to the 754-value grid.

Early sensor runs initialized the new columns at zero, then fine-tuned the existing policy. The sensor was present but did not become decision-critical. The policy kept using the old route.

The sensor existed. The agent did not use it.

### 3. Forcing adoption is possible, but expensive

A 1-1 canary:

- expanded 754 → 794
- sampled the new sensor columns from existing weights
- froze the original 754 input columns
- applied 15% grid dropout

Result:

- full sensor: **2.400 s** canonical
- sensor zeroed: **1.633 s** canonical
- sensor columns: ~50% of the old policy-column norm

So the sensor became causal — zeroing it changed behavior — but the shared network learned a **slower route** around it. The hand was being used, badly.

### 4. From scratch, the sensor becomes necessary

A fresh 794 policy, no expansion and no frozen weights, was trained on 1-1 from random initialization:

- at **400k** steps: **2.400 s** canonical, **60/60** starts
- with the 40 sensor values zeroed: **22/60** starts, canonical timeout

First-layer sensor-column norms at 400k:

- policy: **1.10×** the average base-column norm
- value: **1.42×** the average base-column norm

This is the clean result: the sensor is not decoration. Remove it and the policy collapses.

The 1M run is still in progress. The final comparison to 754 is still open.

## The point

The interesting part is not “sensors good” or “sensors bad.”

It is that the same sensor can be:

1. **ignored** when added to a solved policy;
2. **forcibly adopted** when the old input path is constrained, at the cost of a worse route;
3. **essential** when learned from scratch.

Capability is not a component you attach. It is a relationship between the agent, its body, and the feedback it has learned to act on.

## Caveats

- These results are 1-1 only; 1-3 sensor behavior is not proven yet.
- The 400k scratch model is still slower than the 754 control (**2.400 s vs 1.633 s**).
- Zeroing the sensor is a distribution shift, but it is a direct causal intervention, not just a weight-correlation argument.
- The 1M checkpoint and 1-3 transfer must be checked before claiming the sensor approach wins.

## Possible outline

1. Open with the hand quote.
2. Describe the agent: 754 grid values, then 794 with head rays.
3. Explain why a solved policy ignores a new sense.
4. Show the frozen-input canary: forced use, slower route.
5. Show the from-scratch result: zeroing the sensor collapses 60/60 to 22/60.
6. Zoom out: continual learning, new sensors, prosthetics, tool use, and embodied agents.
7. End with the open question: can the hand make the agent faster, not just different?

## Working titles

- *Growing a Hand*
- *The Hand You Don't Use*
- *You Can't Bolt On a Sense*
- *Why a Better Observation Can Make a Worse Policy*
