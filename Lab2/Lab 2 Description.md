# Lab 2 Description

# Overview

Lab 2 uses **SWI-Prolog** as a logic programming tool to encode knowledge bases and run automated proofs.  
The two exercises cover:
- **Exercise 1** — encoding a real-world scenario as First Order Logic (FOL) and proving a conclusion
- **Exercise 2** — modelling family relationships and succession rules, then querying them

---

# Exercise 1: The Smart Phone Rivalry

## Problem Statement

> sumsum, a competitor of appy, developed some nice smart phone technology called galactica-s3,
> all of which was stolen by stevey, who is a boss of appy. It is unethical for a boss to steal
> business from rival companies. A competitor is a rival. Smart phone technology is business.

**Goal**: Prove that *stevey is unethical*.

---

## Part 1.1 — First Order Logic (FOL)

The eight statements (five ground facts + three universally quantified rules):

| # | FOL | Natural language |
|---|-----|-----------------|
| 1 | `Competitor(sumsum, appy)` | sumsum is a competitor of appy |
| 2 | `Developed(sumsum, galactica_s3)` | sumsum developed galactica_s3 |
| 3 | `Smartphone(galactica_s3)` | galactica_s3 is smart phone technology |
| 4 | `Stole(stevey, galactica_s3)` | stevey stole galactica_s3 |
| 5 | `Boss(stevey, appy)` | stevey is a boss of appy |
| 6 | `∀x∀y [ Competitor(x,y) → Rival(x,y) ]` | a competitor is a rival |
| 7 | `∀x [ Smartphone(x) → Business(x) ]` | smart phone technology is business |
| 8 | `∀x∀y∀z∀w [ Boss(x,y) ∧ Stole(x,z) ∧ Business(z) ∧ Developed(w,z) ∧ Rival(w,y) → Unethical(x) ]` | it is unethical for a boss to steal business from rival companies |

---

## Part 1.2 — Prolog Knowledge Base

FOL statements map directly to Prolog clauses:

| FOL | Prolog |
|-----|--------|
| Ground facts (1–5) | Five unit clauses (facts) |
| Rule 6 | `rival(X,Y) :- competitor(X,Y).` |
| Rule 7 | `business(X) :- smartphone(X).` |
| Rule 8 | `unethical(X) :- boss(X,Y), stole(X,Z), business(Z), developed(W,Z), rival(W,Y).` |

**File**: `part1/task.pl`

---

## Part 1.3 — Proof Trace

**Query**: `?- unethical(stevey).`

Prolog proves this by backward chaining through rule 8:

```
Call: unethical(stevey)
  Call: boss(stevey, Y)          → Y = appy
  Call: stole(stevey, Z)         → Z = galactica_s3
  Call: business(galactica_s3)
    Call: smartphone(galactica_s3)  → true
  Call: developed(W, galactica_s3) → W = sumsum
  Call: rival(sumsum, appy)
    Call: competitor(sumsum, appy)  → true
→ true.  stevey is unethical ✓
```

**Run the trace**:
```powershell
# Windows — auto trace, output saved to part1/trace_output.txt
.\run_task.ps1

# Interactive step-by-step
.\run_task.ps1 -Interactive
# then inside swipl: trace.  →  unethical(stevey).
```

---

# Exercise 2: The Royal Family (Coming Soon)

## Problem Statement

> queen elizabeth has four offsprings in birth order: prince charles, princess ann, prince andrew,
> prince edward. The **old** succession rule passes the throne to male offspring first (in birth order),
> then female offspring (in birth order). The **new** succession rule passes the throne in birth order
> regardless of gender.

**Goals**:
1. Define relations + old succession rule; determine the old succession order.
2. Modify the knowledge base for the new succession rule; determine the new succession order.
