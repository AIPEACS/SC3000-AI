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

# Exercise 2: The Royal Family

## Problem Statement

> queen elizabeth has four offsprings in birth order: prince charles, princess ann, prince andrew,
> prince edward.
>
> **Old rule**: the throne passes to male offspring first (in birth order), then female offspring (in birth order).
>
> **New rule**: the throne passes in birth order irrespective of gender.

---

## Part 2.1 — Old Succession Rule

### Knowledge Base

| Fact | Meaning |
|------|---------|
| `offspring(elizabeth, charles, 1)` | charles is elizabeth's 1st child |
| `offspring(elizabeth, anne, 2)` | anne is elizabeth's 2nd child |
| `offspring(elizabeth, andrew, 3)` | andrew is elizabeth's 3rd child |
| `offspring(elizabeth, edward, 4)` | edward is elizabeth's 4th child |
| `male(charles)`, `male(andrew)`, `male(edward)` | gender facts |
| `female(elizabeth)`, `female(anne)` | gender facts |

### Old Succession Rule

Two clauses — males first by birth order, then females (only when no male exists at that position):

```prolog
% First: male offspring, in birth order
successor(Parent, Successor) :-
    offspring(Parent, Successor, BirthOrder1),
    is_male(Successor),
    \+ (offspring(Parent, Other, BirthOrder2), is_male(Other), BirthOrder2 < BirthOrder1).

% Then: female offspring, in birth order (only when no male is available)
successor(Parent, Successor) :-
    offspring(Parent, Successor, BirthOrder1),
    is_female(Successor),
    \+ (offspring(Parent, Male, _), is_male(Male)),
    \+ (offspring(Parent, Other, BirthOrder2), is_female(Other), BirthOrder2 < BirthOrder1).
```

`line_of_succession` collects males sorted by birth order, then females sorted by birth order, and appends them:

```prolog
line_of_succession(elizabeth, SuccessionList) :-
    findall(BirthOrder-X, (offspring(elizabeth, X, BirthOrder), is_male(X)), MaleHeirs),
    sort(MaleHeirs, SortedMales),
    findall(BirthOrder-X, (offspring(elizabeth, X, BirthOrder), is_female(X)), FemaleHeirs),
    sort(FemaleHeirs, SortedFemales),
    pairs_values(SortedMales, Males),
    pairs_values(SortedFemales, Females),
    append(Males, Females, SuccessionList).
```

### Result

**Old line of succession**: `[charles, andrew, edward, anne]`

- charles (male, 1st born) → andrew (male, 3rd born) → edward (male, 4th born) → anne (female, 2nd born)

---

## Part 2.2 — New Succession Rule

### Necessary Changes

The new rule removes gender-based prioritisation. The `line_of_succession` rule is simplified to collect **all offspring** and sort by birth order alone:

```prolog
% Old: separate findall for males then females, append
% New: single findall for all offspring, sort by birth order
line_of_succession(elizabeth, SuccessionList) :-
    findall(BirthOrder-X, offspring(elizabeth, X, BirthOrder), AllHeirs),
    sort(AllHeirs, SortedHeirs),
    pairs_values(SortedHeirs, SuccessionList).
```

The gender facts (`male/1`, `female/1`) and helper predicates (`is_male/1`, `is_female/1`) are retained in the file but are no longer used by the succession rule. No knowledge about individuals needs to change — only the rule itself.

### Result

**New line of succession**: `[charles, anne, andrew, edward]`

- charles (1st born) → anne (2nd born) → andrew (3rd born) → edward (4th born)

### Comparison

| Position | Old rule | New rule |
|----------|----------|----------|
| 1st | charles (male, born 1st) | charles (born 1st) |
| 2nd | andrew (male, born 3rd) | anne (born 2nd) — promoted |
| 3rd | edward (male, born 4th) | andrew (born 3rd) — demoted |
| 4th | anne (female, born 2nd) | edward (born 4th) — demoted |

Anne moves from last to 2nd in line; andrew and edward each move one position back.

---

## Running the Traces

```powershell
# Windows
.\run_task.ps1 -Interactive   # select 2 → opens swipl for manual stepping
```

Output files:
- `part2/task2_1_LOS.txt` — old rule trace
- `part2/task2_2_LOS.txt` — new rule trace

