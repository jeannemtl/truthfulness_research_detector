# Theory of Mind Introspection - Final Results

## Achievement

✅ **Trained a dedicated Theory of Mind head (Head 6)**
✅ **88.7% validation accuracy**
✅ **Novel contribution beyond Lindsey et al.**

---

## What This Is

**Theory of Mind Introspection:** Model introspects on its internal representations of AUTHORS' epistemic states, not just its own uncertainty.

### The 6-Head Architecture

1. **Head 1:** Token truthfulness
2. **Head 2:** Sentence truthfulness  
3. **Head 3:** Uncertainty (hedging) - *linguistic patterns*
4. **Head 4:** Conflict detection
5. **Head 5:** Epistemic certainty - *model's own confidence*
6. **Head 6:** Theory of Mind - **AUTHOR'S uncertainty** ✨ NEW!

---

## Key Distinction

| Head | What It Detects | Type |
|------|----------------|------|
| **Head 3** | Hedging words ("might", "could") | Linguistic |
| **Head 5** | Model's own certainty | Self-introspection |
| **Head 6** | Author's mental state | **Theory of Mind** |

**Example:**
