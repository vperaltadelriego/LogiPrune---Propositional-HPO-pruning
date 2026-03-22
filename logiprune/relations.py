"""
logiprune.relations
────────────────────
Full propositional relation vector for a binarized feature pair.

Given binary vectors a, b ∈ {0,1}^n, the four tuple counts
n11, n10, n01, n00 determine which propositional formula best
characterizes the relationship between A and B.

The 10 distinguishable relations (up to classical equivalence):
──────────────────────────────────────────────────────────────
  TAUTOLOGY       : all four tuples present, no case is rare
                    (actually means: no structural rule found,
                     but confidence is uniformly high — see CONTINGENCY)

  A_IMPLIES_B     : A → B       (n10 rare)
  B_IMPLIES_A     : B → A       (n01 rare)
  BICONDITIONAL   : A ↔ B       (n10 AND n01 rare)
  A_OR_B          : A ∨ B       (n00 rare)
  A_NOR_B         : ¬A ∧ ¬B     (n11 rare — both absent together is impossible
                                  meaning: they are almost never both 0)
                    equivalently: ¬(¬A ∧ ¬B), i.e. A ∨ B almost always
  INCOMPATIBLE    : ¬(A ∧ B)    (n11 rare — A and B never coexist)
                    equivalently: A → ¬B
  A_ONLY          : A ∧ ¬B only (n01 AND n00 rare — only <1,0> and <1,1> present)
  B_ONLY          : B ∧ ¬A only
  CONTRADICTION   : no tuples (degenerate, e.g. constant column)
  CONTINGENCY     : no case is rare; true statistical independence or noise

Note: INCOMPATIBLE is as useful as BICONDITIONAL for pruning —
it means A and B activate in mutually exclusive contexts.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


# ── Relation labels ──────────────────────────────────────────────────────────

TAUTOLOGY      = 'tautology'       # ⊤ — effectively all patterns present
A_IMPLIES_B    = 'A→B'
B_IMPLIES_A    = 'B→A'
BICONDITIONAL  = 'A↔B'
A_OR_B         = 'A∨B'             # n00 rare
INCOMPATIBLE   = 'A→¬B'            # n11 rare: A and B never coexist
A_NOR_B        = '¬A∧¬B→⊥'        # n11 dominant, n00 absent (always at least one)
CONTINGENCY    = 'contingency'


@dataclass(frozen=True)
class RelationResult:
    relation:   str
    confidence: float   # max directional confidence at T*
    support:    float   # support of dominant tuple
    stability:  float   # SWTS stability score
    threshold:  float   # T* that produced this result
    n11: int; n10: int; n01: int; n00: int
    n_total: int

    @property
    def is_structural(self) -> bool:
        """True if relation is not contingency — i.e. useful for pruning."""
        return self.relation not in (CONTINGENCY, TAUTOLOGY)

    @property
    def is_redundancy(self) -> bool:
        """True if one feature can be safely removed (subject to validation)."""
        return self.relation == BICONDITIONAL

    @property
    def is_implication(self) -> bool:
        return self.relation in (A_IMPLIES_B, B_IMPLIES_A)

    @property
    def is_incompatible(self) -> bool:
        return self.relation == INCOMPATIBLE

    @property
    def coverage(self) -> float:
        """Fraction of observations explained by the dominant pattern."""
        dominant = max(self.n11, self.n10, self.n01, self.n00)
        return dominant / self.n_total if self.n_total > 0 else 0.0


# ── Core classifier ──────────────────────────────────────────────────────────

def classify_pair(a: np.ndarray,
                  b: np.ndarray,
                  epsilon: float = 0.05) -> Tuple[str, float, float]:
    """
    Given binary vectors a and b, return (relation, confidence, support).

    epsilon: fraction below which a tuple count is considered 'rare'.
    """
    n = len(a)
    if n == 0:
        return CONTINGENCY, 0.0, 0.0

    n11 = int(((a == 1) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    n01 = int(((a == 0) & (b == 1)).sum())
    n00 = int(((a == 0) & (b == 0)).sum())
    rare = epsilon * n

    # Directional confidences
    def conf_ab():  # P(B=1 | A=1)
        return n11 / (n11 + n10) if (n11 + n10) > 0 else 0.0
    def conf_ba():  # P(A=1 | B=1)
        return n11 / (n11 + n01) if (n11 + n01) > 0 else 0.0
    def conf_anb(): # P(B=0 | A=1) — A implies NOT B
        return n10 / (n11 + n10) if (n11 + n10) > 0 else 0.0

    # Classify by absent/rare tuples
    rare_10 = n10 < rare
    rare_01 = n01 < rare
    rare_11 = n11 < rare
    rare_00 = n00 < rare

    if rare_10 and rare_01:
        conf = max(conf_ab(), conf_ba())
        supp = n11 / n
        return BICONDITIONAL, conf, supp

    elif rare_10 and rare_00:
        # Only <1,1> and <0,1> — B almost always 1
        conf = conf_ba()
        supp = n11 / n
        return B_IMPLIES_A, conf, supp

    elif rare_01 and rare_00:
        # Only <1,1> and <1,0> — A almost always 1
        conf = conf_ab()
        supp = n11 / n
        return A_IMPLIES_B, conf, supp

    elif rare_10:
        conf = conf_ab()
        supp = n11 / n
        return A_IMPLIES_B, conf, supp

    elif rare_01:
        conf = conf_ba()
        supp = n11 / n
        return B_IMPLIES_A, conf, supp

    elif rare_11:
        # A and B never coexist → A → ¬B
        conf = conf_anb()
        supp = n10 / n
        return INCOMPATIBLE, conf, supp

    elif rare_00:
        # Always at least one present → A ∨ B
        conf = (n11 + n10 + n01) / n
        supp = (n11 + n10 + n01) / n
        return A_OR_B, conf, supp

    else:
        # No structural rule detectable
        conf = max(conf_ab(), conf_ba())
        supp = max(n11, n10, n01, n00) / n
        return CONTINGENCY, conf, supp
