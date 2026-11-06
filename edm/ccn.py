"""CCN: CC plus a uniform noise component.

The G - 1 non-noise components share one λ; the noise component (last index) has λ = 0,
i.e. f_0(s) = v^{-T}. The precision structure is inherited from CC with K = G - 1, which
matches Table B2's CCN step (sums over non-noise components only).
"""
from .cc import CCModel
from .types import EDMType


class CCNModel(CCModel):
    edm_type = EDMType.CCN
    has_noise = True