"""Model components.

`SubjectEncoder` (homogeneous GCN+pool) and `M0Strict` are the strict-single-query
analog lineage (must NOT be presented as MAGP-AHGNN). The recovered MAGP-AHGNN
heterogeneous lineage lives in `heterogeneous_magp` (SAGPool+DiffPool subject
encoder + sex-aware dual-channel TransformerConv population head); import it as
`from cr_magp_ahgnn.models.heterogeneous_magp import HeterogeneousMAGP, MAGPConfig`.
"""
from .m0_strict import M0Strict, StrictOutput
from .population_encoder import PopulationEncoder
from .subject_encoder import SubjectEncoder

__all__ = ["M0Strict", "PopulationEncoder", "StrictOutput", "SubjectEncoder"]
