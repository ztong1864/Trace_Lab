"""Real-lab ask/tell support for TRACE.

The lab package keeps wet-lab deployment state separate from the existing
offline benchmark runner. In lab mode the system recommends candidates and
waits for measured results; it never calls a finite-pool oracle.
"""

from chem_agent_bo.lab.design_space import DesignSpace
from chem_agent_bo.lab.evidence import EvidenceCard, EvidenceStore
from chem_agent_bo.lab.project import LabProject, ProjectConfig
from chem_agent_bo.lab.service import LabBOService

__all__ = [
    "DesignSpace",
    "EvidenceCard",
    "EvidenceStore",
    "LabBOService",
    "LabProject",
    "ProjectConfig",
]
