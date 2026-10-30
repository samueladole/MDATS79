"""
Minimal biomedical extraction template used by the docling-graph builder.

Deliberately small (3 entity types, 4 edge types). The full ontology — a subset
of the BioLink Model selected to cover >=90% of the benchmark queries — is a
separate task; this exists to measure throughput and verify the provenance join.
"""

from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field


def edge(label: str, default: Any = None, **kwargs: Any) -> Any:
    """Mark a field as a graph edge. Mirrors docling-graph's own edge() helper."""
    extra = dict(kwargs.pop("json_schema_extra", {}) or {})
    extra["edge_label"] = label
    if "default_factory" in kwargs:
        return Field(default_factory=kwargs.pop("default_factory"),
                     json_schema_extra=extra, **kwargs)
    return Field(default, json_schema_extra=extra, **kwargs)


class Disease(BaseModel):
    model_config = ConfigDict(is_entity=True, graph_id_fields=["name"])
    name: str = Field(description="Disease or condition name, lowercase")


class GeneOrProtein(BaseModel):
    # Gene and protein are merged: BioASQ abstracts use the terms
    # interchangeably and a small extractor will not reliably separate them.
    model_config = ConfigDict(is_entity=True, graph_id_fields=["name"])
    name: str = Field(description="Gene or protein name/symbol, lowercase")


class ChemicalOrDrug(BaseModel):
    model_config = ConfigDict(is_entity=True, graph_id_fields=["name"])
    name: str = Field(description="Drug or chemical compound name, lowercase")
    inhibits: List[GeneOrProtein] = edge(
        "INHIBITS", default_factory=list,
        description="Proteins or genes this compound inhibits")
    treats: List[Disease] = edge(
        "TREATS", default_factory=list,
        description="Diseases this compound treats")


class BiomedicalAbstract(BaseModel):
    """One BioASQ passage. One passage == one input file == one of our chunks."""
    model_config = ConfigDict(is_entity=True, graph_id_fields=["source_id"])
    source_id: Optional[str] = Field(None, description="Leave null")
    diseases: List[Disease] = edge("MENTIONS_DISEASE", default_factory=list)
    genes: List[GeneOrProtein] = edge("MENTIONS_GENE", default_factory=list)
    compounds: List[ChemicalOrDrug] = edge("MENTIONS_COMPOUND", default_factory=list)
