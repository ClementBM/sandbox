from dataclasses import dataclass, field
from typing import List, Optional, Union
from xsdata.models.datatype import XmlDate


@dataclass
class Audience:
    date_audience: Optional[XmlDate] = field(
        default=None,
        metadata={
            "name": "Date_Audience",
            "type": "Element",
        },
    )
    numero_role: Optional[Union[int, str]] = field(
        default=None,
        metadata={
            "name": "Numero_Role",
            "type": "Element",
        },
    )
    formation_jugement: Optional[str] = field(
        default=None,
        metadata={
            "name": "Formation_Jugement",
            "type": "Element",
        },
    )


@dataclass
class DonneesTechniques:
    class Meta:
        name = "Donnees_Techniques"

    identification: Optional[str] = field(
        default=None,
        metadata={
            "name": "Identification",
            "type": "Element",
            "required": True,
        },
    )
    date_mise_jour: Optional[XmlDate] = field(
        default=None,
        metadata={
            "name": "Date_Mise_Jour",
            "type": "Element",
            "required": True,
        },
    )


@dataclass
class Dossier:
    code_juridiction: Optional[str] = field(
        default=None,
        metadata={
            "name": "Code_Juridiction",
            "type": "Element",
            "required": True,
        },
    )
    nom_juridiction: Optional[str] = field(
        default=None,
        metadata={
            "name": "Nom_Juridiction",
            "type": "Element",
            "required": True,
        },
    )
    numero_dossier: Optional[int] = field(
        default=None,
        metadata={
            "name": "Numero_Dossier",
            "type": "Element",
            "required": True,
        },
    )
    date_lecture: Optional[XmlDate] = field(
        default=None,
        metadata={
            "name": "Date_Lecture",
            "type": "Element",
            "required": True,
        },
    )
    numero_ecli: Optional[str] = field(
        default=None,
        metadata={
            "name": "Numero_ECLI",
            "type": "Element",
            "required": True,
        },
    )
    avocat_requerant: Optional[str] = field(
        default=None,
        metadata={
            "name": "Avocat_Requerant",
            "type": "Element",
        },
    )
    type_decision: Optional[str] = field(
        default=None,
        metadata={
            "name": "Type_Decision",
            "type": "Element",
            "required": True,
        },
    )
    type_recours: Optional[str] = field(
        default=None,
        metadata={
            "name": "Type_Recours",
            "type": "Element",
        },
    )
    code_publication: Optional[str] = field(
        default=None,
        metadata={
            "name": "Code_Publication",
            "type": "Element",
            "required": True,
        },
    )
    solution: Optional[str] = field(
        default=None,
        metadata={
            "name": "Solution",
            "type": "Element",
        },
    )


@dataclass
class TexteIntegral:
    class Meta:
        name = "Texte_Integral"

    p: List[Union[str, int]] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "min_occurs": 1,
        },
    )


@dataclass
class Decision:
    texte_integral: Optional[TexteIntegral] = field(
        default=None,
        metadata={
            "name": "Texte_Integral",
            "type": "Element",
            "required": True,
        },
    )


@dataclass
class Document:
    no_namespace_schema_location: Optional[str] = field(
        default=None,
        metadata={
            "name": "noNamespaceSchemaLocation",
            "type": "Attribute",
            "namespace": "http://www.w3.org/2001/XMLSchema-instance",
            "required": True,
        },
    )
    donnees_techniques: Optional[DonneesTechniques] = field(
        default=None,
        metadata={
            "name": "Donnees_Techniques",
            "type": "Element",
            "required": True,
        },
    )
    dossier: Optional[Dossier] = field(
        default=None,
        metadata={
            "name": "Dossier",
            "type": "Element",
            "required": True,
        },
    )
    audience: Optional[Audience] = field(
        default=None,
        metadata={
            "name": "Audience",
            "type": "Element",
        },
    )
    decision: Optional[Decision] = field(
        default=None,
        metadata={
            "name": "Decision",
            "type": "Element",
            "required": True,
        },
    )
