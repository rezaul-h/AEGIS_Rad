from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml
@dataclass(frozen=True)
class Region:
    id:int; name:str; side:str='none'; family:str='other'; xy:tuple[float,float]=(0.5,0.5)
@dataclass(frozen=True)
class Finding:
    id:int; name:str; aliases:tuple[str,...]=()
class Ontology:
    def __init__(self,regions,findings):
        self.regions=regions; self.findings=findings; self.region_to_id={x.name:x.id for x in regions}; self.finding_to_id={x.name:x.id for x in findings}
        self.finding_alias={}
        for f in findings:
            self.finding_alias[f.name.lower()]=f.name
            for a in f.aliases:self.finding_alias[a.lower()]=f.name
    @classmethod
    def from_yaml(cls,regions_path,findings_path):
        rdata=yaml.safe_load(Path(regions_path).read_text()); fdata=yaml.safe_load(Path(findings_path).read_text())
        regions=[Region(id=i,**r) for i,r in enumerate(rdata['regions'])]
        findings=[Finding(id=i,name=f['name'],aliases=tuple(f.get('aliases',[]))) for i,f in enumerate(fdata['findings'])]
        return cls(regions,findings)
