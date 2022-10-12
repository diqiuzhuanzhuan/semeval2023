# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com


from dataclasses import dataclass
from typing import Dict


@dataclass
class ArgumentItem:
    """_summary_
    """
    argument_id: str
    conclusion: str
    stance: str
    premise: str

    @classmethod
    def from_dict(cls, json_dict: Dict):
        return ArgumentItem(
            argument_id=json_dict['Argument ID'],
            conclusion=json_dict['Conclusion'],
            stance=json_dict['Stance'],
            premise=json_dict['Premise']
        )
