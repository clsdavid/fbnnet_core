"""Shared FBN network container type(s).

Kept in their own module (rather than network_app.py/network_utils.py) to
avoid circular imports between the modules that construct these networks.
"""
from typing import Any, Dict


def _fmt_num(value: Any) -> str:
    """Mimic R's as.character() for doubles: 1.0 -> "1", 0.92 -> "0.92"."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


class FundamentalBooleanNetwork(dict):
    """
    A dict subclass holding FBN network data (genes/interactions/fixed/
    timedecay), with __str__/__repr__ matching R's
    print.FundamentalBooleanNetwork (modelling_FBN.R) output format.
    """

    def __str__(self) -> str:
        genes = self.get("genes", [])
        interactions = self.get("interactions", {})
        timedecay = self.get("timedecay", {})
        fixed = self.get("fixed", {})

        out = [
            f"Fundamental Boolean Network with  {len(genes)} genes\n",
            "Genes involved:\n",
            ", ".join(genes),
            "\n",
            "\nNetworks:",
        ]

        for gene, rules in interactions.items():
            decay = _fmt_num(timedecay.get(gene, 1))
            out.append(f"\nMultiple Transition Functions for {gene} with decay value = {decay}:\n")
            for func_name, rule in rules.items():
                out.append(f"{func_name}: {gene} = {rule.get('expression', '')}")
                if rule.get("error") is not None:
                    out.append(" (")
                    probability = rule.get("probability")
                    if probability not in (None, ""):
                        out.append(f"Confidence: {_fmt_num(probability)}")
                    out.append(", ")
                    timestep = rule.get("timestep")
                    out.append(f"TimeStep: {_fmt_num(timestep) if timestep not in (None, '') else 1}")
                    out.append(")")
                out.append("\n")

        if any(v != -1 for v in fixed.values()):
            out.append("\nKnocked-out and over-expressed genes:\n")
            for gene in genes:
                value = fixed.get(gene, -1)
                if value != -1:
                    out.append(f"{gene} = {value}\n")

        return "".join(out)

    def __repr__(self) -> str:
        return self.__str__()
