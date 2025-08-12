from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple, Dict, Any

from rdkit import Chem
from rdkit import RDConfig
from rdkit import RDLogger
from rdkit.Chem import rdchem
from rdkit.Chem.rdchem import BondType
from rdkit.Chem import ChemicalFeatures
import os


SCORE_MAP = {"++": 2, "+": 1, "-": -1, "--": -2, "0": 0}


@dataclass
class Transformation:
    products: rdchem.Mol
    properties: Dict[str, Any]
    A_idx: int
    B_idx: int
    H_idx: int


@dataclass
class ScoredTransformation:
    """Bundle a transformation with per-rule labels and component scores.

    The total score is computed as a weighted sum of components and exposed via
    the `score` property; weights default to (0.5, 0.5) and can be overridden by
    MechanismProducts.
    """
    transformation: Transformation
    en_label: str
    ar_label: str
    fc_label: str
    res_label: str
    ind_label: str
    en_score: int
    ar_score: int
    fc_score: int
    res_score: int
    ind_score: int
    _weights: Tuple[float, float, float, float, float] = (0.2, 0.2, 0.2, 0.2, 0.2)


    @property
    def labels(self) -> Dict[str, str]:
        return {"EN": self.en_label, "AR": self.ar_label, "FC": self.fc_label, "RES": self.res_label, "IND": self.ind_label}

    @property
    def score(self) -> float:
        w_en, w_ar, w_fc, w_res, w_ind = self._weights
        return (w_en * self.en_score + 
                w_ar * self.ar_score + 
                w_fc * self.fc_score + 
                w_res * self.res_score + 
                w_ind * self.ind_score)

    def with_weights(self, weights: Tuple[float, float, float, float, float]) -> "ScoredTransformation":
        self._weights = weights
        return self

    def products(self) -> List[rdchem.Mol]:
        frags = Chem.GetMolFrags(
            self.transformation.products,
            asMols=True,
            sanitizeFrags=False,
        )
        return [Chem.RemoveHs(m) for m in frags]

    def meta(self) -> Dict[str, Any]:
        t = self.transformation
        return {
            "score": self.score,
            "components": {"EN": self.en_score, "AR": self.ar_score, "FC": self.fc_score, "RES": self.res_score, "IND": self.ind_score},
            "labels": self.labels,
            "A_idx": t.A_idx,
            "B_idx": t.B_idx,
            "weights": {"EN": self._weights[0], "AR": self._weights[1], "FC": self._weights[2], "RES": self._weights[3], "IND": self._weights[4]},
        }

    def products_canonical_smiles(self) -> Tuple[str, ...]:
        """Canonical SMILES (without explicit Hs) for use as a dedup key.

        Fragments are converted to canonical SMILES and sorted to be order-invariant.
        """
        smiles = [Chem.MolToSmiles(m, canonical=True) for m in self.products()]
        return tuple(sorted(smiles))

    def products_explicit_smiles(self) -> List[str]:
        """SMILES with explicit hydrogens for display."""
        out: List[str] = []
        for m in self.products():
            mh = Chem.AddHs(m)
            out.append(Chem.MolToSmiles(mh, canonical=True, allHsExplicit=True))
        return out


class MechanismProducts:
    """Holds all proton-transfer candidates.

    - best(): returns (products, meta) for the highest-scoring candidate
    - ranked(): returns a list of ScoredTransformation sorted by descending score
    - to_ranked_dicts(): returns serializable dicts including products, score, labels, indices
    """

    def __init__(
        self,
        scored: List[ScoredTransformation],
        reactants: List[rdchem.Mol],
        weights: Tuple[float, float, float, float, float] = (0.2, 0.1, 0.2, 0.3, 0.2),
    ):
        # assign weights to each scored item so that `score` reflects current policy
        self._weights = weights
        self._scored = [s.with_weights(weights) for s in scored]
        self._reactants = list(reactants)

    def best(self) -> Tuple[List[rdchem.Mol], Dict[str, Any]]:
        if not self._scored:
            return self._reactants, {
                "score": 0,
                "labels": {},
                "note": "no valid proton transfers generated",
            }
        s = max(self._scored, key=lambda x: x.score)
        return s.products(), s.meta()

    def ranked(self) -> List[ScoredTransformation]:
        return sorted(self._scored, key=lambda x: x.score, reverse=True)

    def to_ranked_dicts(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for s in self.ranked():
            out.append({
                "products": s.products(),
                "score": s.score,
                "components": s.meta()["components"],
                "labels": s.labels,
                "A_idx": s.transformation.A_idx,
                "B_idx": s.transformation.B_idx,
                "description": self.describe(s),
                "weights": s.meta()["weights"],
            })
        return out

    def ranked_unique(self) -> List[ScoredTransformation]:
        """Ranked list with duplicates removed by canonical product SMILES."""
        seen = set()
        unique: List[ScoredTransformation] = []
        for s in self.ranked():
            key = s.products_canonical_smiles()
            if key in seen:
                continue
            seen.add(key)
            unique.append(s)
        return unique

    def to_ranked_dicts_unique(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for s in self.ranked_unique():
            out.append({
                "products": s.products(),
                "score": s.score,
                "components": s.meta()["components"],
                "labels": s.labels,
                "A_idx": s.transformation.A_idx,
                "B_idx": s.transformation.B_idx,
                "description": self.describe(s),
                "weights": s.meta()["weights"],
            })
        return out

    def _combined_with_offsets(self) -> Tuple[rdchem.Mol, List[int]]:
        # Rebuild the combined molecule (with explicit Hs) and return offsets mapping
        mols_h = [Chem.AddHs(m) for m in self._reactants]
        offsets: List[int] = []
        n = 0
        for m in mols_h:
            offsets.append(n)
            n += m.GetNumAtoms()
        combined = mols_h[0]
        for m in mols_h[1:]:
            combined = Chem.CombineMols(combined, m)
        return combined, offsets

    def _map_global_to_reactant(self, idx: int, offsets: List[int]) -> Tuple[int, int]:
        # Return (reactant_index, local_atom_index)
        n_total = 0
        mols_h = [Chem.AddHs(m) for m in self._reactants]
        for i, m in enumerate(mols_h):
            start = offsets[i]
            end = start + m.GetNumAtoms()
            if start <= idx < end:
                return i, idx - start
            n_total = end
        raise IndexError(idx)

    def describe(self, s: ScoredTransformation) -> str:
        """Return a human-readable sentence: 'Lone pair on A attacks H on B'."""
        combined, offsets = self._combined_with_offsets()
        A_idx = s.transformation.A_idx
        B_idx = s.transformation.B_idx
        H_idx = s.transformation.H_idx
        atomA = combined.GetAtomWithIdx(A_idx)
        atomB = combined.GetAtomWithIdx(B_idx)
        # Map to reactants and create short names using SMILES
        a_r, _ = self._map_global_to_reactant(A_idx, offsets)
        b_r, _ = self._map_global_to_reactant(B_idx, offsets)
        reactant_names = [
            Chem.MolToSmiles(Chem.RemoveHs(m), canonical=True) for m in self._reactants
        ]
        label_str = ", ".join([f"{k}: {v}" for k, v in s.labels.items()])
        return (
            f"Lone pair on {atomA.GetSymbol()} in {reactant_names[a_r]} "
            f"attacks H on {atomB.GetSymbol()} in {reactant_names[b_r]}\n"
            f"({label_str})"
        )


@dataclass
class MechanismOptions:
    """Options to control enumeration/validation behavior.

    - include_positive_acceptors: allow positively charged atoms to be considered
      as acceptors (didactic/unrealistic enumerations like O+ of hydronium).
    - enumerate_all_acceptors: if True, treat every non-hydrogen atom as a
      potential acceptor (no feature/charge gating).
    - include_bridging_hydrogens: if True and enumerate_all_donors is enabled,
      include hydrogens with degree > 1 by pairing that H with each heavy
      neighbor.
    """

    include_positive_acceptors: bool = False
    enumerate_all_acceptors: bool = False
    include_bridging_hydrogens: bool = False


@lru_cache(maxsize=None)
def periodic_en(atomic_number: int) -> float:
    """Return Pauling electronegativity for an atomic number.

    Tries to use pymatgen's periodic table first, and falls back to the
    local `EN_PAULING` mapping if pymatgen is unavailable or returns None.
    """
    try:
        # Lazy import so pymatgen is an optional dependency
        from pymatgen.core.periodic_table import Element as PmgElement  # type: ignore
        x = PmgElement.from_Z(atomic_number).X
        if x is not None:
            return float(x)
    except Exception as e:
        # Any import/runtime issue: fall back to local table
        raise ValueError(f"Failed to import pymatgen: {e}")


def get_en(atom: rdchem.Atom) -> float:
    """Wrapper that accepts an RDKit Atom and returns Pauling EN."""
    return periodic_en(atom.GetAtomicNum())


def get_radius_angstrom(atom: rdchem.Atom) -> float:
    """
    Get the atomic radius in angstroms for a given atom.

    Prefers covalent radius; falls back to van der Waals radius if unavailable.

    Args:
        atom (rdchem.Atom): The atom whose radius is to be retrieved.

    Returns:
        float: The atomic radius in angstroms.
    """
    r = 0.0
    pt = rdchem.GetPeriodicTable()
    try:
        r = float(pt.GetRcovalent(atom.GetAtomicNum()))
    except Exception:
        pass
    if not r or r <= 0:
        try:
            r = float(pt.GetRvdw(atom.GetAtomicNum()))
        except Exception:
            r = 1.5
    return r


_FDEF = ChemicalFeatures.BuildFeatureFactory(
    os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
)


def _feature_atom_indices(mol: rdchem.Mol, family: str) -> List[int]:
    # Ensure ring info/property cache are initialized for feature detection
    try:
        mol.UpdatePropertyCache(strict=False)
    except Exception:
        pass
    try:
        Chem.GetSymmSSSR(mol)
    except Exception:
        pass
    feats = _FDEF.GetFeaturesForMol(mol)
    indices: List[int] = []
    for f in feats:
        if f.GetFamily() == family:
            indices.extend(f.GetAtomIds())
    return sorted(set(indices))


def find_acceptors_deprecated(mol: rdchem.Mol, options: MechanismOptions) -> List[int]:
    """Return indices of atoms that can act as proton acceptors.

    Uses RDKit's feature factory (Acceptor) and augments with
    any atoms carrying a negative formal charge.
    """
    acceptor_atoms = set(_feature_atom_indices(mol, "Acceptor"))
    for atom in mol.GetAtoms():
        print(f"atom: {atom.GetSymbol()} {atom.GetFormalCharge()}")
        if atom.GetFormalCharge() < 0:
            acceptor_atoms.add(atom.GetIdx())
        if options.include_positive_acceptors and atom.GetFormalCharge() > 0:
            acceptor_atoms.add(atom.GetIdx())

    # Also include neutral atoms with likely lone pairs (N/O/S) depending on charge policy
    for atom in mol.GetAtoms():
        if not options.include_positive_acceptors and atom.GetFormalCharge() > 0:
            continue
        if atom.GetAtomicNum() in (7, 8, 16):
            acceptor_atoms.add(atom.GetIdx())
    return sorted(acceptor_atoms)


def find_acceptors(mol: rdchem.Mol, options: MechanismOptions) -> List[int]:
    """Return indices of atoms that can act as proton acceptors.

    Uses RDKit's feature factory (Acceptor) and augments with
    any atoms carrying a negative formal charge.
    """
    # Exhaustive mode: all non-hydrogen atoms are potential acceptors
    if options.enumerate_all_acceptors:
        return [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 1]

    acceptor_atoms = set(_feature_atom_indices(mol, "Acceptor"))
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() < 0:
            acceptor_atoms.add(atom.GetIdx())
        if options.include_positive_acceptors and atom.GetFormalCharge() > 0:
            acceptor_atoms.add(atom.GetIdx())

    # Also include neutral atoms with likely lone pairs (N/O/S) depending on charge policy
    for atom in mol.GetAtoms():
        if not options.include_positive_acceptors and atom.GetFormalCharge() > 0:
            continue
        if atom.GetAtomicNum() in (7, 8, 16):
            acceptor_atoms.add(atom.GetIdx())
    return sorted(acceptor_atoms)


def find_acidic_hydrogens(
    mol: rdchem.Mol, options: MechanismOptions
) -> List[Tuple[int, int]]:
    """Return (donor_idx, hydrogen_idx) pairs.

    Strategy (default): enumerate heavy–H bonds (bridging H excluded).

    If options.include_bridging_hydrogens is also True, pair
    each H with each heavy neighbor (degree >= 1).
    """
    pairs_set: set[Tuple[int, int]] = set()

    for h in mol.GetAtoms():
        if h.GetAtomicNum() != 1:
            continue
        neighbors = list(h.GetNeighbors())
        if not options.include_bridging_hydrogens:
            if len(neighbors) != 1:
                continue
            heavy = neighbors[0]
            if heavy.GetAtomicNum() != 1:
                pairs_set.add((heavy.GetIdx(), h.GetIdx()))
        else:
            for nb in neighbors:
                if nb.GetAtomicNum() != 1:
                    pairs_set.add((nb.GetIdx(), h.GetIdx()))
    return sorted(pairs_set)


def apply_proton_transfer(
    mol: rdchem.Mol, A_idx: int, H_idx: int, B_idx: int
) -> rdchem.Mol | None:
    """
    Apply a proton transfer from atom B (donor) to atom A (acceptor) via hydrogen H.

    Args:
        mol (rdchem.Mol): The molecule to modify.
        A_idx (int): Index of the acceptor atom.
        H_idx (int): Index of the hydrogen atom to transfer.
        B_idx (int): Index of the donor atom.

    Returns:
        rdchem.Mol | None: The new molecule after proton transfer, or None if invalid.
    """
    # Make an editable copy
    rwm = rdchem.RWMol(mol)
    # Suppress RDKit warnings/errors during tentative edits and sanitize
    RDLogger.DisableLog('rdApp.error')
    RDLogger.DisableLog('rdApp.warning')
    try:
        if rwm.GetBondBetweenAtoms(B_idx, H_idx) is None:
            return None
        # Break B-H and form A-H
        try:
            rwm.RemoveBond(B_idx, H_idx)
        except Exception:
            return None
        try:
            rwm.AddBond(A_idx, H_idx, BondType.SINGLE)
        except Exception:
            return None

        atomA = rwm.GetAtomWithIdx(A_idx)
        atomB = rwm.GetAtomWithIdx(B_idx)
        atomA.SetFormalCharge(atomA.GetFormalCharge() + 1)
        atomB.SetFormalCharge(atomB.GetFormalCharge() - 1)

        newmol = rwm.GetMol()
        try:
            Chem.SanitizeMol(newmol)
        except Exception:
            # Discard invalid structures (e.g., valence violations)
            return None
        return newmol
    finally:
        RDLogger.EnableLog('rdApp.error')
        RDLogger.EnableLog('rdApp.warning')
    return None


def _is_resonance_stabilized(mol: rdchem.Mol, anion_idx: int) -> bool:
    """
    Checks if an anion at a given index is resonance-stabilized by being adjacent to a pi system.

    Args:
        mol (rdchem.Mol): The molecule to check.
        anion_idx (int): The index of the anion to check.

    Returns:
        bool: True if the anion is resonance-stabilized, False otherwise.
    """
    # SMARTS patterns for common resonance structures involving an anion.
    # Pattern 1: [Anion]-[Single Bond]-[Atom]=[Atom] (e.g., carboxylate, enolate)
    # Pattern 2: [Anion]-[Single Bond]-[Atom]#[Atom] (e.g., deprotonated alkyne/nitrile)
    # Pattern 3: Aromatic anion (e.g., phenoxide, where the anion is directly on the ring)
    patterns = [
        Chem.MolFromSmarts("[*-;!#1]-[!#1]=[!#1]"),  # General anion next to double bond
        Chem.MolFromSmarts("[*-;!#1]-[!#1]#[!#1]"),  # General anion next to triple bond
    ]
    
    # Check if the anion atom itself is aromatic
    anion_atom = mol.GetAtomWithIdx(anion_idx)
    if anion_atom.GetIsAromatic():
        return True

    # Check the substructure patterns
    for p in patterns:
        if mol.HasSubstructMatch(p):
            # Check if our specific anion is part of any match
            for match in mol.GetSubstructMatches(p):
                if anion_idx in match:
                    return True
    return False


def _calculate_inductive_score(mol: rdchem.Mol, anion_idx: int) -> float:
    """
    Calculates a numerical score representing the inductive pull on an anion.
    The score is the sum of (EN / distance^2) for all EWGs.

    Args:
        mol (rdchem.Mol): The molecule to check.
        anion_idx (int): The index of the anion to check.

    Returns:
        float: The inductive score.
    """
    # Define Electron Withdrawing Groups by atomic number. We include C because it is more
    # electronegative than H and can participate in inductive effects.
    EWG_ATOMIC_NUMS = {6, 7, 8, 9, 15, 16, 17, 35, 53}  # C, N, O, F, P, S, Cl, Br, I
    
    inductive_score = 0.0
    
    # We want to find the path from each EWG to the anion.
    for atom in mol.GetAtoms():
        # Rule 1: Is the atom an EWG?
        if atom.GetAtomicNum() not in EWG_ATOMIC_NUMS:
            continue
            
        # Rule 2: The EWG cannot be the anion itself.
        if atom.GetIdx() == anion_idx:
            continue
            
        # Rule 3: The EWG cannot be part of the resonance system with the anion,
        # as that's a resonance effect, not inductive. We check if the bond to the anion is non-single.
        bond_to_anion = mol.GetBondBetweenAtoms(atom.GetIdx(), anion_idx)
        if bond_to_anion and bond_to_anion.GetBondType() != BondType.SINGLE:
            continue

        # Calculate shortest path distance (in number of bonds)
        path = Chem.GetShortestPath(mol, atom.GetIdx(), anion_idx)
        bond_distance = len(path) - 1
        
        if bond_distance > 0:
            # The contribution is scaled by electronegativity and decays with square of distance.
            contribution = get_en(atom) / (bond_distance ** 2)
            inductive_score += contribution
            
    return inductive_score


def compute_transformation_properties(
    old: rdchem.Mol, new: rdchem.Mol, A_idx: int, B_idx: int
) -> Dict[str, Any]:
    """
    Compute properties of a proton transfer transformation.

    Args:
        old (rdchem.Mol): The original molecule.
        new (rdchem.Mol): The molecule after transformation.
        A_idx (int): Index of the acceptor atom.
        B_idx (int): Index of the donor atom.

    Returns:
        Dict[str, Any]: Dictionary of computed properties.
    """
    a_old = old.GetAtomWithIdx(A_idx)
    b_old = old.GetAtomWithIdx(B_idx)
    a_new = new.GetAtomWithIdx(A_idx)
    b_new = new.GetAtomWithIdx(B_idx)
    a_old_charge = int(a_old.GetFormalCharge())
    b_old_charge = int(b_old.GetFormalCharge())
    a_new_charge = int(a_new.GetFormalCharge())
    b_new_charge = int(b_new.GetFormalCharge())

    props = {
        "delta_charge_on_A": float(a_new_charge - a_old_charge),
        "delta_charge_on_B": float(b_new_charge - b_old_charge),
        "charge_A_old": a_old_charge,
        "charge_B_old": b_old_charge,
        "charge_A_new": a_new_charge,
        "charge_B_new": b_new_charge,
        "EN_A": get_en(a_old),
        "EN_B": get_en(b_old),
        "Radius_A": get_radius_angstrom(a_old),
        "Radius_B": get_radius_angstrom(b_old),
        "is_resonance_stabilized": _is_resonance_stabilized(new, B_idx),
        "inductive_score": _calculate_inductive_score(new, B_idx),
    }
    return props
    

def electronegativity_object_rule(transformation: Transformation) -> str:
    """
    Assign a qualitative label to a transformation based on electronegativity difference.

    Args:
        transformation (Transformation): The transformation to evaluate.

    Returns:
        str: One of "++", "+", "-", "--", or "0".
    """
    prop = transformation.properties
    delta_en = float(prop["EN_A"] - prop["EN_B"])
    if prop["delta_charge_on_A"] > 0:
        if delta_en < -0.5:
            return "++"
        elif delta_en < 0:
            return "+"
        elif delta_en > 0.5:
            return "--"
        else:
            return "-"
    elif prop["delta_charge_on_A"] < 0:
        if delta_en > 0.5:
            return "++"
        elif delta_en > 0:
            return "+"
        elif delta_en < -0.5:
            return "--"
        else:
            return "-"
    return "0"


def atomic_radius_object_rule(transformation: Transformation) -> str:
    """
    Assign a qualitative label to a transformation based on atomic radius difference.

    Args:
        transformation (Transformation): The transformation to evaluate.

    Returns:
        str: One of "++", "+", "-", "--", or "0".
    """
    prop = transformation.properties
    if prop["delta_charge_on_A"] == 0:
        return "0"
    delta_r = float(prop["Radius_A"] - prop["Radius_B"])  # in Angstrom
    # Thresholds adapted from 15 pm -> 0.15 Å
    if delta_r > 0.15:
        return "++"
    elif delta_r > 0.0:
        return "+"
    elif delta_r < -0.15:
        return "--"
    else:
        return "-"
    

def resonance_object_rule(transformation: Transformation) -> str:
    """
    Assigns a label based on the presence of resonance stabilization
    in the conjugate base. This is a very strong stabilizing effect.
    """
    if transformation.properties.get("is_resonance_stabilized", False):
        return "++"  # Strong stabilization
    return "0"  # No effect


def inductive_object_rule(transformation: Transformation) -> str:
    """
    Assigns a label based on the calculated inductive score.
    Higher score = more stabilization. Thresholds are empirical.
    """
    score = transformation.properties.get("inductive_score", 0.0)
    # These thresholds can be tuned based on desired sensitivity.
    if score > 2.0:  # e.g., Multiple strong EWGs like in trichloroacetic acid
        return "++"
    elif score > 0.5: # e.g., A single strong EWG nearby like in chloroacetic acid
        return "+"
    return "0"
    

def formal_charge_object_rule(transformation: Transformation) -> str:
    """Prefer neutrality on both centers after transfer using absolute charges.

    Scoring (labels map via SCORE_MAP):
    - "++": both A and B are neutral in products (charge_A_new == 0 and charge_B_new == 0)
    - "+": exactly one of A or B is neutral in products
    - "-": neither A nor B is neutral in products
    """
    props = transformation.properties
    a_neutral = (props.get("charge_A_new", 0) == 0)
    b_neutral = (props.get("charge_B_new", 0) == 0)
    if a_neutral and b_neutral:
        return "++"
    if a_neutral or b_neutral:
        return "+"
    return "-"


def score_transformation_components(t: Transformation) -> Tuple[str, int, str, int, str, int]:
    """Return per-rule labels and scores: (EN_label, EN_score, AR_label, AR_score, FC_label, FC_score)."""
    en_label = electronegativity_object_rule(t)
    ar_label = atomic_radius_object_rule(t)
    fc_label = formal_charge_object_rule(t)
    res_label = resonance_object_rule(t)
    ind_label = inductive_object_rule(t)
    en_score = SCORE_MAP[en_label]
    ar_score = SCORE_MAP[ar_label]
    fc_score = SCORE_MAP[fc_label]
    res_score = SCORE_MAP[res_label]
    ind_score = SCORE_MAP[ind_label]
    return (
        en_label, 
        en_score,
        ar_label,
        ar_score,
        fc_label,
        fc_score,
        res_label,
        res_score,
        ind_label,
        ind_score,
    )


def combine_reactants(reactants: List[rdchem.Mol]) -> rdchem.Mol:
    """
    Combine a list of reactant molecules into a single molecule with explicit hydrogens.

    Args:
        reactants (List[rdchem.Mol]): List of reactant molecules.

    Returns:
        rdchem.Mol: The combined molecule.

    Raises:
        ValueError: If any reactant is None.
    """
    if any(m is None for m in reactants):
        bad_idx = [i for i, m in enumerate(reactants) if m is None]
        raise ValueError(
            f"Invalid reactant(s) at indices {bad_idx}: one or more SMILES failed to parse"
        )
    mols_h = [Chem.AddHs(m) for m in reactants]
    combined = mols_h[0]
    for m in mols_h[1:]:
        combined = Chem.CombineMols(combined, m)
    # Initialize ring info on the combined molecule as features may require it
    try:
        combined.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(combined)
    except Exception:
        pass
    return combined


def proton_transfer_process_rule(
    reactants: List[rdchem.Mol], options: MechanismOptions
) -> List[Transformation]:
    """
    Generate all possible proton transfer transformations for a set of reactants.

    Args:
        reactants (List[rdchem.Mol]): List of reactant molecules.

    Returns:
        List[Transformation]: List of possible transformations.
    """
    combined = combine_reactants(reactants)
    acceptors = find_acceptors(combined, options)
    donors = find_acidic_hydrogens(combined, options)  # list of (B_idx, H_idx)

    candidates: List[Transformation] = []
    for A_idx in acceptors:
        for B_idx, H_idx in donors:
            if A_idx == B_idx or A_idx == H_idx:
                continue
            newmol = apply_proton_transfer(combined, A_idx, H_idx, B_idx)
            if newmol is None:
                continue
            props = compute_transformation_properties(combined, newmol, A_idx, B_idx)
            candidates.append(
                Transformation(
                    products=newmol,
                    properties=props,
                    A_idx=A_idx,
                    B_idx=B_idx,
                    H_idx=H_idx,
                )
            )
    return candidates


def find_best_transformation(
    scored: List[Tuple[Transformation, int, Dict[str, str]]]
) -> Tuple[Transformation, int, Dict[str, str]] | None:
    """
    Find the best transformation from a list of scored transformations.

    Args:
        scored (List[Tuple[Transformation, int, Dict[str, str]]]): 
            List of (Transformation, score, labels) tuples.

    Returns:
        Tuple[Transformation, int, Dict[str, str]] | None: 
            The best transformation tuple, or None if list is empty.
    """
    if not scored:
        return None
    return max(scored, key=lambda x: x[1])


def proton_transfer_predict(
    reactants: List[rdchem.Mol],
    options: MechanismOptions | None = None,
    weights: Tuple[float, float, float, float, float] = (0.2, 0.2, 0.2, 0.2, 0.2),
) -> MechanismProducts:
    """Generate all proton-transfer candidates and wrap them as MechanismProducts.

    Args:
        reactants: list of RDKit molecules
        options: behavior flags; if None, defaults are used
    """
    if options is None:
        options = MechanismOptions()
    # Step 1: Apply process rule.
    candidates = proton_transfer_process_rule(reactants, options)

    # Step 2: Apply object rules.
    scored: List[ScoredTransformation] = []
    for t in candidates:
        (en_label, en_score, ar_label, ar_score, 
         fc_label, fc_score, res_label, res_score, 
         ind_label, ind_score) = score_transformation_components(t)
        
        scored.append(
            ScoredTransformation(
                transformation=t,
                en_label=en_label, ar_label=ar_label, fc_label=fc_label,
                res_label=res_label, ind_label=ind_label,
                en_score=en_score, ar_score=ar_score, fc_score=fc_score,
                res_score=res_score, ind_score=ind_score,
            )
        )

    # Step 3: Return the results.
    return MechanismProducts(scored=scored, reactants=reactants, weights=weights)