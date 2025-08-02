"""
RDKit Integration for Medical Research AI

This module provides integration with RDKit for cheminformatics and molecular modeling,
supporting drug discovery and molecular analysis for neurodegeneration research.

RDKit is available via PyPI: pip install rdkit
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add RDKit submodule to path
rdkit_path = Path(__file__).parent.parent / "neural" / "rdkit"
if str(rdkit_path) not in sys.path:
    sys.path.insert(0, str(rdkit_path))

# Global flags for RDKit availability - will be set on first use
RDKIT_AVAILABLE = None
RDKIT_INITIALIZED = False


class RDKitIntegration:
    """
    Integration wrapper for RDKit (Cheminformatics Toolkit).
    
    RDKit provides cheminformatics and molecular modeling capabilities for drug discovery,
    molecular analysis, and chemical structure manipulation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize RDKit integration.
        
        Args:
            config: Configuration dictionary with RDKit settings
        """
        self.config = config or {}
        self.mol_drawer = None
        self.fingerprint_generator = None
        self.descriptor_calculator = None
        self._rdkit_components = {}
        
        # Don't initialize anything at startup - use lazy loading
        logger.info("RDKit integration initialized with lazy loading")
    
    def _check_rdkit_availability(self) -> bool:
        """Check if RDKit is available and initialize if needed."""
        global RDKIT_AVAILABLE, RDKIT_INITIALIZED
        
        if RDKIT_AVAILABLE is None:
            try:
                # Try to import RDKit components only when needed
                from rdkit import Chem
                from rdkit.Chem import AllChem, Descriptors, Draw, rdMolDescriptors
                from rdkit.Chem.Draw import rdMolDraw2D
                from rdkit.Chem.Fingerprints import FingerprintMols
                from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
                
                # Store components for later use
                self._rdkit_components = {
                    'Chem': Chem,
                    'AllChem': AllChem,
                    'Descriptors': Descriptors,
                    'Draw': Draw,
                    'rdMolDescriptors': rdMolDescriptors,
                    'rdMolDraw2D': rdMolDraw2D,
                    'FingerprintMols': FingerprintMols,
                    'GetMorganFingerprintAsBitVect': GetMorganFingerprintAsBitVect
                }
                
                RDKIT_AVAILABLE = True
                logger.info("RDKit components loaded successfully")
                
            except ImportError as e:
                RDKIT_AVAILABLE = False
                logger.warning(f"RDKit not available: {e}")
                logger.info("Install with: pip install rdkit")
        
        return RDKIT_AVAILABLE
    
    def _initialize_rdkit_systems(self) -> None:
        """Initialize RDKit systems and components - called only when needed."""
        global RDKIT_INITIALIZED
        
        if RDKIT_INITIALIZED:
            return
            
        try:
            if not self._check_rdkit_availability():
                return
                
            # Initialize molecular drawer
            self._initialize_molecular_drawer()
            
            # Initialize fingerprint generator
            self._initialize_fingerprint_generator()
            
            # Initialize descriptor calculator
            self._initialize_descriptor_calculator()
            
            RDKIT_INITIALIZED = True
            logger.info("RDKit systems initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing RDKit systems: {e}")
    
    def _initialize_molecular_drawer(self) -> None:
        """Initialize molecular drawing components."""
        try:
            rdMolDraw2D = self._rdkit_components['rdMolDraw2D']
            self.mol_drawer = rdMolDraw2D.MolDraw2DCairo(300, 300)
            logger.info("RDKit molecular drawer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing molecular drawer: {e}")
    
    def _initialize_fingerprint_generator(self) -> None:
        """Initialize fingerprint generation components."""
        try:
            GetMorganFingerprintAsBitVect = self._rdkit_components['GetMorganFingerprintAsBitVect']
            self.fingerprint_generator = GetMorganFingerprintAsBitVect
            logger.info("RDKit fingerprint generator initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing fingerprint generator: {e}")
    
    def _initialize_descriptor_calculator(self) -> None:
        """Initialize molecular descriptor calculation components."""
        try:
            Descriptors = self._rdkit_components['Descriptors']
            self.descriptor_calculator = Descriptors
            logger.info("RDKit descriptor calculator initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing descriptor calculator: {e}")
    
    def parse_molecular_structure(self, 
                                structure_input: str,
                                input_type: str = "smiles") -> Dict[str, Any]:
        """
        Parse molecular structure from various input formats.
        
        Args:
            structure_input: Molecular structure input (SMILES, SMARTS, etc.)
            input_type: Type of input (smiles, smarts, mol_block)
            
        Returns:
            Dictionary containing parsed molecular information
        """
        # Initialize RDKit only when this method is called
        if not self._check_rdkit_availability():
            return self._mock_structure_parsing(structure_input, input_type)
        
        try:
            # Initialize systems on first use
            if not RDKIT_INITIALIZED:
                self._initialize_rdkit_systems()
            
            Chem = self._rdkit_components['Chem']
            
            # Parse based on input type
            if input_type == "smiles":
                mol = Chem.MolFromSmiles(structure_input)
            elif input_type == "smarts":
                mol = Chem.MolFromSmarts(structure_input)
            elif input_type == "mol_block":
                mol = Chem.MolFromMolBlock(structure_input)
            else:
                mol = Chem.MolFromSmiles(structure_input)  # Default to SMILES
            
            if mol is None:
                return self._mock_structure_parsing(structure_input, input_type)
            
            # Calculate molecular properties
            molecular_weight = Chem.Descriptors.ExactMolWt(mol)
            logp = Chem.Descriptors.MolLogP(mol)
            hbd = Chem.Descriptors.NumHDonors(mol)
            hba = Chem.Descriptors.NumHAcceptors(mol)
            tpsa = Chem.Descriptors.TPSA(mol)
            
            # Generate SMILES representation
            smiles = Chem.MolToSmiles(mol)
            
            # Calculate molecular formula
            formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
            
            return {
                "structure_input": structure_input,
                "input_type": input_type,
                "status": "completed",
                "molecular_properties": {
                    "molecular_weight": molecular_weight,
                    "logp": logp,
                    "hydrogen_bond_donors": hbd,
                    "hydrogen_bond_acceptors": hba,
                    "topological_polar_surface_area": tpsa,
                    "molecular_formula": formula
                },
                "smiles": smiles,
                "atom_count": mol.GetNumAtoms(),
                "bond_count": mol.GetNumBonds(),
                "metadata": {
                    "model": "RDKit",
                    "parsing_method": input_type
                }
            }
            
        except Exception as e:
            logger.error(f"Error parsing molecular structure: {e}")
            return self._mock_structure_parsing(structure_input, input_type)
    
    def calculate_molecular_descriptors(self, 
                                     smiles: str,
                                     descriptor_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculate molecular descriptors for drug-like properties.
        
        Args:
            smiles: SMILES representation of the molecule
            descriptor_types: List of descriptor types to calculate
            
        Returns:
            Dictionary containing calculated molecular descriptors
        """
        # Initialize RDKit only when this method is called
        if not self._check_rdkit_availability():
            return self._mock_descriptor_calculation(smiles, descriptor_types)
        
        try:
            # Initialize systems on first use
            if not RDKIT_INITIALIZED:
                self._initialize_rdkit_systems()
            
            Chem = self._rdkit_components['Chem']
            Descriptors = self._rdkit_components['Descriptors']
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return self._mock_descriptor_calculation(smiles, descriptor_types)
            
            # Calculate descriptors
            descriptors = {}
            
            # Lipinski's Rule of Five
            descriptors["lipinski"] = {
                "molecular_weight": Descriptors.ExactMolWt(mol),
                "logp": Descriptors.MolLogP(mol),
                "hydrogen_bond_donors": Descriptors.NumHDonors(mol),
                "hydrogen_bond_acceptors": Descriptors.NumHAcceptors(mol)
            }
            
            # Additional drug-like properties
            descriptors["drug_like"] = {
                "topological_polar_surface_area": Descriptors.TPSA(mol),
                "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
                "aromatic_rings": Descriptors.NumAromaticRings(mol),
                "saturated_rings": Descriptors.NumSaturatedRings(mol),
                "fraction_csp3": Descriptors.FractionCsp3(mol)
            }
            
            # Pharmacokinetic properties
            descriptors["pharmacokinetic"] = {
                "bioavailability_score": Descriptors.BioavailabilityScore(mol),
                "synthetic_accessibility": Descriptors.SAScore(mol),
                "natural_product_likeness": Descriptors.NPLikeness(mol)
            }
            
            return {
                "smiles": smiles,
                "status": "completed",
                "descriptors": descriptors,
                "metadata": {
                    "model": "RDKit",
                    "descriptor_types": descriptor_types or ["all"]
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating molecular descriptors: {e}")
            return self._mock_descriptor_calculation(smiles, descriptor_types)
    
    def generate_molecular_fingerprints(self, 
                                     smiles: str,
                                     fingerprint_type: str = "morgan") -> Dict[str, Any]:
        """
        Generate molecular fingerprints for similarity analysis.
        
        Args:
            smiles: SMILES representation of the molecule
            fingerprint_type: Type of fingerprint (morgan, rdkit, atom_pair)
            
        Returns:
            Dictionary containing molecular fingerprints
        """
        # Initialize RDKit only when this method is called
        if not self._check_rdkit_availability():
            return self._mock_fingerprint_generation(smiles, fingerprint_type)
        
        try:
            # Initialize systems on first use
            if not RDKIT_INITIALIZED:
                self._initialize_rdkit_systems()
            
            Chem = self._rdkit_components['Chem']
            AllChem = self._rdkit_components['AllChem']
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return self._mock_fingerprint_generation(smiles, fingerprint_type)
            
            # Generate fingerprints based on type
            if fingerprint_type == "morgan":
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            elif fingerprint_type == "rdkit":
                fp = Chem.RDKFingerprint(mol)
            elif fingerprint_type == "atom_pair":
                fp = AllChem.GetAtomPairFingerprintAsBitVect(mol)
            else:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            
            # Convert to list for easier handling
            fingerprint_list = list(fp.GetOnBits())
            
            return {
                "smiles": smiles,
                "fingerprint_type": fingerprint_type,
                "status": "completed",
                "fingerprint": fingerprint_list,
                "fingerprint_length": len(fp),
                "on_bits": len(fingerprint_list),
                "metadata": {
                    "model": "RDKit",
                    "fingerprint_method": fingerprint_type
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating molecular fingerprints: {e}")
            return self._mock_fingerprint_generation(smiles, fingerprint_type)
    
    def analyze_drug_likeness(self, 
                            smiles: str,
                            analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Analyze drug-likeness properties of a molecule.
        
        Args:
            smiles: SMILES representation of the molecule
            analysis_type: Type of analysis (lipinski, comprehensive, custom)
            
        Returns:
            Dictionary containing drug-likeness analysis
        """
        # Initialize RDKit only when this method is called
        if not self._check_rdkit_availability():
            return self._mock_drug_likeness_analysis(smiles, analysis_type)
        
        try:
            # Initialize systems on first use
            if not RDKIT_INITIALIZED:
                self._initialize_rdkit_systems()
            
            Chem = self._rdkit_components['Chem']
            Descriptors = self._rdkit_components['Descriptors']
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return self._mock_drug_likeness_analysis(smiles, analysis_type)
            
            # Calculate properties
            mw = Descriptors.ExactMolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            tpsa = Descriptors.TPSA(mol)
            rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            
            # Lipinski's Rule of Five analysis
            lipinski_violations = 0
            lipinski_rules = {
                "molecular_weight": {"value": mw, "limit": 500, "violation": mw > 500},
                "logp": {"value": logp, "limit": 5, "violation": logp > 5},
                "hydrogen_bond_donors": {"value": hbd, "limit": 5, "violation": hbd > 5},
                "hydrogen_bond_acceptors": {"value": hba, "limit": 10, "violation": hba > 10}
            }
            
            for rule, data in lipinski_rules.items():
                if data["violation"]:
                    lipinski_violations += 1
            
            # Drug-likeness score
            drug_likeness_score = max(0, 5 - lipinski_violations) / 5
            
            return {
                "smiles": smiles,
                "analysis_type": analysis_type,
                "status": "completed",
                "lipinski_rules": lipinski_rules,
                "lipinski_violations": lipinski_violations,
                "drug_likeness_score": drug_likeness_score,
                "additional_properties": {
                    "topological_polar_surface_area": tpsa,
                    "rotatable_bonds": rotatable_bonds,
                    "recommendation": "Good drug candidate" if lipinski_violations <= 1 else "Needs optimization"
                },
                "metadata": {
                    "model": "RDKit",
                    "analysis_method": analysis_type
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing drug-likeness: {e}")
            return self._mock_drug_likeness_analysis(smiles, analysis_type)
    
    # Mock implementations for when RDKit is not available
    def _mock_structure_parsing(self, structure_input: str, input_type: str) -> Dict[str, Any]:
        """Mock implementation for structure parsing."""
        return {
            "structure_input": structure_input,
            "input_type": input_type,
            "status": "mock_completed",
            "molecular_properties": {"mock_properties": "RDKit not available"},
            "metadata": {"model": "mock", "parsing_method": "mock"}
        }
    
    def _mock_descriptor_calculation(self, smiles: str, descriptor_types: Optional[List[str]]) -> Dict[str, Any]:
        """Mock implementation for descriptor calculation."""
        return {
            "smiles": smiles,
            "status": "mock_completed",
            "descriptors": {"mock_descriptors": "RDKit not available"},
            "metadata": {"model": "mock", "descriptor_types": "mock"}
        }
    
    def _mock_fingerprint_generation(self, smiles: str, fingerprint_type: str) -> Dict[str, Any]:
        """Mock implementation for fingerprint generation."""
        return {
            "smiles": smiles,
            "fingerprint_type": fingerprint_type,
            "status": "mock_completed",
            "fingerprint": [1, 2, 3, 4, 5],  # Mock fingerprint
            "metadata": {"model": "mock", "fingerprint_method": "mock"}
        }
    
    def _mock_drug_likeness_analysis(self, smiles: str, analysis_type: str) -> Dict[str, Any]:
        """Mock implementation for drug-likeness analysis."""
        return {
            "smiles": smiles,
            "analysis_type": analysis_type,
            "status": "mock_completed",
            "lipinski_rules": {"mock_rules": "RDKit not available"},
            "drug_likeness_score": 0.5,
            "metadata": {"model": "mock", "analysis_method": "mock"}
        }


# Example usage and testing
def test_rdkit_integration():
    """Test the RDKit integration."""
    config = {
        "fingerprint_size": 2048,
        "descriptor_set": "comprehensive"
    }
    
    rdkit_integration = RDKitIntegration(config)
    
    # Test structure parsing
    structure_result = rdkit_integration.parse_molecular_structure(
        "CC(=O)OC1=CC=CC=C1C(=O)O", "smiles"
    )
    print(f"Structure Parsing: {structure_result['status']}")
    
    # Test descriptor calculation
    descriptor_result = rdkit_integration.calculate_molecular_descriptors(
        "CC(=O)OC1=CC=CC=C1C(=O)O"
    )
    print(f"Descriptor Calculation: {descriptor_result['status']}")
    
    # Test drug-likeness analysis
    drug_likeness_result = rdkit_integration.analyze_drug_likeness(
        "CC(=O)OC1=CC=CC=C1C(=O)O"
    )
    print(f"Drug-Likeness Analysis: {drug_likeness_result['status']}")


if __name__ == "__main__":
    test_rdkit_integration() 