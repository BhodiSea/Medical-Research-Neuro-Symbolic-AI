"""
Real-World Data Connectors
Interface with external medical research databases and APIs
"""

import asyncio
import aiohttp
import json
from typing import Dict, List, Any, Optional
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseDataConnector(ABC):
    """Base class for all data connectors"""
    
    def __init__(self, api_key: Optional[str] = None, rate_limit: float = 1.0):
        self.api_key = api_key
        self.rate_limit = rate_limit  # requests per second
        self._last_request_time = 0
        
    async def _rate_limit_request(self):
        """Implement rate limiting"""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < self.rate_limit:
            await asyncio.sleep(self.rate_limit - time_since_last)
        self._last_request_time = asyncio.get_event_loop().time()

class PubMedConnector(BaseDataConnector):
    """Connector for NCBI PubMed literature database"""
    
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    
    def __init__(self, email: str, api_key: Optional[str] = None):
        super().__init__(api_key)
        self.email = email
        
    async def search_literature(self, query: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """Search PubMed for literature"""
        await self._rate_limit_request()
        
        try:
            # Step 1: Search for PMIDs
            search_params = {
                'db': 'pubmed',
                'term': query,
                'retmax': max_results,
                'retmode': 'json',
                'email': self.email
            }
            
            if self.api_key:
                search_params['api_key'] = self.api_key
                
            async with aiohttp.ClientSession() as session:
                # Search for article IDs
                async with session.get(f"{self.BASE_URL}esearch.fcgi", params=search_params) as response:
                    search_data = await response.json()
                    
                pmids = search_data.get('esearchresult', {}).get('idlist', [])
                
                if not pmids:
                    return []
                
                # Step 2: Fetch article details
                fetch_params = {
                    'db': 'pubmed',
                    'id': ','.join(pmids),
                    'retmode': 'json',
                    'email': self.email
                }
                
                if self.api_key:
                    fetch_params['api_key'] = self.api_key
                
                async with session.get(f"{self.BASE_URL}esummary.fcgi", params=fetch_params) as response:
                    fetch_data = await response.json()
                    
                articles = []
                for pmid in pmids:
                    if pmid in fetch_data.get('result', {}):
                        article_data = fetch_data['result'][pmid]
                        articles.append({
                            'pmid': pmid,
                            'title': article_data.get('title', ''),
                            'authors': [author.get('name', '') for author in article_data.get('authors', [])],
                            'journal': article_data.get('source', ''),
                            'pub_date': article_data.get('pubdate', ''),
                            'doi': article_data.get('elocationid', ''),
                            'abstract': article_data.get('abstract', '')
                        })
                        
                return articles
                
        except Exception as e:
            logger.error(f"Error searching PubMed: {e}")
            return []
    
    async def fetch_contradictory_evidence(self, hypothesis: str) -> List[Dict[str, Any]]:
        """Search for studies that contradict a given hypothesis"""
        # Create search query for contradictory evidence
        contradiction_terms = ["NOT", "controversy", "conflicting", "opposing", "alternative"]
        search_query = f"({hypothesis}) AND ({' OR '.join(contradiction_terms)})"
        
        return await self.search_literature(search_query, max_results=10)

class PubChemConnector(BaseDataConnector):
    """Connector for NCBI PubChem compound database"""
    
    BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
    
    async def search_compounds(self, query: str) -> List[Dict[str, Any]]:
        """Search PubChem for chemical compounds"""
        await self._rate_limit_request()
        
        try:
            async with aiohttp.ClientSession() as session:
                # Search for compound CIDs
                search_url = f"{self.BASE_URL}compound/name/{query}/cids/JSON"
                async with session.get(search_url) as response:
                    search_data = await response.json()
                    
                cids = search_data.get('IdentifierList', {}).get('CID', [])[:10]  # Limit to 10
                
                compounds = []
                for cid in cids:
                    # Get compound properties
                    props_url = f"{self.BASE_URL}compound/cid/{cid}/property/MolecularFormula,MolecularWeight,CanonicalSMILES/JSON"
                    async with session.get(props_url) as props_response:
                        props_data = await props_response.json()
                        
                        for prop in props_data.get('PropertyTable', {}).get('Properties', []):
                            compounds.append({
                                'cid': cid,
                                'molecular_formula': prop.get('MolecularFormula', ''),
                                'molecular_weight': prop.get('MolecularWeight', 0),
                                'smiles': prop.get('CanonicalSMILES', ''),
                                'query_term': query
                            })
                            
                return compounds
                
        except Exception as e:
            logger.error(f"Error searching PubChem: {e}")
            return []

class NCBIGenomicsConnector(BaseDataConnector):
    """Connector for NCBI genomics databases"""
    
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    
    def __init__(self, email: str, api_key: Optional[str] = None):
        super().__init__(api_key)
        self.email = email
        
    async def search_gene_variants(self, gene_names: List[str]) -> Dict[str, Any]:
        """Search for genetic variants in specified genes"""
        await self._rate_limit_request()
        
        try:
            variants_data = {}
            
            async with aiohttp.ClientSession() as session:
                for gene in gene_names:
                    search_params = {
                        'db': 'clinvar',
                        'term': f"{gene}[gene]",
                        'retmax': 50,
                        'retmode': 'json',
                        'email': self.email
                    }
                    
                    if self.api_key:
                        search_params['api_key'] = self.api_key
                    
                    async with session.get(f"{self.BASE_URL}esearch.fcgi", params=search_params) as response:
                        search_data = await response.json()
                        
                    variant_ids = search_data.get('esearchresult', {}).get('idlist', [])
                    variants_data[gene] = {
                        'variant_count': len(variant_ids),
                        'variant_ids': variant_ids[:10],  # Limit to first 10
                        'search_query': f"{gene}[gene]"
                    }
                    
            return variants_data
            
        except Exception as e:
            logger.error(f"Error searching NCBI genomics: {e}")
            return {}

class ClinicalTrialsConnector(BaseDataConnector):
    """Connector for ClinicalTrials.gov database"""
    
    BASE_URL = "https://clinicaltrials.gov/api/query/"
    
    async def search_trials(self, condition: str, status: str = "recruiting") -> List[Dict[str, Any]]:
        """Search for clinical trials"""
        await self._rate_limit_request()
        
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    'expr': condition,
                    'rnk': 20,  # Max 20 results
                    'fmt': 'json'
                }
                
                async with session.get(f"{self.BASE_URL}study_fields", params=params) as response:
                    data = await response.json()
                    
                trials = []
                for study in data.get('StudyFieldsResponse', {}).get('StudyFields', []):
                    trial_data = {
                        'nct_id': study.get('NCTId', [''])[0],
                        'title': study.get('BriefTitle', [''])[0],
                        'status': study.get('OverallStatus', [''])[0],
                        'phase': study.get('Phase', [''])[0],
                        'condition': study.get('Condition', []),
                        'intervention': study.get('InterventionName', []),
                        'sponsor': study.get('LeadSponsorName', [''])[0]
                    }
                    trials.append(trial_data)
                    
                return trials
                
        except Exception as e:
            logger.error(f"Error searching clinical trials: {e}")
            return []

class ProteinDataBankConnector(BaseDataConnector):
    """Connector for Protein Data Bank (PDB)"""
    
    BASE_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
    
    async def search_protein_structures(self, protein_name: str) -> List[Dict[str, Any]]:
        """Search for protein structures in PDB"""
        await self._rate_limit_request()
        
        try:
            query = {
                "query": {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "struct.title",
                        "operator": "contains_phrase",
                        "value": protein_name
                    }
                },
                "return_type": "entry",
                "request_options": {
                    "paginate": {
                        "start": 0,
                        "rows": 10
                    }
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.BASE_URL, json=query) as response:
                    data = await response.json()
                    
                structures = []
                for entry in data.get('result_set', []):
                    pdb_id = entry.get('identifier')
                    if pdb_id:
                        structures.append({
                            'pdb_id': pdb_id,
                            'protein_name': protein_name,
                            'url': f"https://www.rcsb.org/structure/{pdb_id}"
                        })
                        
                return structures
                
        except Exception as e:
            logger.error(f"Error searching PDB: {e}")
            return []

class RealWorldDataConnector:
    """Main connector class that orchestrates all data sources"""
    
    def __init__(self, email: str, ncbi_api_key: Optional[str] = None):
        self.pubmed = PubMedConnector(email, ncbi_api_key)
        self.pubchem = PubChemConnector()
        self.ncbi_genomics = NCBIGenomicsConnector(email, ncbi_api_key)
        self.clinical_trials = ClinicalTrialsConnector()
        self.pdb = ProteinDataBankConnector()
        
    async def fetch_pubmed_literature(self, query: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """Fetch literature from PubMed"""
        return await self.pubmed.search_literature(query, max_results)
        
    async def fetch_pubmed_contradictory(self, hypothesis: str) -> List[Dict[str, Any]]:
        """Fetch contradictory evidence from PubMed"""
        return await self.pubmed.fetch_contradictory_evidence(hypothesis)
        
    async def query_pubchem_compounds(self, molecular_target: str) -> List[Dict[str, Any]]:
        """Query PubChem for compounds"""
        return await self.pubchem.search_compounds(molecular_target)
        
    async def access_ncbi_genomics(self, gene_set: List[str]) -> Dict[str, Any]:
        """Access NCBI genomics data"""
        return await self.ncbi_genomics.search_gene_variants(gene_set)
        
    async def fetch_clinical_trials_data(self, condition: str) -> List[Dict[str, Any]]:
        """Fetch clinical trials data"""
        return await self.clinical_trials.search_trials(condition)
        
    async def fetch_pdb_structures(self, protein_name: str) -> List[Dict[str, Any]]:
        """Fetch protein structures from PDB"""
        return await self.pdb.search_protein_structures(protein_name)
        
    async def comprehensive_research_query(self, topic: str) -> Dict[str, Any]:
        """Perform comprehensive research query across all sources"""
        try:
            # Execute queries in parallel
            literature_task = self.fetch_pubmed_literature(topic)
            compounds_task = self.query_pubchem_compounds(topic)
            trials_task = self.fetch_clinical_trials_data(topic)
            structures_task = self.fetch_pdb_structures(topic)
            
            literature, compounds, trials, structures = await asyncio.gather(
                literature_task, compounds_task, trials_task, structures_task,
                return_exceptions=True
            )
            
            return {
                'topic': topic,
                'literature': literature if not isinstance(literature, Exception) else [],
                'compounds': compounds if not isinstance(compounds, Exception) else [],
                'clinical_trials': trials if not isinstance(trials, Exception) else [],
                'protein_structures': structures if not isinstance(structures, Exception) else [],
                'query_timestamp': asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive research query: {e}")
            return {
                'topic': topic,
                'error': str(e),
                'literature': [],
                'compounds': [],
                'clinical_trials': [],
                'protein_structures': []
            }

# Factory function
def create_real_world_data_connector(email: str, ncbi_api_key: Optional[str] = None) -> RealWorldDataConnector:
    """Factory function to create RealWorldDataConnector"""
    return RealWorldDataConnector(email, ncbi_api_key)