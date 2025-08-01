# PremedPro AI - Hybrid Neuro-Symbolic Medical AI System

A comprehensive AI platform designed for medical education and ethical decision-making, integrating multiple open-source components with custom extensions for the medical domain.

## 🏗️ Architecture Overview

PremedPro AI is built as a hybrid neuro-symbolic system with the following layers:

### 1. **Core Layer** (Python)
- **Symbolic Reasoning**: Integration of IBM NSTK, Nucleoid, and PEIRCE
- **Neural Reasoning**: Custom quantum-inspired uncertainty models with TorchLogic and SymbolicAI
- **Hybrid Bridge**: Fuses symbolic and neural reasoning for comprehensive medical AI

### 2. **Mathematical Foundation** (Julia + Python)
- **Quantum Field Theory Analogs**: Uncertainty quantification and truth evaluation
- **Thermodynamic Entropy**: Truth and ethics evaluation using entropy principles
- **Python Integration**: Seamless Julia-Python integration via PyJulia

### 3. **Ethical Audit Layer** (Rust)
- **Consciousness Detection**: Monitors for potential AI consciousness
- **Privacy Enforcement**: Differential privacy for medical data protection
- **Audit Trail**: Traceable logs with symbolic proofs

### 4. **Middleman Layer** (Rust + Python)
- **Data Interceptor**: Secure data capture from users/APIs
- **Learning Loop**: Coordinates data processing and model training

### 5. **Orchestration Layer** (Python)
- **Agent System**: Built on OpenSSA for agentic control
- **Phase Manager**: Evolution from middleman to independent operation
- **API Endpoints**: FastAPI integration points

## 🚀 Quick Start

### Prerequisites

- **Python 3.9+** with Poetry
- **Rust 1.70+** with Cargo
- **Julia 1.9+** (optional, fallback implementations available)
- **Docker** (for containerized deployment)

### Installation

1. **Clone and Setup**
   ```bash
   git clone <your-repo-url>
   cd premedpro-ai
   chmod +x scripts/setup.sh
   ./scripts/setup.sh
   ```

2. **Python Dependencies**
   ```bash
   # Install Poetry if not already installed
   curl -sSL https://install.python-poetry.org | python3 -
   
   # Install Python dependencies
   poetry install
   ```

3. **Rust Components**
   ```bash
   cd ethical_audit
   cargo build --release
   cd ..
   ```

4. **Julia Mathematical Foundation (Optional)**
   ```bash
   # Install Julia dependencies
   julia -e 'using Pkg; Pkg.add(["DifferentialEquations", "LinearAlgebra", "Statistics", "SymbolicUtils"])'
   
   # Install PyJulia for Python integration
   poetry run pip install julia
   poetry run python -c "import julia; julia.install()"
   ```

### Basic Usage

```python
from core.hybrid_bridge import create_hybrid_reasoning_engine
from math_foundation.python_wrapper import create_math_foundation
from ethical_audit import EthicalAuditSystem

# Initialize the system
config = {
    "reasoning_mode": "adaptive",
    "neural_config": {
        "input_dim": 512,
        "output_dim": 256
    }
}

# Create reasoning engine
reasoning_engine = create_hybrid_reasoning_engine(config)

# Create mathematical foundation
math_foundation = create_math_foundation()

# Process a medical query
result = await reasoning_engine.reason(
    "What are the symptoms of myocardial infarction?",
    {"user_type": "medical_student"}
)

print(f"Response: {result.final_answer}")
print(f"Confidence: {result.confidence}")
print(f"Ethical Compliance: {result.ethical_compliance}")
```

## 📁 Project Structure
premedpro-ai/  # Root: Consolidated monorepo for Dharma Engine in Premedpro
├── .git/  # Git repo (with submodules for OSS)
├── .gitignore  # Ignore build artifacts, venvs, etc.
├── README.md  # Overview, setup, phasing (middleman to independent), new features explained
├── LICENSE  # Your license + OSS attributions
├── CREDITS.md  # List of OSS sources and modifications
├── docs/  # Architecture diagrams (e.g., via Draw.io), API specs, ethical guidelines
│   ├── architecture.md  # Detailed system map, including new features like 10th man and simulations
│   └── ethical_memory_design.md  # Docs on simulation-based ethical training (inspired by Keleth PoC)
├── config/  # Shared configs (e.g., ethical rules YAML, simulation scenarios, API keys securely referenced)
│   ├── ethical_constraints.yaml  # Seed for ethical memory (e.g., core principles for simulations)
│   ├── simulation_scenarios.yaml  # Predefined human life templates for ethical training
│   └── agent_domains.yaml  # Definitions for 10 agents' expertise and 10th man rules
├── core/  # Hybrid Neuro-Symbolic Engine (Python primary; ~40% OSS, extended for memory/sim integrations)
│   ├── symbolic/  # Symbolic reasoning layer (extended for ethical memory graphs)
│   │   ├── nstk/  # Submodule: Forked IBM NSTK (for LNNs; extend with ethical rule verification)
│   │   ├── nucleoid/  # Submodule: Forked Nucleoid (for knowledge graphs; **extend with ethical_memory_graph.py for long-term ethics storage**)
│   │   ├── peirce/  # Submodule: Forked PEIRCE (for inference loops; integrate with 10th man dissent)
│   │   └── custom_logic.py  # Extensions: Symbolic rules for query formulation and simulation analysis
│   ├── neural/  # Neural learning layer (extended for learning from queries/simulations)
│   │   ├── torchlogic/  # Submodule: Forked TorchLogic (for weighted logic; fine-tune on ethical sim data)
│   │   ├── symbolicai/  # Submodule: Forked SymbolicAI (for LLM fusion; adapt for internal narrative gen in sims)
│   │   └── custom_neural.py  # Extensions: QM-inspired uncertainty models, trained on query responses
│   └── hybrid_bridge.py  # Custom: Fuse symbolic/neural; now includes hooks for audit logs and ethical memory updates
├── math_foundation/  # Mathematical Core (Julia primary; extended for simulation metrics)
│   ├── qft_qm.jl  # Custom: QFT/QM analogs (branching sim paths for life alternatives)
│   ├── thermo_entropy.jl  # Custom: Entropy-based truth/ethics (now measures ethical alignment in sims)
│   └── python_wrapper.py  # Custom: PyJulia integration; add functions for sim entropy scoring
├── ethical_audit/  # Safety Layer (Rust primary; **expanded for further auditability and thinking traces**)
│   ├── Cargo.toml  # Rust manifest
│   ├── src/  # Rust source
│   │   ├── main.rs  # Entry for testing
│   │   ├── consciousness_detector.rs  # Custom: Extended for sim-based ethical checks
│   │   ├── privacy_enforcer.rs  # Custom: Differential privacy for sim data
│   │   ├── audit_trail.rs  # Custom: Traceable logs; **now with english_reasoning_logger.rs for plain English traces**
│   │   └── **thinking_auditor.rs**  # **New: Module for auditing agent reasoning paths, integrating 10th man logs**
│   └── py_bindings/  # Custom: PyO3 crates to expose Rust to Python (e.g., for real-time audit in sims/queries)
├── middleman/  # Data Pipeline (Mixed Rust/Python; extended for query/learning feeds)
│   ├── interceptor.rs  # Custom Rust: Secure data capture from users/APIs
│   └── learning_loop.py  # Custom Python: Pipe data to sims/queries; **now feeds into ethical memory**
├── orchestration/  # Agentic/Phased Control (Python; **expanded for 10th man, querying, and simulations**)
│   ├── openssa/  # Submodule: Forked OpenSSA (for agents; adapt for multi-agent hierarchies)
│   ├── agents/  # Custom extensions: Domain-specific agents
│   │   ├── base_agent.py  # Shared class with access to ethical memory
│   │   ├── domain_expert.py  # Polymath agents (1-9) with specialized domains
│   │   └── **tenth_man_agent.py**  # **New: Devil's Advocate logic—disagrees with consensus, draws from ethical memory**
│   │   └── multi_agent_deliberation.py  # Consensus mechanism with 10th man rule
│   ├── **external_ai_integration/**  # **New submodule: For continuous autonomous querying**
│   │   ├── api_wrappers.py  # Clients for Grok4, GPT-4, Claude4, Gemini 2.5
│   │   └── query_orchestrator.py  # Logic for generating/processing queries (e.g., ethics topics first)
│   ├── phase_manager.py  # Custom: Switch from middleman (query-heavy) to independent (sim/memory-only)
│   ├── api_endpoints.py  # Custom: FastAPI for premedpro integration
│   └── **simulation/**  # **New: For long-term ethical memory via human life simulations**
│       ├── life_simulator.py  # Core sim generator (text-based narratives, inspired by Keleth PoC)
│       ├── ethical_distiller.py  # Extracts lessons from sims into memory (entropy-based)
│       └── sim_debater.py  # Integrates 10th man—agents debate sim outcomes for ethical refinement
├── utils/  # Shared Utilities (Cross-language; expanded for new features)
│   ├── data_streams.py  # Custom: Processing user/AI inputs; **now includes sim data feeders**
│   ├── testing/  # Custom tests: Unit/integration (e.g., ethical sim scenarios, query mocks)
│   └── build_scripts/  # Scripts: e.g., build.sh to compile Rust/bind to Python
├── docker/  # Deployment (for enterprise/local runs; updated for sim/query scaling)
│   ├── Dockerfile  # Multi-stage: Python/Rust/Julia, with GPU support for neural sims
│   └── docker-compose.yml  # Services for layers (e.g., simulation as scalable microservice)
└── scripts/  # Automation (expanded for new workflows)
    ├── setup.sh  # Clone submodules, install deps
    ├── train_ethical_memory.sh  # Script to run batch simulations for ethics bootstrap
    └── **autonomous_learning.sh**  # **New: Triggers querying and sim loops for continuous learning**

## 🔧 Configuration

### Ethical Constraints

The system uses a comprehensive ethical configuration in `config/ethical_constraints.yaml`:

- **Core Principles**: Beneficence, non-maleficence, autonomy, justice
- **Privacy Protection**: Data retention limits, differential privacy parameters
- **Medical Ethics**: Professional boundaries, competence limits
- **Safety Mechanisms**: Consciousness detection, bias monitoring

### System Modes

The system supports multiple reasoning modes:

1. **Symbolic First**: Prioritizes rule-based reasoning for safety
2. **Neural First**: Uses neural networks with symbolic validation
3. **Parallel**: Executes both approaches simultaneously
4. **Adaptive**: Automatically selects the best approach based on query characteristics

## 🔒 Privacy & Security

### Data Protection
- **Differential Privacy**: Built-in privacy preservation for medical data
- **Encryption**: AES-GCM encryption for sensitive data at rest
- **Audit Trails**: Complete logging of all decisions and data access

### Ethical Safeguards
- **Consciousness Detection**: Monitors for potential AI consciousness emergence
- **Bias Detection**: Continuous monitoring for demographic and outcome bias
- **Professional Boundaries**: Strict limits on diagnostic and treatment advice

### Compliance
- **HIPAA Ready**: Designed with healthcare privacy regulations in mind
- **Explainable AI**: All decisions include reasoning chains and confidence scores
- **Right to Deletion**: Support for user data deletion requests

## 🧪 Testing

Run the test suite:

```bash
# Python tests
poetry run pytest

# Rust tests
cd ethical_audit && cargo test

# Julia tests (if available)
julia -e 'using Pkg; Pkg.test()'

# Integration tests
poetry run python -m pytest tests/integration/
```

## 📈 Development Phases

### Phase 1: Middleman (Current)
- Observe and intercept data flows
- Build knowledge graphs from interactions
- Perform basic ethical auditing

### Phase 2: Hybrid Operation
- Combine intercepted data with local inference
- Selective use of external APIs with privacy preservation
- Enhanced learning from accumulated data

### Phase 3: Independent Operation
- Fully local inference capabilities
- Complete privacy preservation
- Self-improving through local data only

## 🤝 Contributing

### Adding Open Source Components

1. Fork the component on GitHub
2. Add as git submodule: `git submodule add <fork-url> path/to/component`
3. Update `CREDITS.md` with attribution
4. Create integration wrapper in appropriate layer
5. Add tests and documentation

### Development Guidelines

- **Ethical First**: All changes must pass ethical audit checks
- **Privacy by Design**: Consider privacy implications in all features
- **Transparent AI**: Ensure all decisions are explainable
- **Medical Safety**: Medical domain features require extra scrutiny

## 📚 Documentation

- **Architecture**: Detailed system design in `docs/architecture.md`
- **API Reference**: Auto-generated from code documentation
- **Ethical Guidelines**: Comprehensive ethical framework documentation
- **Integration Guides**: Step-by-step integration tutorials

## 🔄 Integration with Open Source Projects

This project integrates the following open-source components:

- **IBM Neuro-Symbolic AI Toolkit (NSTK)**: Logical Neural Networks
- **Nucleoid**: Knowledge graph construction
- **TorchLogic**: Weighted logic operations
- **SymbolicAI**: LLM-symbolic reasoning fusion
- **PEIRCE**: Inference engines and reasoning chains
- **OpenSSA**: Agent system framework

See `CREDITS.md` for complete attribution and licensing information.

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

All integrated open-source components maintain their original licenses with proper attribution as specified in `CREDITS.md`.

## 🆘 Support

For questions, issues, or contributions:

1. **Issues**: Open a GitHub issue for bugs or feature requests
2. **Discussions**: Use GitHub discussions for questions and ideas
3. **Security**: Report security issues privately to the maintainers

## 🎯 Roadmap

- [ ] Complete integration of all OSS components
- [ ] Implement gRPC API for external access
- [ ] Add comprehensive bias detection algorithms
- [ ] Develop federated learning capabilities
- [ ] Create medical domain-specific fine-tuning pipelines
- [ ] Implement quantum computing backends for uncertainty calculation
- [ ] Add support for multimodal medical data (images, signals)

---

**⚠️ Important Medical Disclaimer**: This system is designed for educational purposes and research. It should not be used for actual medical diagnosis or treatment decisions without proper medical supervision and validation.