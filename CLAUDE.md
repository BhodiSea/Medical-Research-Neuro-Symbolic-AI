# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Python Environment Setup
```bash
# Install Python dependencies
pip install -r requirements-api.txt

# Or use pyproject.toml with pip
pip install -e .

# For development with all optional dependencies
pip install -e ".[dev,testing,julia]"
```

### Running the Application
```bash
# Start the FastAPI server
python run_api.py

# Or run directly
python -m api.main

# Development mode with auto-reload
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=core --cov=api --cov=math_foundation

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m "not slow"    # Exclude slow tests
pytest -m julia         # Julia integration tests
pytest -m rust          # Rust component tests
```

### Code Quality
```bash
# Format code
black .

# Sort imports
isort .

# Type checking
mypy core/ api/ math_foundation/

# Linting (if flake8 is available)
flake8 core/ api/
```

### Rust Components
```bash
# Build ethical audit system
cd ethical_audit
cargo build --release
cargo test

# Return to root
cd ..
```

### Julia Mathematical Foundation (Optional)
```bash
# Install Julia dependencies
julia -e 'using Pkg; Pkg.add(["DifferentialEquations", "LinearAlgebra", "Statistics", "SymbolicUtils"])'

# Install PyJulia for Python integration
pip install julia
python -c "import julia; julia.install()"

# Test Julia integration
julia -e 'using Pkg; Pkg.test()'
```

## Architecture Overview

Medical Research AI is a hybrid neuro-symbolic medical AI system with multiple interconnected layers:

### Core Components

1. **Hybrid Bridge** (`core/hybrid_bridge.py`): Central orchestrator that fuses symbolic and neural reasoning
   - Four reasoning modes: symbolic_first, neural_first, parallel, adaptive
   - Handles query preprocessing, strategy selection, and result fusion
   - Main entry point: `create_hybrid_reasoning_engine(config)`

2. **Symbolic Reasoning** (`core/symbolic/`): Rule-based logical reasoning
   - Integrates IBM NSTK, Nucleoid, and PEIRCE open-source components
   - Located in subdirectories with custom integration wrappers
   - Safety-first approach for medical queries

3. **Neural Reasoning** (`core/neural/`): Deep learning and quantum-inspired models
   - Integrates TorchLogic and SymbolicAI components
   - Custom quantum uncertainty models for medical domain
   - Confidence interval calculations

4. **Medical Agents** (`core/medical_agents/`): Domain-specific AI agents
   - Medical Research agent for clinical analysis assistance
   - Built on OpenSSA framework (`orchestration/openssa/`)

### Data Flow

1. **API Layer** (`api/`): FastAPI-based REST API
   - Main application: `api/main.py`
   - Route modules: medical, user, application, health
   - Middleware: rate limiting, security headers, request logging
   - Database integration with SQLAlchemy

2. **Mathematical Foundation** (`math_foundation/`): Julia + Python integration
   - QFT quantum analogs (`qft_qm.jl`)
   - Thermodynamic entropy calculations (`thermo_entropy.jl`)
   - Python wrapper with PyJulia integration

3. **Ethical Audit** (`ethical_audit/`): Rust-based safety layer
   - Consciousness detection and privacy enforcement
   - Differential privacy implementation
   - Audit trail with symbolic proofs

### Configuration

- **Ethical Constraints**: `config/ethical_constraints.yaml` defines comprehensive ethical rules
- **Database**: SQLite by default (`premedpro_ai.db`)
- **Environment Settings**: Via FastAPI settings in `api/core/config.py`

### Key Integration Patterns

1. **Open Source Components**: Integrated as subdirectories under respective layers
   - Excluded from linting/formatting (see pyproject.toml exclude patterns)
   - Custom wrapper files provide integration points

2. **Reasoning Strategy Selection**: 
   - High privacy sensitivity → symbolic_first
   - Medical diagnosis keywords → symbolic_first (safety)
   - Research queries → neural_first
   - Complex queries → parallel processing

3. **Safety Mechanisms**:
   - All neural outputs validated through symbolic reasoning
   - Ethical compliance checking on every result
   - Mandatory uncertainty disclosure for medical responses

## Important Development Notes

- **Medical Safety**: This system is for research purposes only. Never remove medical disclaimers or safety checks.
- **Privacy First**: All personal data handling must comply with differential privacy settings in ethical_constraints.yaml.
- **Hybrid Reasoning**: The system is designed to combine symbolic and neural approaches for comprehensive medical research analysis - avoid bypassing either component.
- **Open Source Integration**: Third-party components maintain their original licenses (see CREDITS.md).
- **Multi-language**: Python (main), Rust (safety), Julia (math) - ensure compatibility when making changes.

## Testing Strategy

- Unit tests for individual components
- Integration tests for cross-component interactions
- Separate test markers for Julia and Rust dependencies
- Ethics and safety validation in all test suites
- Mock medical agents available for testing without full initialization

## Production Considerations

- API documentation disabled in production (see `api/main.py`)
- Comprehensive logging with request IDs
- Rate limiting and security headers enforced
- Database connection pooling and proper cleanup
- Ethical audit system monitors all operations in real-time