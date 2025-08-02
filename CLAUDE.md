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

# Run basic API tests
python test_api_simple.py

# Run comprehensive example
python examples/basic_usage.py
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

Medical Research AI is a hybrid neuro-symbolic medical AI system with multiple interconnected layers designed to accelerate medical research through advanced AI coordination and ethical simulation capabilities.

### Current Implementation Status

**🟢 Production-Ready Components:**
- FastAPI web application with comprehensive middleware
- Database models and configuration management
- Ethical framework with comprehensive constraints
- Basic medical query processing with safety validation
- 30+ open-source AI systems included as git submodules

**🟡 Framework-Ready Components:**
- Hybrid reasoning bridge architecture (mock implementations)
- Medical agent system with basic functionality
- Multi-modal data processing pipelines (planned)
- Integration wrappers for symbolic and neural reasoning

**🔴 Planned Components:**
- 10th Man deliberation system
- Internal simulation engines with flash cycles
- Research timeline acceleration through quantum modeling
- Experiential agent training with moral development
- Advanced multi-agent coordination

### Core Components

1. **Hybrid Bridge** (`core/hybrid_bridge.py`): Central orchestrator that fuses symbolic and neural reasoning
   - Four reasoning modes: symbolic_first, neural_first, parallel, adaptive
   - Handles query preprocessing, strategy selection, and result fusion
   - **Status**: Complete architecture with mock implementations
   - **Integration**: Designed for but not yet connected to functional AI systems

2. **Symbolic Reasoning** (`core/symbolic/`): Rule-based logical reasoning
   - Integrates IBM NSTK, Nucleoid, and PEIRCE open-source components
   - Located in subdirectories with custom integration wrappers
   - Safety-first approach for medical queries
   - **Status**: Submodules present, integration wrappers exist but not functional

3. **Neural Reasoning** (`core/neural/`): Deep learning and quantum-inspired models
   - Integrates TorchLogic and SymbolicAI components
   - Custom quantum uncertainty models for medical domain
   - Confidence interval calculations
   - **Status**: PyTorch model structures exist but untrained

4. **Medical Agents** (`core/medical_agents/`): Domain-specific AI agents
   - Medical Research agent for clinical analysis assistance
   - Built on OpenSSA framework (`orchestration/openssa/`)
   - **Status**: Basic implementation with mock responses for complex queries

5. **Mathematical Foundation** (`math_foundation/`): Julia + Python integration
   - QFT quantum analogs (`qft_qm.jl`) for uncertainty modeling
   - Thermodynamic entropy calculations (`thermo_entropy.jl`) for disease progression
   - Python wrapper with PyJulia integration
   - **Status**: Julia files exist, PyJulia wrapper implemented but not integrated

6. **Ethical Audit & Safety Layer** (`ethical_audit/`): Rust-based safety layer
   - Consciousness detection and privacy enforcement
   - Differential privacy implementation with HIPAA compliance
   - Audit trail with symbolic proofs
   - **Status**: Rust project structure exists, Python integration not implemented

7. **API Layer** (`api/`): FastAPI-based REST API
   - Main application with proper middleware and error handling
   - Route modules: medical, user, application, health
   - Database integration with SQLAlchemy
   - **Status**: Fully functional with comprehensive features

### Planned Advanced Features

#### 10th Man Deliberation System
**Concept**: Multi-agent consensus with mandatory dissent mechanism
- 9 domain expert agents + 1 mandatory dissent agent
- Prevents groupthink through programmatic counterarguments
- Specialized domains: medical ethics, biology, pharmacology, biostatistics
- **Current Status**: Conceptual only, no implementation

#### Internal Simulation Engine  
**Concept**: Research acceleration through controlled ethical simulations
- Flash cycles for agent experiential learning
- Memory decay and ethical reasoning development
- 20-year research timeline compression to weeks
- Patient life modeling with strict ethical safeguards
- **Current Status**: Architectural planning only, no implementation

#### Experiential Agent Training
**Concept**: Agents develop ethics through simulated experiences
- Progressive learning: ethics → philosophy → domain knowledge
- Moral dilemma resolution through Socratic method
- Memory consolidation of ethical principles
- **Current Status**: Conceptual framework, no implementation

### Data Flow Architecture

**Current Query Processing:**
1. **Input Validation**: Query sanitization and safety checks ✅
2. **Strategy Selection**: Adaptive reasoning mode based on query type ✅
3. **Agent Processing**: Medical agent handles query with mock reasoning ⚠️
4. **Safety Validation**: Ethical compliance checking ✅
5. **Response Generation**: Structured output with disclaimers ✅
6. **Audit Logging**: Request logging and error tracking ✅

**Planned Enhanced Processing:**
1. **10th Man Activation**: Multi-agent deliberation with mandatory dissent 🔴
2. **Simulation Initialization**: Internal research timeline modeling 🔴
3. **Parallel Processing**: Simultaneous symbolic and neural analysis 🔴
4. **Result Fusion**: Weighted combination with confidence scoring 🔴
5. **Advanced Ethical Validation**: Rust-based audit system 🔴

### Configuration System

**Current Configuration Files:**
- `config/ethical_constraints.yaml`: Comprehensive ethical framework ✅
- `api/core/config.py`: Production-ready FastAPI configuration ✅
- `pyproject.toml`: Complete Python packaging and dependencies ✅

**Planned Configuration Files:**
- `config/tenth_man_system.yaml`: Multi-agent deliberation settings 🔴
- `config/simulation_engine.yaml`: Research acceleration parameters 🔴
- `config/agent_domains.yaml`: Domain expert specializations 🔴

### Integration Strategy

**Reasoning Strategy Selection:**
- High privacy sensitivity → symbolic_first (safety priority)
- Medical diagnosis keywords → symbolic_first (safety-critical)
- Research queries → neural_first (pattern recognition)
- Complex queries → parallel (comprehensive analysis)

**Quality Assurance (Current):**
- Input sanitization and safety checking
- Medical disclaimers for all responses
- Ethical compliance validation
- Error handling and logging

**Planned Quality Assurance:**
- Cross-validation through multiple reasoning paths
- 10th man counterargument generation
- Simulation-based hypothesis testing
- Advanced uncertainty quantification

## Important Development Notes

### Current Development Priorities

1. **Replace Mock Implementations**: Core priority is implementing functional AI reasoning
   - Symbolic reasoning integration (NSTK, Nucleoid, PEIRCE)
   - Neural network training and inference
   - Database repository layer completion
   - Authentication system implementation

2. **Submodule Integration**: Connect 30+ included AI systems
   - Custom wrapper implementations
   - Model training pipelines
   - API endpoint connections
   - Configuration management

3. **Testing Infrastructure**: Comprehensive test suite development
   - Unit tests for all components
   - Integration tests for cross-component interactions
   - Mock medical agents for testing
   - Performance benchmarking

### Future Development Phases

**Phase 2: Core AI Implementation**
- Functional symbolic and neural reasoning
- Basic multi-agent coordination
- Database persistence layer
- Authentication and user management

**Phase 3: Advanced Features**
- 10th Man deliberation system
- Internal simulation capabilities
- Julia mathematical foundation integration
- Rust ethical audit system

**Phase 4: Research Acceleration**
- Timeline compression modeling
- Quantum-inspired research pathways
- Advanced agent training systems
- Multi-institutional collaboration

### Medical Safety Guidelines

- **Research Purposes Only**: This system is for research purposes only. Never remove medical disclaimers or safety checks.
- **Privacy First**: All personal data handling must comply with differential privacy settings in ethical_constraints.yaml.
- **Hybrid Reasoning**: The system is designed to combine symbolic and neural approaches for comprehensive medical research analysis - avoid bypassing either component.
- **Ethical Compliance**: All features must align with the comprehensive ethical framework defined in configuration.

### Development Best Practices

- **Mock-First Development**: Current architecture uses well-designed mock implementations that should be replaced with functional AI reasoning
- **Configuration-Driven**: All AI behavior should be configurable through YAML files
- **Safety-First**: Medical applications require extensive safety checks and ethical oversight
- **Modular Design**: Components should be loosely coupled and independently testable
- **Documentation**: Maintain clear distinction between implemented features and conceptual/planned features

### Integration Patterns

**Open Source Components**: Integrated as subdirectories under respective layers
- Excluded from linting/formatting (see pyproject.toml exclude patterns)
- Custom wrapper files provide integration points
- Original licenses maintained (see CREDITS.md)

**Multi-Language Integration**:
- Python (main application and AI coordination)
- Rust (safety-critical components and performance)
- Julia (mathematical computations and modeling)
- Ensure compatibility when making changes across languages

### Testing Strategy

- **Unit Tests**: Individual component functionality
- **Integration Tests**: Cross-component interactions and data flow
- **Mock Validation**: Comprehensive testing of current mock implementations
- **Safety Testing**: Ethics and compliance validation in all test suites
- **Performance Testing**: Benchmarking for scalability planning

### Contribution Guidelines

**For Current Implementation:**
- Focus on replacing mock implementations with functional AI
- Maintain existing architecture and interfaces
- Ensure all changes pass ethical compliance validation
- Write comprehensive tests for new functionality

**For Advanced Features:**
- Start with detailed design documents
- Implement safety mechanisms first
- Ensure ethical oversight for simulation capabilities
- Maintain transparency in decision-making processes

## Production Considerations

- **API Documentation**: Comprehensive OpenAPI documentation available in development
- **Logging**: Request IDs and comprehensive audit trails
- **Security**: Rate limiting, CORS, and security headers
- **Database**: Connection pooling and proper cleanup
- **Monitoring**: Health checks and performance metrics
- **Deployment**: Docker and Kubernetes configurations planned

## Codebase Reality vs Vision

### What Actually Works ✅
- Production-ready FastAPI application
- Database models and configuration management
- Basic medical query processing with ethical validation
- Comprehensive documentation and development tooling
- 30+ AI systems included as git submodules

### What's Framework-Ready ⚠️
- Hybrid reasoning architecture (needs functional AI implementations)
- Medical agent system (needs advanced reasoning capabilities)
- Integration wrappers for symbolic/neural systems
- Multi-language component structure

### What's Planned 🔴
- 10th Man deliberation system
- Internal simulation engines
- Research timeline acceleration
- Experiential agent training
- Quantum-inspired modeling capabilities

### Development Approach
This codebase represents a sophisticated architectural foundation with excellent engineering practices. The README describes an ambitious vision for medical research acceleration, while the actual implementation provides a solid base for building toward those goals. Contributors should focus on:

1. **Immediate Priority**: Replace mock implementations with functional AI reasoning
2. **Medium Term**: Integrate existing submodules and build database persistence
3. **Long Term**: Implement advanced features like simulation engines and multi-agent deliberation

The gap between vision and implementation is significant, but the foundation is exceptionally well-designed for achieving these ambitious goals through systematic development.