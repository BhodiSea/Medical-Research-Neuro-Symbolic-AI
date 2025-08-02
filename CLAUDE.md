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

**PremedPro AI** is a hybrid neuro-symbolic medical AI system for medical research analysis and educational support, particularly focused on neurodegeneration research (Parkinson's, ALS, Alzheimer's). The system combines rule-based logical reasoning with machine learning pattern recognition for comprehensive medical research analysis.

### Current Implementation Status

**🟢 Production-Ready Components:**
- FastAPI web application with comprehensive middleware, security headers, rate limiting
- Database models with SQLAlchemy ORM and repository patterns
- Configuration management with Pydantic settings and YAML-based ethical constraints  
- Request/response handling with proper exception management and logging
- Health monitoring and system status endpoints

**🟡 Framework-Ready Components:**
- Hybrid reasoning bridge architecture with 4 reasoning modes (symbolic_first, neural_first, parallel, adaptive)
- Medical agent system with integration wrapper framework
- 31 AI systems integrated as git submodules with integration stubs
- Multi-language support framework (Python/Rust/Julia)
- Authentication and security middleware framework

**🔴 Conceptual/Mock Components:**
- Actual AI reasoning implementations (currently use mock responses)
- 10th Man deliberation system (architectural design only)
- Internal simulation engines (conceptual framework)
- Advanced multi-agent coordination (integration wrappers exist)
- Mathematical foundation integration (Julia files exist but not connected)

### Key Architecture Components

1. **API Layer** (`api/`): Production-ready FastAPI application
   - **main.py**: Complete application with lifespan management, middleware stack, exception handling
   - **routes/**: Medical, user, application, and health endpoints with proper validation
   - **core/**: Configuration, logging, middleware, authentication, and exception management
   - **database/**: SQLAlchemy models, connection management, and repository patterns
   - **Status**: ✅ Fully functional and production-ready

2. **Hybrid Bridge** (`core/hybrid_bridge.py`): Central reasoning orchestrator
   - Four reasoning modes with adaptive strategy selection based on query characteristics
   - Async processing with parallel execution capabilities
   - Performance metrics tracking and reasoning history
   - **Status**: ⚠️ Complete architecture but depends on mock AI implementations

3. **AI System Integration** (`core/`): 31 AI systems with integration framework
   - **Symbolic**: NSTK, Nucleoid, PEIRCE, Mem0, Weaviate, RDKit (6 systems)
   - **Neural**: SymbolicAI, TorchLogic, DeepChem, MONAI, BioBERT, etc. (8 systems)  
   - **Orchestration**: CrewAI, AutoGen, LangChain, OpenSSA, etc. (8 systems)
   - **Ethics**: HolisticAI, AIX360 (2 systems)
   - **Math/Clinical**: BioNeMo, OpenMM, FHIR, OMOP, etc. (7 systems)
   - **Status**: ⚠️ All submodules cloned, integration wrappers exist but need implementation

4. **Medical Agent** (`core/medical_agents/premedpro_agent.py`): Domain-specific agent
   - Medical query processing with safety validation
   - Integration with hybrid reasoning bridge
   - **Status**: ⚠️ Basic structure with mock implementations

5. **Ethical Framework** (`config/ethical_constraints.yaml`): Comprehensive safety system
   - Core principles (beneficence, non-maleficence, autonomy, justice)
   - Privacy protection with differential privacy settings
   - Research ethics with validation requirements
   - **Status**: ✅ Complete configuration framework

6. **Multi-language Components**:
   - **Rust** (`ethical_audit/`): Audit system with Cargo.toml and source structure
   - **Julia** (`math_foundation/qft_qm.jl`, `thermo_entropy.jl`): Mathematical modeling
   - **Status**: 🔴 Project structures exist but no functional integration

### Data Flow Architecture

**Current Implementation (API to Response):**
1. **HTTP Request** → FastAPI application with middleware stack ✅
2. **Request Validation** → Pydantic models with input sanitization ✅  
3. **Authentication** → JWT-based user authentication (framework ready) ⚠️
4. **Rate Limiting** → Per-IP request throttling with configurable limits ✅
5. **Medical Agent Query** → PremedPro agent with hybrid reasoning bridge ⚠️
6. **Reasoning Strategy** → Adaptive mode selection (symbolic_first/neural_first/parallel) ⚠️
7. **AI Processing** → Mock implementations for symbolic and neural reasoning 🔴
8. **Result Fusion** → Confidence scoring and uncertainty quantification ⚠️
9. **Ethical Validation** → Safety checking and compliance verification ✅
10. **Response Formatting** → Structured JSON with medical disclaimers ✅
11. **Audit Logging** → Request tracking with unique IDs and performance metrics ✅

**Current Database Flow:**
- **Models**: User, MedicalQuery, QueryResult, AuditLog with proper relationships ✅
- **Repositories**: Abstract base with concrete implementations (framework ready) ⚠️
- **Migrations**: SQLAlchemy-based schema management ✅
- **Connection Management**: Proper lifecycle and cleanup ✅

**Current AI Integration Status:**
- **Symbolic Reasoning**: Integration stubs for 6 systems (NSTK, Nucleoid, PEIRCE, etc.) 🔴
- **Neural Networks**: Integration stubs for 8 systems (SymbolicAI, TorchLogic, etc.) 🔴
- **Multi-Agent**: Integration stubs for 8 systems (CrewAI, AutoGen, etc.) 🔴
- **Ethics & Safety**: Integration stubs for 2 systems (HolisticAI, AIX360) 🔴

### Implementation Priority Analysis

**Immediate Development Needs (Weeks 1-2):**
1. **Replace Mock AI Implementations**: Core reasoning in `core/hybrid_bridge.py:141-174`
2. **Database Repository Layer**: Complete CRUD operations in `api/database/repositories.py`  
3. **Authentication System**: JWT implementation in `api/core/auth.py`
4. **Basic AI Integration**: Connect at least 2-3 submodules (SymbolicAI, CrewAI, Mem0)

**Medium-term Development (Weeks 3-8):**
1. **Neural Network Training**: Implement actual PyTorch models for medical reasoning
2. **Symbolic Logic Integration**: Connect NSTK, Nucleoid, PEIRCE for rule-based reasoning
3. **Multi-Agent Coordination**: Deploy CrewAI and AutoGen for agent collaboration
4. **Ethical Audit System**: Rust integration with Python bindings for safety monitoring

**Long-term Development (Months 3-6):**
1. **Mathematical Foundation**: Julia integration for quantum-inspired uncertainty modeling
2. **Advanced Multi-Agent**: Full 10th Man deliberation system implementation
3. **Clinical Data Integration**: FHIR and OMOP connectivity for real medical data
4. **Performance Optimization**: Caching, load balancing, and scalability improvements

### Configuration System

**Production-Ready Configuration:**
- `api/core/config.py`: Complete Pydantic settings with environment variable support ✅
- `config/ethical_constraints.yaml`: Comprehensive ethical framework with core principles ✅  
- `pyproject.toml`: Full Python packaging, dependencies, and tool configuration ✅

**Development Infrastructure:**
- **Code Quality**: Black, isort, mypy, flake8 with proper exclude patterns ✅
- **Testing**: Pytest with markers for unit/integration/julia/rust tests ✅
- **Documentation**: Project structure with comprehensive README and examples ✅

### Current System Capabilities

**What Works Now:**
1. **API Server**: Complete FastAPI application with production middleware
2. **Request Processing**: Input validation, rate limiting, error handling, logging
3. **Database Layer**: SQLAlchemy models with proper relationships and lifecycle management  
4. **Configuration Management**: Pydantic settings with environment-based configuration
5. **Code Quality**: Comprehensive tooling setup with proper formatting and type checking
6. **Architectural Framework**: Complete hybrid reasoning architecture ready for AI integration

**What Needs Implementation:**
1. **AI Reasoning**: Replace mock implementations with functional symbolic/neural processing
2. **Database Operations**: Complete repository pattern implementation for data persistence
3. **Authentication**: JWT-based user management and authorization
4. **AI System Integration**: Connect 31 submodules to functional reasoning pipeline
5. **Advanced Features**: Multi-agent coordination, mathematical modeling, ethical auditing

### Integration Strategy

**Current Reasoning Strategy (Implemented):**
- **Adaptive Mode Selection**: Query analysis determines reasoning approach `core/hybrid_bridge.py:222-245`
- **High Privacy** → `symbolic_first` (safety priority)
- **Medical Keywords** → `symbolic_first` (safety-critical) 
- **Research Queries** → `neural_first` (pattern recognition)
- **Complex Queries** → `parallel` (comprehensive analysis)

**Current Quality Assurance (Implemented):**
- Input sanitization and query preprocessing ✅
- Medical disclaimers for all responses ✅  
- Ethical compliance validation framework ✅
- Comprehensive error handling and audit logging ✅
- Request ID tracking and performance metrics ✅

## Important Development Notes

### Critical Implementation Gap Analysis

**Production-Ready Foundation vs Missing Core Functionality:**
The codebase provides an excellent architectural foundation with production-quality infrastructure, but the core AI functionality remains unimplemented. This creates a specific development scenario:

- **Excellent Infrastructure** ✅: FastAPI, database models, configuration, middleware, logging
- **Complete Architecture** ✅: Hybrid reasoning framework, agent coordination patterns, ethical constraints  
- **Missing Core Logic** 🔴: Actual AI reasoning, database operations, authentication, AI system integration

### Immediate Development Priorities (Critical Path)

1. **Replace Mock AI Implementations** (`core/hybrid_bridge.py:141-174`):
   ```python
   # Current: Mock implementations that return placeholder data
   # Needed: Functional symbolic and neural reasoning with actual AI models
   ```

2. **Complete Database Repository Layer** (`api/database/repositories.py`):
   ```python
   # Current: Abstract base classes and interface definitions
   # Needed: Concrete CRUD implementations for all data models
   ```

3. **Implement Authentication System** (`api/core/auth.py`):
   ```python
   # Current: JWT framework and middleware setup
   # Needed: User registration, login, token validation, role-based access
   ```

4. **Connect AI Submodules** (31 systems in `core/` subdirectories):
   ```python
   # Current: Integration wrapper files with import stubs
   # Needed: Functional connections to SymbolicAI, CrewAI, TorchLogic, etc.
   ```

### Development Strategy

**Focus on Infrastructure → AI → Advanced Features:**

**Week 1-2 (Foundation Completion):**
- Authentication system implementation
- Database repository completion  
- Basic AI integration (2-3 systems minimum)
- Integration testing framework

**Week 3-4 (Core AI):**
- Symbolic reasoning (NSTK, Nucleoid, PEIRCE)
- Neural network training (SymbolicAI, TorchLogic)
- Multi-agent coordination (CrewAI, AutoGen)
- Hybrid bridge functional implementation

**Week 5-8 (Advanced Integration):**
- Mathematical foundation (Julia integration)
- Ethical audit system (Rust integration)
- Clinical data systems (FHIR, OMOP)
- Performance optimization and caching

### Medical Safety Guidelines

- **Research Support Only**: System designed for medical research analysis, not clinical diagnosis or treatment decisions
- **Privacy First**: All data handling complies with differential privacy settings in `config/ethical_constraints.yaml`  
- **Safety-First Reasoning**: Medical queries always use `symbolic_first` mode for rule-based safety validation
- **Ethical Compliance**: All features must align with comprehensive ethical framework in configuration files
- **Medical Disclaimers**: All responses include appropriate medical disclaimers and safety warnings

### Development Guidelines for This Codebase

**Current Reality Assessment:**
- **Excellent Foundation**: Production-ready API infrastructure with comprehensive middleware
- **Complete Architecture**: Hybrid reasoning framework ready for AI integration  
- **Missing Core AI**: Mock implementations need replacement with functional reasoning
- **31 AI Systems**: All submodules cloned and integration wrappers exist but are non-functional

**Development Approach:**
1. **Prioritize Core Functionality**: Focus on replacing mocks with functional AI reasoning
2. **Maintain Architecture**: Preserve existing patterns and interfaces during implementation
3. **Safety First**: Ensure all medical AI implementations include proper safety validation
4. **Test-Driven**: Write comprehensive tests for all new AI integrations
5. **Gradual Integration**: Connect AI systems incrementally rather than attempting all at once

**Integration Patterns (Already Established):**
- **Submodule Structure**: 31 AI systems organized in logical subdirectories ✅
- **Integration Wrappers**: Python files for each system with defined interfaces ✅  
- **License Compliance**: Original licenses maintained in submodule directories ✅
- **Code Quality**: Proper exclusion patterns for submodules in `pyproject.toml` ✅

**Multi-Language Integration (Framework Ready):**
- **Python**: Main application logic and AI coordination ✅
- **Rust**: Ethical audit system with Cargo.toml structure ⚠️
- **Julia**: Mathematical foundation with PyJulia integration ⚠️

### Testing Strategy (Framework Complete)

**Current Testing Infrastructure:**
- **Pytest Configuration**: Complete setup with markers for different test types ✅
- **Coverage Configuration**: Proper source tracking and exclusion patterns ✅  
- **Mock Validation**: Framework for testing current mock implementations ✅
- **CI/CD Ready**: Tool configuration supports automated testing workflows ✅

**Required Test Implementation:**
- Unit tests for AI integration wrappers
- Integration tests for hybrid reasoning pipeline
- End-to-end API tests with functional AI components
- Performance benchmarking for AI processing workflows

### Production Readiness Assessment

**Currently Production-Ready:**
- FastAPI application with comprehensive middleware stack
- Database models with proper relationships and lifecycle management
- Configuration management with environment variable support
- Security headers, rate limiting, CORS, and error handling
- Health monitoring and system status endpoints
- Comprehensive logging with request ID tracking

**Requires Implementation for Production:**
- Functional AI reasoning pipeline (currently mock implementations)
- Database repository CRUD operations (abstract interfaces exist)
- User authentication and authorization (JWT framework ready)
- AI system integration (wrappers exist but non-functional)
- Performance optimization for AI processing workflows

### Development Recommendations

**For Contributors:**
1. **Start with Database Repositories**: Complete `api/database/repositories.py` implementations
2. **Implement Authentication**: Build out JWT-based user management in `api/core/auth.py`
3. **Focus on 2-3 AI Systems**: Begin with SymbolicAI, CrewAI, and Mem0 integration
4. **Write Integration Tests**: Validate AI pipeline functionality before adding more systems
5. **Maintain Medical Safety**: Ensure all AI implementations include proper ethical validation

**Architecture Strengths:**
- Excellent separation of concerns with clear module boundaries
- Production-ready infrastructure that scales well
- Comprehensive configuration management and security implementation
- Well-designed integration patterns for complex AI system coordination

The codebase provides an exceptional foundation for medical AI development, with the primary need being implementation of the core AI reasoning functionality to replace existing mock components.