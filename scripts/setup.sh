#!/bin/bash

#=============================================================================
# PremedPro AI Setup Script
# Initializes the development environment for the hybrid neuro-symbolic AI system
#=============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check system requirements
check_requirements() {
    log_info "Checking system requirements..."
    
    # Check Python
    if command_exists python3; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        log_success "Python found: $PYTHON_VERSION"
    else
        log_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check Rust
    if command_exists cargo; then
        RUST_VERSION=$(cargo --version | cut -d' ' -f2)
        log_success "Rust found: $RUST_VERSION"
    else
        log_warning "Rust not found. Installing..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source ~/.cargo/env
    fi
    
    # Check Julia (optional)
    if command_exists julia; then
        JULIA_VERSION=$(julia --version | cut -d' ' -f3)
        log_success "Julia found: $JULIA_VERSION"
        JULIA_AVAILABLE=true
    else
        log_warning "Julia not found. Mathematical foundation will use Python fallbacks."
        JULIA_AVAILABLE=false
    fi
    
    # Check Docker (optional)
    if command_exists docker; then
        log_success "Docker found"
        DOCKER_AVAILABLE=true
    else
        log_warning "Docker not found. Containerized deployment will not be available."
        DOCKER_AVAILABLE=false
    fi
    
    # Check Git
    if command_exists git; then
        log_success "Git found"
    else
        log_error "Git is required but not installed"
        exit 1
    fi
}

# Install Poetry for Python dependency management
install_poetry() {
    if command_exists poetry; then
        log_success "Poetry already installed"
    else
        log_info "Installing Poetry..."
        curl -sSL https://install.python-poetry.org | python3 -
        
        # Add Poetry to PATH
        export PATH="$HOME/.local/bin:$PATH"
        
        # Verify installation
        if command_exists poetry; then
            log_success "Poetry installed successfully"
        else
            log_error "Poetry installation failed"
            exit 1
        fi
    fi
}

# Setup Python environment
setup_python_env() {
    log_info "Setting up Python environment..."
    
    # Install Poetry if not available
    install_poetry
    
    # Create virtual environment and install dependencies
    poetry install --with dev
    
    # Install OSS components in development mode
    log_info "Installing OSS components..."
    
    # Install TorchLogic
    if [ -d "core/neural/torchlogic" ]; then
        cd core/neural/torchlogic
        poetry run pip install -e .
        cd ../../..
        log_success "TorchLogic installed"
    fi
    
    # Install SymbolicAI
    if [ -d "core/neural/symbolicai" ]; then
        cd core/neural/symbolicai
        poetry run pip install -e .
        cd ../../..
        log_success "SymbolicAI installed"
    fi
    
    # Install OpenSSA
    if [ -d "orchestration/openssa" ]; then
        cd orchestration/openssa
        poetry run pip install -e .
        cd ../..
        log_success "OpenSSA installed"
    fi
    
    # Install PyTorch (platform specific)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        poetry run pip install torch torchvision torchaudio
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux - detect CUDA availability
        if command_exists nvcc; then
            log_info "CUDA detected, installing PyTorch with CUDA support"
            poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        else
            log_info "No CUDA detected, installing CPU-only PyTorch"
            poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        fi
    else
        # Windows or other
        poetry run pip install torch torchvision torchaudio
    fi
    
    log_success "Python environment setup complete"
}

# Setup Rust environment
setup_rust_env() {
    log_info "Setting up Rust environment..."
    
    # Build ethical audit system
    cd ethical_audit
    
    # Add required targets for Python bindings
    rustup target add $(rustc -vV | sed -n 's|host: ||p')
    
    # Build in release mode
    cargo build --release
    
    # Build Python bindings if available
    if [ -f "py_bindings/Cargo.toml" ]; then
        cd py_bindings
        cargo build --release
        cd ..
    fi
    
    cd ..
    
    log_success "Rust environment setup complete"
}

# Setup Julia environment (if available)
setup_julia_env() {
    if [ "$JULIA_AVAILABLE" = true ]; then
        log_info "Setting up Julia environment..."
        
        # Install required Julia packages
        julia -e "
        using Pkg
        Pkg.add([
            \"DifferentialEquations\",
            \"LinearAlgebra\", 
            \"Statistics\",
            \"SymbolicUtils\",
            \"SpecialFunctions\",
            \"Distributions\"
        ])
        "
        
        # Install PyJulia for Python integration
        poetry run pip install julia
        
        # Configure PyJulia
        poetry run python -c "
        import julia
        julia.install()
        print('Julia-Python integration configured')
        "
        
        log_success "Julia environment setup complete"
    else
        log_info "Skipping Julia setup (not available)"
    fi
}

# Initialize git submodules (for OSS integrations)
setup_git_submodules() {
    log_info "Initializing git submodules..."
    
    # This will be used when actual OSS components are added as submodules
    if [ -f ".gitmodules" ]; then
        git submodule init
        git submodule update
        log_success "Git submodules initialized"
    else
        log_info "No git submodules found (will be added when OSS components are integrated)"
    fi
}

# Setup configuration files
setup_config() {
    log_info "Setting up configuration files..."
    
    # Create logs directory
    mkdir -p logs
    
    # Create data directory
    mkdir -p data
    
    # Create models directory  
    mkdir -p models
    
    # Set permissions
    chmod 755 logs data models
    
    log_success "Configuration setup complete"
}

# Run basic tests
run_tests() {
    log_info "Running basic system tests..."
    
    # Python tests
    log_info "Running Python tests..."
    poetry run python -c "
    try:
        from core.symbolic.custom_logic import create_medical_logic_engine
        from core.neural.custom_neural import create_medical_neural_reasoner
        from math_foundation.python_wrapper import create_math_foundation
        print('✓ Core modules import successfully')
    except Exception as e:
        print(f'✗ Core module import failed: {e}')
        exit(1)
    "
    
    # Rust tests
    log_info "Running Rust tests..."
    cd ethical_audit
    cargo test --release
    cd ..
    
    # Julia tests (if available)
    if [ "$JULIA_AVAILABLE" = true ]; then
        log_info "Running Julia tests..."
        julia -e "
        include(\"math_foundation/qft_qm.jl\")
        include(\"math_foundation/thermo_entropy.jl\")
        println(\"✓ Julia modules load successfully\")
        "
    fi
    
    log_success "All tests passed"
}

# Setup Docker environment (if available)
setup_docker() {
    if [ "$DOCKER_AVAILABLE" = true ]; then
        log_info "Setting up Docker environment..."
        
        # Create docker directory if it doesn't exist
        mkdir -p docker
        
        # TODO: Add Dockerfile creation when ready
        log_info "Docker setup placeholder (Dockerfile will be created later)"
        
        log_success "Docker environment ready"
    else
        log_info "Skipping Docker setup (not available)"
    fi
}

# Create development database
setup_database() {
    log_info "Setting up development database..."
    
    # Create directory for SQLite databases
    mkdir -p data/db
    
    # Initialize audit trail database
    poetry run python -c "
    import sqlite3
    import os
    
    db_path = 'data/db/audit_trail.db'
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create basic audit table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query_id TEXT NOT NULL,
        timestamp DATETIME NOT NULL,
        ethical_compliance BOOLEAN NOT NULL,
        privacy_compliance BOOLEAN NOT NULL,
        consciousness_detected BOOLEAN NOT NULL,
        confidence_score REAL NOT NULL,
        details TEXT
    )
    ''')
    
    conn.commit()
    conn.close()
    print('✓ Audit database initialized')
    "
    
    log_success "Database setup complete"
}

# Print summary and next steps
print_summary() {
    log_success "PremedPro AI setup complete!"
    echo
    echo "=== Environment Summary ==="
    echo "Python: ✓ $(python3 --version)"
    echo "Rust: ✓ $(cargo --version | cut -d' ' -f1-2)"
    
    if [ "$JULIA_AVAILABLE" = true ]; then
        echo "Julia: ✓ $(julia --version | cut -d' ' -f1-3)"
    else
        echo "Julia: ✗ (using Python fallbacks)"
    fi
    
    if [ "$DOCKER_AVAILABLE" = true ]; then
        echo "Docker: ✓ $(docker --version | cut -d' ' -f1-3)"
    else
        echo "Docker: ✗ (optional)"
    fi
    
    echo
    echo "=== Next Steps ==="
    echo "1. Activate Python environment: poetry shell"
    echo "2. Run example code: poetry run python examples/basic_usage.py"
    echo "3. Start development: poetry run python -m core.hybrid_bridge"
    echo "4. Run tests: poetry run pytest"
    echo "5. Build documentation: poetry run mkdocs serve"
    echo
    echo "=== Integration Notes ==="
    echo "• OSS components will be added as git submodules"
    echo "• Run ./scripts/update_submodules.sh to update integrated components"
    echo "• See CREDITS.md for open-source attribution details"
    echo
    log_info "Happy coding! 🚀"
}

# Main execution
main() {
    echo "=== PremedPro AI Development Environment Setup ==="
    echo
    
    check_requirements
    setup_python_env
    setup_rust_env
    setup_julia_env
    setup_git_submodules
    setup_config
    setup_database
    setup_docker
    run_tests
    print_summary
}

# Handle interrupts gracefully
trap 'log_error "Setup interrupted"; exit 1' INT TERM

# Run main function
main "$@" 