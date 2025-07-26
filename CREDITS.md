# Credits and Attributions

This project integrates several open-source repositories and libraries. We acknowledge and thank the following projects and their contributors:

## Core Open-Source Integrations

### IBM Neuro-Symbolic AI Toolkit (NSTK)
- **Repository**: https://github.com/IBM/neuro-symbolic-ai
- **License**: Apache 2.0
- **Usage**: Symbolic reasoning layer with Logical Neural Networks (LNNs)
- **Modifications**: Extended with premed-specific rules and medical logic

### Nucleoid (NucleoidAI)
- **Repository**: https://github.com/NucleoidAI/Nucleoid
- **License**: MIT
- **Usage**: Knowledge graph construction and management
- **Modifications**: Added medical graph builders and domain-specific ontologies

### TorchLogic (IBM)
- **Repository**: https://github.com/IBM/torchlogic
- **License**: Apache 2.0
- **Usage**: Weighted logic operations in neural networks
- **Modifications**: Fine-tuned for medical data streams and uncertainty quantification

### SymbolicAI (ExtensityAI)
- **Repository**: https://github.com/ExtensityAI/symbolicai
- **License**: MIT
- **Usage**: LLM integration with symbolic reasoning
- **Modifications**: Added local model hooks and privacy-preserving inference

### PEIRCE (neuro-symbolic-ai)
- **Repository**: https://github.com/neuro-symbolic-ai/peirce
- **License**: MIT
- **Usage**: Inference loops and reasoning chains
- **Modifications**: Customized for thermodynamic entropy checks and ethical constraints

### OpenSSA (aitomatic)
- **Repository**: https://github.com/aitomatic/openssa
- **License**: Apache 2.0
- **Usage**: Agentic systems and orchestration
- **Modifications**: Adapted for premed-specific tasks and ethical oversight

## Additional Dependencies

### Python Libraries
- PyTorch: Deep learning framework
- SymPy: Symbolic mathematics
- PyO3: Rust-Python bindings
- PyJulia: Python-Julia integration
- FastAPI: Web framework for API endpoints

### Rust Crates
- serde: Serialization framework
- tokio: Async runtime
- differential-privacy: Privacy-preserving algorithms

### Julia Packages
- DifferentialEquations.jl: Mathematical modeling
- SymbolicUtils.jl: Symbolic computation
- Plots.jl: Visualization

## License Compliance

This project is licensed under MIT License with proper attribution to all integrated open-source components. All modifications and extensions are clearly documented and maintain compatibility with original licenses.

## Contributing

When adding new open-source dependencies, please:
1. Add attribution to this file
2. Verify license compatibility
3. Document any modifications made
4. Maintain clear separation between original and custom code 