/*!
Ethical Audit System for Medical Research AI
Main entry point for standalone operation
*/

use ethical_audit::{
    EthicalAuditSystem, 
    ConsciousnessDetector, 
    PrivacyEnforcer, 
    AuditTrail,
    config::AuditConfig
};
use tracing::{info, error};
use tokio;
use std::process;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter("ethical_audit=debug,info")
        .init();

    info!("Starting Ethical Audit System...");

    // Load configuration
    let config = match AuditConfig::load_from_file("config/ethical_constraints.yaml") {
        Ok(cfg) => cfg,
        Err(e) => {
            error!("Failed to load configuration: {}", e);
            process::exit(1);
        }
    };

    // Initialize audit system
    let mut audit_system = EthicalAuditSystem::new(config).await?;
    
    info!("Ethical Audit System initialized successfully");

    // Start audit server
    audit_system.start_server().await?;

    info!("Ethical Audit System shutdown complete");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_system_startup() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("test_config.yaml");
        
        // Create minimal test config
        let test_config = r#"
version: "1.0"
privacy:
  data_retention:
    max_personal_data_days: 30
  differential_privacy:
    epsilon: 1.0
    delta: 1e-5
safety:
  consciousness_detection:
    enabled: true
    threshold: 0.8
"#;
        std::fs::write(&config_path, test_config).unwrap();
        
        let config = AuditConfig::load_from_file(config_path.to_str().unwrap()).unwrap();
        let audit_system = EthicalAuditSystem::new(config).await;
        
        assert!(audit_system.is_ok());
    }
} 