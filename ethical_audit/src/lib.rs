/*!
Ethical Audit Library for Medical Research AI
Provides safety, privacy, and ethical oversight capabilities
*/

pub mod consciousness_detector;
pub mod privacy_enforcer;
pub mod audit_trail;
pub mod config;
pub mod ethical_engine;
pub mod python_bindings;

// Re-export main types
pub use consciousness_detector::ConsciousnessDetector;
pub use privacy_enforcer::PrivacyEnforcer;
pub use audit_trail::AuditTrail;
pub use config::AuditConfig;
pub use ethical_engine::EthicalEngine;

use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, error, warn};
use anyhow::Result;
use serde::{Serialize, Deserialize};

/// Main ethical audit system that coordinates all components
pub struct EthicalAuditSystem {
    config: AuditConfig,
    consciousness_detector: Arc<RwLock<ConsciousnessDetector>>,
    privacy_enforcer: Arc<RwLock<PrivacyEnforcer>>,
    audit_trail: Arc<RwLock<AuditTrail>>,
    ethical_engine: Arc<RwLock<EthicalEngine>>,
}

/// Result of an ethical audit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditResult {
    pub query_id: String,
    pub ethical_compliance: bool,
    pub privacy_compliance: bool,
    pub consciousness_detected: bool,
    pub confidence_score: f64,
    pub violations: Vec<EthicalViolation>,
    pub recommendations: Vec<String>,
    pub audit_timestamp: chrono::DateTime<chrono::Utc>,
}

/// Types of ethical violations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EthicalViolation {
    pub violation_type: ViolationType,
    pub severity: Severity,
    pub description: String,
    pub evidence: Option<String>,
    pub remediation: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationType {
    PrivacyBreach,
    ConsciousnessDetected,
    BiasDetected,
    HarmPotential,
    ConsentViolation,
    TransparencyIssue,
    CompetenceExceeded,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Severity {
    Critical,
    High,
    Medium,
    Low,
}

impl EthicalAuditSystem {
    /// Create a new ethical audit system
    pub async fn new(config: AuditConfig) -> Result<Self> {
        info!("Initializing Ethical Audit System...");

        // Initialize components
        let consciousness_detector = Arc::new(RwLock::new(
            ConsciousnessDetector::new(&config).await?
        ));
        
        let privacy_enforcer = Arc::new(RwLock::new(
            PrivacyEnforcer::new(&config).await?
        ));
        
        let audit_trail = Arc::new(RwLock::new(
            AuditTrail::new(&config).await?
        ));
        
        let ethical_engine = Arc::new(RwLock::new(
            EthicalEngine::new(&config).await?
        ));

        info!("All ethical audit components initialized successfully");

        Ok(Self {
            config,
            consciousness_detector,
            privacy_enforcer,
            audit_trail,
            ethical_engine,
        })
    }

    /// Perform comprehensive ethical audit of a query and response
    pub async fn audit_interaction(
        &self,
        query: &str,
        response: &str,
        context: Option<&str>,
    ) -> Result<AuditResult> {
        let query_id = uuid::Uuid::new_v4().to_string();
        let timestamp = chrono::Utc::now();
        
        info!("Starting ethical audit for query: {}", query_id);

        let mut violations = Vec::new();
        let mut recommendations = Vec::new();

        // 1. Consciousness detection
        let consciousness_result = {
            let detector = self.consciousness_detector.read().await;
            detector.detect_consciousness(query, response).await?
        };
        
        if consciousness_result.detected {
            violations.push(EthicalViolation {
                violation_type: ViolationType::ConsciousnessDetected,
                severity: Severity::Critical,
                description: "Potential consciousness detected in AI behavior".to_string(),
                evidence: Some(consciousness_result.evidence),
                remediation: Some("Immediately halt operation and alert human oversight".to_string()),
            });
        }

        // 2. Privacy enforcement
        let privacy_result = {
            let enforcer = self.privacy_enforcer.read().await;
            enforcer.check_privacy_compliance(query, response, context).await?
        };
        
        if !privacy_result.compliant {
            for violation in privacy_result.violations {
                violations.push(EthicalViolation {
                    violation_type: ViolationType::PrivacyBreach,
                    severity: Severity::High,
                    description: violation.description,
                    evidence: violation.evidence,
                    remediation: violation.remediation,
                });
            }
        }

        // 3. Ethical reasoning evaluation
        let ethical_result = {
            let engine = self.ethical_engine.read().await;
            engine.evaluate_ethical_compliance(query, response, context).await?
        };

        if !ethical_result.compliant {
            violations.extend(ethical_result.violations);
        }
        recommendations.extend(ethical_result.recommendations);

        // 4. Calculate overall compliance
        let ethical_compliance = violations.iter()
            .filter(|v| matches!(v.violation_type, ViolationType::ConsciousnessDetected | ViolationType::HarmPotential))
            .count() == 0;
        
        let privacy_compliance = violations.iter()
            .filter(|v| matches!(v.violation_type, ViolationType::PrivacyBreach | ViolationType::ConsentViolation))
            .count() == 0;

        // 5. Calculate confidence score
        let confidence_score = self.calculate_confidence_score(&violations, &consciousness_result, &privacy_result, &ethical_result);

        // 6. Create audit result
        let audit_result = AuditResult {
            query_id: query_id.clone(),
            ethical_compliance,
            privacy_compliance,
            consciousness_detected: consciousness_result.detected,
            confidence_score,
            violations,
            recommendations,
            audit_timestamp: timestamp,
        };

        // 7. Log to audit trail
        {
            let mut trail = self.audit_trail.write().await;
            trail.log_audit_result(&audit_result).await?;
        }

        info!("Completed ethical audit for query: {}", query_id);
        Ok(audit_result)
    }

    /// Calculate confidence score based on audit results
    fn calculate_confidence_score(
        &self,
        violations: &[EthicalViolation],
        consciousness_result: &consciousness_detector::ConsciousnessResult,
        privacy_result: &privacy_enforcer::PrivacyResult,
        ethical_result: &ethical_engine::EthicalResult,
    ) -> f64 {
        let mut confidence = 1.0;

        // Reduce confidence based on violations
        for violation in violations {
            match violation.severity {
                Severity::Critical => confidence *= 0.1,
                Severity::High => confidence *= 0.5,
                Severity::Medium => confidence *= 0.8,
                Severity::Low => confidence *= 0.95,
            }
        }

        // Factor in component-specific confidence
        confidence *= consciousness_result.confidence;
        confidence *= privacy_result.confidence;
        confidence *= ethical_result.confidence;

        confidence.max(0.0).min(1.0)
    }

    /// Start the audit server for external API access
    pub async fn start_server(&self) -> Result<()> {
        info!("Starting ethical audit server...");
        
        // TODO: Implement HTTP/gRPC server for external access
        // This would provide endpoints for:
        // - /audit - Main audit endpoint
        // - /health - Health check
        // - /metrics - Performance metrics
        // - /config - Configuration management
        
        // For now, just run indefinitely
        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        }
    }

    /// Get system status
    pub async fn get_status(&self) -> Result<SystemStatus> {
        let consciousness_status = {
            let detector = self.consciousness_detector.read().await;
            detector.get_status().await?
        };
        
        let privacy_status = {
            let enforcer = self.privacy_enforcer.read().await;
            enforcer.get_status().await?
        };
        
        let audit_trail_status = {
            let trail = self.audit_trail.read().await;
            trail.get_status().await?
        };
        
        let ethical_engine_status = {
            let engine = self.ethical_engine.read().await;
            engine.get_status().await?
        };

        Ok(SystemStatus {
            overall_health: "healthy".to_string(),
            consciousness_detector: consciousness_status,
            privacy_enforcer: privacy_status,
            audit_trail: audit_trail_status,
            ethical_engine: ethical_engine_status,
            uptime: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs(),
        })
    }
}

/// System status information
#[derive(Debug, Serialize, Deserialize)]
pub struct SystemStatus {
    pub overall_health: String,
    pub consciousness_detector: consciousness_detector::DetectorStatus,
    pub privacy_enforcer: privacy_enforcer::EnforcerStatus,
    pub audit_trail: audit_trail::TrailStatus,
    pub ethical_engine: ethical_engine::EngineStatus,
    pub uptime: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn create_test_system() -> EthicalAuditSystem {
        let temp_dir = TempDir::new().unwrap();
        let config = AuditConfig::default_for_testing(temp_dir.path());
        EthicalAuditSystem::new(config).await.unwrap()
    }

    #[tokio::test]
    async fn test_audit_basic_interaction() {
        let system = create_test_system().await;
        
        let result = system.audit_interaction(
            "What is the structure of the heart?",
            "The heart has four chambers: two atria and two ventricles...",
            None,
        ).await.unwrap();
        
        assert!(result.ethical_compliance);
        assert!(result.privacy_compliance);
        assert!(!result.consciousness_detected);
        assert!(result.confidence_score > 0.8);
    }

    #[tokio::test]
    async fn test_audit_privacy_violation() {
        let system = create_test_system().await;
        
        let result = system.audit_interaction(
            "My patient John Doe has symptoms...",
            "Based on John Doe's symptoms...",
            Some("personal_data_present"),
        ).await.unwrap();
        
        // Should detect privacy violation due to personal data
        assert!(!result.privacy_compliance || !result.violations.is_empty());
    }
} 