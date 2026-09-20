//! AiclEndpoint — a named, addressable service entry point.
//!
//! Endpoints are registered in an [`AiclEndpointRegistry`] so that discovery
//! and routing can locate them by name.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use crate::error::{AiclError, AiclResult, ErrorCode};
use crate::transport::Endpoint;

/// An endpoint — a named, addressable service registered with the runtime.
#[derive(Debug, Clone)]
pub struct AiclEndpoint {
    /// Human-readable service name (unique within a registry).
    pub name: String,
    /// Transport address at which this endpoint can be reached.
    pub address: Endpoint,
    /// Indices of capabilities served by this endpoint.
    pub capabilities: Vec<u16>,
    /// Arbitrary key/value metadata (version, description, …).
    pub metadata: BTreeMap<String, String>,
}

impl AiclEndpoint {
    /// Construct an endpoint with a name and address.
    pub fn new(name: impl Into<String>, address: Endpoint) -> Self {
        Self {
            name: name.into(),
            address,
            capabilities: Vec::new(),
            metadata: BTreeMap::new(),
        }
    }

    /// Add a capability index to this endpoint (builder pattern).
    pub fn with_capability(mut self, cap: u16) -> Self {
        self.capabilities.push(cap);
        self
    }

    /// Add a metadata key/value pair (builder pattern).
    pub fn with_metadata(mut self, key: impl Into<String>, val: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), val.into());
        self
    }

    /// Check whether this endpoint matches the given name (case-sensitive).
    #[inline]
    pub fn matches_name(&self, name: &str) -> bool {
        self.name == name
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Endpoint registry
// ─────────────────────────────────────────────────────────────────────────────

/// A registry of named endpoints.
#[derive(Debug, Default)]
pub struct AiclEndpointRegistry {
    endpoints: BTreeMap<String, AiclEndpoint>,
}

impl AiclEndpointRegistry {
    /// Register an endpoint. Returns an error if an endpoint with the same
    /// name is already registered.
    pub fn register(&mut self, ep: AiclEndpoint) -> AiclResult<()> {
        if self.endpoints.contains_key(&ep.name) {
            return Err(AiclError::new(ErrorCode::CapabilityExists,
                alloc::format!("endpoint '{}' already registered", ep.name)));
        }
        self.endpoints.insert(ep.name.clone(), ep);
        Ok(())
    }

    /// Unregister and return the endpoint with the given name.
    pub fn unregister(&mut self, name: &str) -> AiclResult<AiclEndpoint> {
        self.endpoints.remove(name)
            .ok_or_else(|| AiclError::new(ErrorCode::CapabilityNotFound,
                alloc::format!("endpoint '{}' not found", name)))
    }

    /// Look up an endpoint by exact name.
    pub fn find(&self, name: &str) -> Option<&AiclEndpoint> {
        self.endpoints.get(name)
    }

    /// List all registered endpoints in sorted order.
    pub fn list(&self) -> Vec<&AiclEndpoint> {
        self.endpoints.values().collect()
    }
}
