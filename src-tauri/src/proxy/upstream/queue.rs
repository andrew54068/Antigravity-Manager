// Model-Specific Request Queue
// Prevents thundering herd on capacity-limited models like claude-opus-4-5-thinking
//
// Problem: Multiple concurrent requests to thinking models race for limited capacity slots,
// causing 429s even with retry logic (all retries fire simultaneously).
//
// Solution: Per-model semaphores that serialize requests to capacity-constrained models.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// Model queue configuration
#[derive(Debug, Clone)]
pub struct ModelQueueConfig {
    /// Default concurrency for unknown models
    pub default_concurrency: usize,
    /// Specific concurrency limits per model pattern
    pub model_limits: HashMap<String, usize>,
}

impl Default for ModelQueueConfig {
    fn default() -> Self {
        let mut model_limits = HashMap::new();
        // Thinking models have very limited capacity
        model_limits.insert("opus-thinking".to_string(), 1);
        model_limits.insert("sonnet-thinking".to_string(), 2);
        // Non-thinking Claude models
        model_limits.insert("opus".to_string(), 2);
        model_limits.insert("sonnet".to_string(), 4);
        model_limits.insert("haiku".to_string(), 8);
        // Gemini models (generally more capacity)
        model_limits.insert("gemini-pro".to_string(), 4);
        model_limits.insert("gemini-flash".to_string(), 8);

        Self {
            default_concurrency: 4,
            model_limits,
        }
    }
}

/// Per-model request queue using semaphores
pub struct ModelQueue {
    /// Semaphores keyed by normalized model name
    semaphores: RwLock<HashMap<String, Arc<Semaphore>>>,
    /// Configuration
    config: ModelQueueConfig,
}

impl ModelQueue {
    pub fn new() -> Self {
        Self::with_config(ModelQueueConfig::default())
    }

    pub fn with_config(config: ModelQueueConfig) -> Self {
        Self {
            semaphores: RwLock::new(HashMap::new()),
            config,
        }
    }

    /// Normalize model name to a queue key
    /// Groups similar models together (e.g., claude-opus-4-5-20251101 -> opus)
    fn normalize_model_key(&self, model: &str) -> String {
        let m = model.to_lowercase();

        // Check for thinking models first (highest priority)
        if m.contains("thinking") {
            if m.contains("opus") {
                return "opus-thinking".to_string();
            }
            if m.contains("sonnet") {
                return "sonnet-thinking".to_string();
            }
            // Default thinking
            return "thinking".to_string();
        }

        // Claude models
        if m.contains("opus") {
            return "opus".to_string();
        }
        if m.contains("sonnet") {
            return "sonnet".to_string();
        }
        if m.contains("haiku") {
            return "haiku".to_string();
        }

        // Gemini models
        if m.contains("gemini") {
            if m.contains("flash") {
                return "gemini-flash".to_string();
            }
            if m.contains("pro") {
                return "gemini-pro".to_string();
            }
            return "gemini".to_string();
        }

        // Fallback: use full model name
        m
    }

    /// Get concurrency limit for a model
    fn get_concurrency(&self, model_key: &str) -> usize {
        self.config
            .model_limits
            .get(model_key)
            .copied()
            .unwrap_or(self.config.default_concurrency)
    }

    /// Acquire a permit to make a request for the given model.
    /// This will block if the model's concurrency limit is reached.
    pub async fn acquire(&self, model: &str) -> ModelQueuePermit {
        let model_key = self.normalize_model_key(model);

        let semaphore: Arc<Semaphore> = {
            // Fast path: check if semaphore exists
            let read = self.semaphores.read().unwrap();
            if let Some(sem) = read.get(&model_key) {
                sem.clone()
            } else {
                drop(read);
                // Slow path: create new semaphore
                let mut write = self.semaphores.write().unwrap();
                // Double-check after acquiring write lock
                if let Some(sem) = write.get(&model_key) {
                    sem.clone()
                } else {
                    let concurrency = self.get_concurrency(&model_key);
                    let sem = Arc::new(Semaphore::new(concurrency));
                    tracing::info!(
                        "[ModelQueue] Created semaphore for '{}' with concurrency={}",
                        model_key,
                        concurrency
                    );
                    write.insert(model_key.clone(), sem.clone());
                    sem
                }
            }
        };

        let permit = semaphore
            .clone()
            .acquire_owned()
            .await
            .expect("Semaphore closed unexpectedly");

        tracing::debug!(
            "[ModelQueue] Acquired permit for '{}' (available: {})",
            model_key,
            semaphore.available_permits()
        );

        ModelQueuePermit {
            _permit: permit,
            model_key,
        }
    }

    /// Try to acquire a permit without blocking.
    /// Returns None if the model's concurrency limit is reached.
    #[allow(dead_code)]
    pub async fn try_acquire(&self, model: &str) -> Option<ModelQueuePermit> {
        let model_key = self.normalize_model_key(model);

        let semaphore: Arc<Semaphore> = {
            let read = self.semaphores.read().unwrap();
            read.get(&model_key)?.clone()
        };

        let permit = semaphore.clone().try_acquire_owned().ok()?;

        Some(ModelQueuePermit {
            _permit: permit,
            model_key,
        })
    }

    /// Get current queue status for monitoring
    #[allow(dead_code)]
    pub fn get_status(&self) -> HashMap<String, QueueStatus> {
        let read = self.semaphores.read().unwrap();
        read.iter()
            .map(|(key, sem): (&String, &Arc<Semaphore>)| {
                let concurrency = self.get_concurrency(key);
                let available = sem.available_permits();
                (
                    key.clone(),
                    QueueStatus {
                        concurrency,
                        available,
                        in_use: concurrency.saturating_sub(available),
                    },
                )
            })
            .collect()
    }
}

impl Default for ModelQueue {
    fn default() -> Self {
        Self::new()
    }
}

/// A permit that allows making a request to a specific model.
/// The permit is automatically released when dropped.
pub struct ModelQueuePermit {
    _permit: OwnedSemaphorePermit,
    model_key: String,
}

impl Drop for ModelQueuePermit {
    fn drop(&mut self) {
        tracing::debug!("[ModelQueue] Released permit for '{}'", self.model_key);
    }
}

/// Queue status for a single model
#[derive(Debug, Clone)]
pub struct QueueStatus {
    pub concurrency: usize,
    pub available: usize,
    pub in_use: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_model_key() {
        let queue = ModelQueue::new();

        assert_eq!(queue.normalize_model_key("claude-opus-4-5-20251101"), "opus");
        assert_eq!(queue.normalize_model_key("claude-opus-4-5-thinking"), "opus-thinking");
        assert_eq!(queue.normalize_model_key("claude-sonnet-4-5"), "sonnet");
        assert_eq!(queue.normalize_model_key("gemini-2.5-flash"), "gemini-flash");
        assert_eq!(queue.normalize_model_key("gemini-2.5-pro"), "gemini-pro");
    }

    #[tokio::test]
    async fn test_acquire_permit() {
        let queue = ModelQueue::new();

        // Opus-thinking should have concurrency=1
        let permit1 = queue.acquire("claude-opus-4-5-thinking").await;

        // Try to acquire another - should not be available immediately
        assert!(queue.try_acquire("claude-opus-4-5-thinking").await.is_none());

        // Drop the first permit
        drop(permit1);

        // Now we should be able to acquire
        assert!(queue.try_acquire("claude-opus-4-5-thinking").await.is_some());
    }

    #[tokio::test]
    async fn test_different_models_independent() {
        let queue = ModelQueue::new();

        // Acquire opus-thinking (concurrency=1)
        let _opus_permit = queue.acquire("claude-opus-4-5-thinking").await;

        // Sonnet should still be available (different queue)
        assert!(queue.try_acquire("claude-sonnet-4-5").await.is_some());
    }
}
