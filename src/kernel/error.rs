// Scope: Pipeline error types

#[derive(Debug, Clone)]
pub enum PipelineError {
    StageError(&'static str),
}
