use std::fmt;

#[derive(Debug)]
pub enum Error {
    Parsing,
    Initialise(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parsing => write!(f, "Failed to parse"),
            Self::Initialise(err) => write!(f, "Failed to initialise: {err}"),
        }
    }
}

impl std::error::Error for Error {}
