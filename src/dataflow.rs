#![allow(unused)]




/**
 * Library-level error codes.
 */
#[derive(thiserror::Error, Debug)]
pub enum DataflowError {
    #[error("no such file {0}")]
    NoSuchFile(String),

    #[error("no such directory {0}")]
    NoSuchDirectory(String),

    #[error("found a directory {0} where a file was expected")]
    DirectoryExistsWhereFileExpected(String),

    #[error("unable to create directory {0}")]
    UnableToCreateDirectory(String),

    #[error("unable to remove directory {0}")]
    UnableToRemoveDirectory(String),

    #[error("directory {0} contained no files matching {1}")]
    NoMatchingFiles(String, String),

    #[error(transparent)]
    GlobPatternError(#[from] glob::PatternError),
}




/**
 * A struct representing a file system location that has already been confirmed
 * to exist as a file.
 */
pub struct ExistingFile
{
    path_string: String,
}




/**
 * A struct representing a file system location that has already been confirmed
 * to exist as a directory.
 */
pub struct ExistingDirectory
{
    path_string: String,
}




// ============================================================================
impl ExistingFile
{
    pub fn new(path_str: &str) -> Result<ExistingFile, DataflowError>
    {
        let path_buf = std::path::PathBuf::from(path_str);

        if path_buf.is_file() {
            Ok(ExistingFile{path_string: path_str.into()})
        } else if path_buf.is_dir() {
            Err(DataflowError::DirectoryExistsWhereFileExpected(path_str.into()))
        } else {            
            Err(DataflowError::NoSuchFile(path_str.into()))
        }
    }
}




// ============================================================================
impl ExistingDirectory
{
    pub fn require(path_str: &str) -> Result<ExistingDirectory, DataflowError>
    {
        std::fs::create_dir_all(path_str).map_err(|_| DataflowError::UnableToCreateDirectory(path_str.into()))?;
        Ok(ExistingDirectory{path_string: path_str.into()})
    }

    pub fn remove(&self) -> Result<(), DataflowError>
    {
        std::fs::remove_dir(self.as_path()).map_err(|_| DataflowError::UnableToRemoveDirectory(self.path_string.clone()))
    }

    pub fn as_path(&self) -> &std::path::Path
    {
        std::path::Path::new(&self.path_string)
    }

    pub fn most_recent_file_matching(&self, pattern: &str) -> Result<ExistingFile, DataflowError>
    {
        let mut path_buf = self.as_path().to_path_buf();
        path_buf.push(pattern);

        let matches = glob::glob(path_buf.to_str().unwrap())?;

        if let Some(result) = matches.filter_map(Result::ok).max() {
            Ok(ExistingFile{path_string: result.to_str().unwrap().into()})
        } else {
            Err(DataflowError::NoMatchingFiles(self.path_string.clone(), pattern.into()))
        }
    }
}




// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    fn unable_to_create_dir() {
        assert!(ExistingDirectory::require("/no_permissions_here").is_err());
    }

    #[test]
    #[serial]
    fn able_to_require_dir() {
        let directory = ExistingDirectory::require("test_dir").unwrap();
        assert!(!directory.remove().is_err());
        assert!( directory.remove().is_err());
    }

    #[test]
    #[serial]
    fn illegal_glob_pattern() {
        let directory = ExistingDirectory::require("test_dir").unwrap();

        match directory.most_recent_file_matching("a**/b") {
            Err(DataflowError::GlobPatternError(_)) => assert!(true),
            _ => assert!(false),
        }
        directory.remove().unwrap();
    }

    #[test]
    #[serial]
    fn unable_to_find_most_recent_file_matching() {
        let directory = ExistingDirectory::require("test_dir").unwrap();

        match directory.most_recent_file_matching("*") {
            Err(DataflowError::NoMatchingFiles(_, _)) => assert!(true),
            _ => assert!(false),
        }
        directory.remove().unwrap();
    }
}




// ============================================================================
pub trait SimulationData
{
    type SolutionState;
    type TimeSeriesSample;
    type TaskSchedule;
}


pub struct Dataflow
{
    restart: ExistingFile,
    outdir: ExistingDirectory,
}
