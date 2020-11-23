use std::path::Path;
use either::Either;




/**
 * Module-level error codes.
 */
#[derive(thiserror::Error, Debug)]
pub enum Error {

    #[error("no such file {0}")]
    NoSuchFile(String),

    #[error("no such directory {0}")]
    NoSuchDirectory(String),

    #[error("no such file or directory {0}")]
    NoSuchFileOrDirectory(String),

    #[error("found a file {0} where a directory was expected")]
    FileExistsWhereDirectoryExpected(String),

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
pub struct File
{
    path_string: String,
}




/**
 * A struct representing a file system location that has already been confirmed
 * to exist as a directory.
 */
pub struct Directory
{
    path_string: String,
}




// ============================================================================
impl File
{
    pub fn from_path(path: &Path) -> Result<File, Error>
    {
        let path_str = path.to_str().unwrap().into();

        if path.is_file() {
            Ok(File{path_string: path_str})
        } else if path.is_dir() {
            Err(Error::DirectoryExistsWhereFileExpected(path_str))
        } else {            
            Err(Error::NoSuchFile(path_str))
        }
    }

    pub fn from_str(path_str: &str) -> Result<File, Error>
    {
        File::from_path(Path::new(path_str))
    }

    pub fn as_path(&self) -> &Path
    {
        Path::new(&self.path_string)
    }

    pub fn parent(&self) -> Directory
    {
        Directory{path_string: self.as_path().to_path_buf().parent().unwrap().to_str().unwrap().into()}
    }
}




// ============================================================================
impl Directory
{
    pub fn from_path(path: &Path) -> Result<Directory, Error>
    {
        let path_str = path.to_str().unwrap().into();

        if path.is_dir() {
            Ok(Directory{path_string: path_str})
        } else if path.is_file() {
            Err(Error::FileExistsWhereDirectoryExpected(path_str))
        } else {            
            Err(Error::NoSuchFile(path_str))
        }
    }

    pub fn from_str(path_str: &str) -> Result<Directory, Error>
    {
        Directory::from_path(Path::new(path_str))
    }

    pub fn require(path_string: String) -> Result<Directory, Error>
    {
        std::fs::create_dir_all(&path_string).map_err(|_| Error::UnableToCreateDirectory(path_string.clone()))?;
        Ok(Directory{path_string: path_string})
    }

    pub fn remove(&self) -> Result<(), Error>
    {
        std::fs::remove_dir(self.as_path()).map_err(|_| Error::UnableToRemoveDirectory(self.path_string.clone()))
    }

    pub fn as_path(&self) -> &Path
    {
        Path::new(&self.path_string)
    }

    pub fn child(&self, filename: &str) -> String
    {
        let mut path_buf = self.as_path().to_path_buf();
        path_buf.push(filename);
        path_buf.to_str().unwrap().into()
    }

    pub fn existing_child(&self, filename: &str) -> Result<File, Error>
    {
        let mut path_buf = self.as_path().to_path_buf();
        path_buf.push(filename);

        File::from_path(path_buf.as_path())
    }

    pub fn most_recent_file_matching(&self, pattern: &str) -> Result<File, Error>
    {
        let mut path_buf = self.as_path().to_path_buf();
        path_buf.push(pattern);

        let matches = glob::glob(path_buf.to_str().unwrap())?;

        if let Some(result) = matches.filter_map(Result::ok).max() {
            Ok(File{path_string: result.to_str().unwrap().into()})
        } else {
            Err(Error::NoMatchingFiles(self.path_string.clone(), pattern.into()))
        }
    }
}




/**
 * Return either an existing file, or an existing directory, depending on
 * whether the file system status of the given path string. If no file or
 * directory exists at that location, return an error.
 */
pub fn file_or_directory(path_string: String) -> Result<Either<File, Directory>, Error>
{
    if let Ok(file) = File::from_str(&path_string) {
        Ok(Either::Left(file))
    } else if let Ok(directory) = Directory::from_str(&path_string) {
        Ok(Either::Right(directory))
    } else {
        Err(Error::NoSuchFileOrDirectory(path_string))
    }
}




// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    fn unable_to_create_dir() {
        assert!(Directory::require("/no_permissions_here".into()).is_err());
    }

    #[test]
    #[serial]
    fn able_to_require_dir() {
        let directory = Directory::require("test_dir".into()).unwrap();
        assert!(!directory.remove().is_err());
        assert!( directory.remove().is_err());
    }

    #[test]
    #[serial]
    fn illegal_glob_pattern() {
        let directory = Directory::require("test_dir".into()).unwrap();

        match directory.most_recent_file_matching("a**/b") {
            Err(Error::GlobPatternError(_)) => assert!(true),
            _ => assert!(false),
        }
        directory.remove().unwrap();
    }

    #[test]
    #[serial]
    fn unable_to_find_most_recent_file_matching() {
        let directory = Directory::require("test_dir".into()).unwrap();

        match directory.most_recent_file_matching("*") {
            Err(Error::NoMatchingFiles(_, _)) => assert!(true),
            _ => assert!(false),
        }
        directory.remove().unwrap();
    }
}
