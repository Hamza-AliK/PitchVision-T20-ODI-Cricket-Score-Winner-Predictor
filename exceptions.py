class DataUnavailableError(Exception):
    """Custom exception raised when neither network nor cache can supply data."""
    def __init__(self, message="Data is unavailable from both network and local cache."):
        super().__init__(message)