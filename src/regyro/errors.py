class RegyroError(Exception):
    """Base class for every error regyro raises deliberately.

    Lets the CLI report expected failures without catching bugs it should not hide.
    """
