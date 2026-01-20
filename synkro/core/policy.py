"""Policy document handling with multi-format support."""

from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, Field

from synkro.errors import FileNotFoundError as SynkroFileNotFoundError
from synkro.errors import PolicyTooShortError

MIN_POLICY_WORDS = 10  # Minimum words for meaningful generation

# Supported file extensions
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


class Policy(BaseModel):
    """
    A policy document to generate training data from.

    Supports loading from multiple formats:
    - Plain text (.txt, .md)
    - PDF documents (.pdf) - via pymupdf
    - Word documents (.docx) - via mammoth

    Can load from:
    - Single file: Policy.from_file("compliance.pdf")
    - Folder of files: Policy.from_file("policies/")
    - Multiple files: Policy.from_files(["doc1.pdf", "doc2.docx"])
    - URL: Policy.from_url("https://example.com/policy")

    Examples:
        >>> # From text
        >>> policy = Policy(text="All expenses over $50 require approval")

        >>> # From single file
        >>> policy = Policy.from_file("compliance.pdf")

        >>> # From folder (loads all supported files)
        >>> policy = Policy.from_file("policies/")

        >>> # From multiple files
        >>> policy = Policy.from_files(["doc1.pdf", "doc2.docx", "doc3.txt"])

        >>> # From URL
        >>> policy = Policy.from_url("https://example.com/policy")
    """

    text: str = Field(description="Full policy text in markdown format")
    source: str | None = Field(default=None, description="Source file path or URL")

    @classmethod
    def from_file(cls, path: str | Path) -> "Policy":
        """
        Load policy from a file or folder.

        Supports: .txt, .md, .pdf, .docx

        If path is a directory, loads all supported files from that directory.
        If path is a file, loads that single file.

        Args:
            path: Path to the policy file or folder containing policy files

        Returns:
            Policy object with extracted text (combined if multiple files)

        Examples:
            >>> # Single file
            >>> policy = Policy.from_file("compliance.pdf")
            >>> len(policy.text) > 0
            True

            >>> # Folder of documents
            >>> policy = Policy.from_file("policies/")
            >>> len(policy.text) > 0
            True
        """
        path = Path(path)

        if not path.exists():
            # Find similar files to suggest
            similar = []
            if path.parent.exists():
                for ext in SUPPORTED_EXTENSIONS:
                    similar.extend(path.parent.glob(f"*{ext}"))
            similar_names = [str(f.name) for f in similar[:5]]
            raise SynkroFileNotFoundError(str(path), similar_names if similar_names else None)

        # If it's a directory, load all supported files
        if path.is_dir():
            return cls._from_folder(path)

        # Otherwise, load single file
        return cls._load_single_file(path)

    @classmethod
    def _load_single_file(cls, path: Path) -> "Policy":
        """
        Load a single policy file.

        Args:
            path: Path to a single file

        Returns:
            Policy object with extracted text
        """
        suffix = path.suffix.lower()

        if suffix in (".txt", ".md"):
            return cls(text=path.read_text(), source=str(path))

        if suffix == ".pdf":
            return cls._from_pdf(path)

        if suffix == ".docx":
            return cls._from_docx(path)

        raise ValueError(f"Unsupported file type: {suffix}. Use .txt, .md, .pdf, or .docx")

    @classmethod
    def _from_folder(cls, folder_path: Path) -> "Policy":
        """
        Load all supported files from a folder.

        Args:
            folder_path: Path to folder containing policy files

        Returns:
            Policy object with combined text from all files

        Raises:
            ValueError: If no supported files found in folder
        """
        # Find all supported files in folder (non-recursive)
        files = [
            f
            for f in folder_path.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        if not files:
            raise ValueError(
                f"No supported policy files found in {folder_path}. "
                f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}"
            )

        # Sort files for consistent ordering
        files.sort(key=lambda f: f.name)

        # Load each file and combine
        return cls.from_files(files, source_prefix=str(folder_path))

    @classmethod
    def from_files(
        cls,
        paths: Sequence[str | Path],
        separator: str = "\n\n---\n\n",
        source_prefix: str | None = None,
    ) -> "Policy":
        """
        Load and combine multiple policy documents.

        Args:
            paths: List of paths to policy files
            separator: String to separate documents (default: "\\n\\n---\\n\\n")
            source_prefix: Optional prefix for source description (e.g., folder path)

        Returns:
            Policy object with combined text from all files

        Raises:
            ValueError: If paths list is empty
            FileNotFoundError: If any file doesn't exist

        Examples:
            >>> # Multiple files
            >>> policy = Policy.from_files(["doc1.pdf", "doc2.docx", "doc3.txt"])
            >>> len(policy.text) > 0
            True

            >>> # With custom separator
            >>> policy = Policy.from_files(
            ...     ["part1.txt", "part2.txt"],
            ...     separator="\\n\\n=== NEXT DOCUMENT ===\\n\\n"
            ... )
        """
        if not paths:
            raise ValueError("paths list cannot be empty")

        texts: list[str] = []
        sources: list[str] = []

        for path in paths:
            path_obj = Path(path)

            if not path_obj.exists():
                raise SynkroFileNotFoundError(str(path_obj), None)

            if not path_obj.is_file():
                raise ValueError(f"Path is not a file: {path_obj}")

            # Load the file
            policy = cls._load_single_file(path_obj)
            texts.append(policy.text)
            sources.append(str(path_obj))

        # Combine texts
        combined_text = separator.join(texts)

        # Create source description
        if source_prefix:
            combined_source = f"{source_prefix} ({len(sources)} files: {', '.join(Path(s).name for s in sources)})"
        else:
            combined_source = (
                f"multiple_files ({len(sources)} files: {', '.join(Path(s).name for s in sources)})"
            )

        return cls(text=combined_text, source=combined_source)

    @classmethod
    def _from_pdf(cls, path: Path) -> "Policy":
        """
        Parse PDF to text using pymupdf.

        Args:
            path: Path to PDF file

        Returns:
            Policy with extracted text
        """
        try:
            import pymupdf

            doc = pymupdf.open(str(path))
            text = "\n\n".join(page.get_text() for page in doc)
            doc.close()
            return cls(text=text, source=str(path))
        except ImportError:
            raise ImportError(
                "pymupdf is required for PDF support. " "Install with: pip install pymupdf"
            )

    @classmethod
    def _from_docx(cls, path: Path) -> "Policy":
        """
        Parse DOCX to markdown using mammoth.

        Args:
            path: Path to DOCX file

        Returns:
            Policy with extracted markdown text
        """
        try:
            import mammoth

            with open(path, "rb") as f:
                result = mammoth.convert_to_markdown(f)
                return cls(text=result.value, source=str(path))
        except ImportError:
            raise ImportError(
                "mammoth is required for DOCX support. " "Install with: pip install mammoth"
            )

    @classmethod
    def from_url(cls, url: str) -> "Policy":
        """
        Fetch and parse policy from a URL.

        Extracts main content and converts to markdown.

        Args:
            url: URL to fetch

        Returns:
            Policy with extracted content

        Example:
            >>> policy = Policy.from_url("https://example.com/terms")
        """
        try:
            import html2text
            import httpx
            from bs4 import BeautifulSoup

            response = httpx.get(url, follow_redirects=True)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Remove scripts, styles, nav, footer
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            # Convert to markdown
            h = html2text.HTML2Text()
            h.ignore_links = False
            h.ignore_images = True
            markdown = h.handle(str(soup))

            return cls(text=markdown, source=url)
        except ImportError as e:
            missing = str(e).split("'")[1] if "'" in str(e) else "required packages"
            raise ImportError(
                f"{missing} is required for URL support. "
                "This should be installed automatically with synkro. "
                "If you see this error, try: pip install --upgrade synkro"
            )

    @property
    def word_count(self) -> int:
        """Get the word count of the policy."""
        return len(self.text.split())

    @property
    def char_count(self) -> int:
        """Get the character count of the policy."""
        return len(self.text)

    def validate_length(self) -> None:
        """
        Validate that the policy has enough content for meaningful generation.

        Raises:
            PolicyTooShortError: If policy is too short
        """
        if self.word_count < MIN_POLICY_WORDS:
            raise PolicyTooShortError(self.word_count)

    def __str__(self) -> str:
        """String representation showing source and length."""
        source = self.source or "inline"
        return f"Policy(source={source}, words={self.word_count})"

    def __repr__(self) -> str:
        return self.__str__()
