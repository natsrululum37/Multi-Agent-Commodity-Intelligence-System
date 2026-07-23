"""Base class untuk semua agent."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseAgent(ABC):
    """Abstract base class untuk agent."""

    def __init__(self, name: str = "base_agent"):
        """Inisialisasi agent.

        Args:
            name: Nama agent.
        """
        self.name = name
        self.status: str = "initialized"
        self.last_result: Optional[Any] = None
        self.metadata: Dict[str, Any] = {}

    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """Proses input data.

        Args:
            input_data: Data yang akan diproses.

        Returns:
            Hasil proses.
        """
        pass

    def get_status(self) -> Dict[str, Any]:
        """Dapatkan status agent.

        Returns:
            Dictionary berisi informasi status.
        """
        return {
            "name": self.name,
            "status": self.status,
            "last_result_type": type(self.last_result).__name__ if self.last_result else None,
        }

    def reset(self) -> None:
        """Reset agent ke state awal."""
        self.status = "initialized"
        self.last_result = None
