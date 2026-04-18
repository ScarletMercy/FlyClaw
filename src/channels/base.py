from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional


class Channel(ABC):
    """Abstract base class for messaging channels."""

    @abstractmethod
    def set_message_callback(self, callback: Callable) -> None:
        """Set the callback function for incoming messages."""
        pass

    @abstractmethod
    async def start(self) -> None:
        """Start the channel and begin listening for messages."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the channel and cleanup resources."""
        pass

    @abstractmethod
    async def send_text(
        self,
        chat_id: str,
        text: str,
        reply_to: Optional[str] = None,
    ) -> Any:
        """Send a text message to a chat."""
        pass

    @abstractmethod
    async def send_image(self, chat_id: str, image_key: str) -> bool:
        """Send an image message to a chat."""
        pass

    @abstractmethod
    async def send_file(self, chat_id: str, file_key: str) -> bool:
        """Send a file message to a chat."""
        pass

    @abstractmethod
    async def send_card(
        self,
        chat_id: str,
        card_content: str,
        reply_to: Optional[str] = None,
    ) -> Any:
        """Send an interactive card to a chat."""
        pass
