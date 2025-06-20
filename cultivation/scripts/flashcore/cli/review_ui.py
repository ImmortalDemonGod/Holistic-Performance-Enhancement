"""
Command-line interface for reviewing flashcards.
"""

import logging
import time
from datetime import date

from rich.console import Console
from rich.panel import Panel

from cultivation.scripts.flashcore.card import Card
from cultivation.scripts.flashcore.review_manager import ReviewSessionManager

logger = logging.getLogger(__name__)
console = Console()


def _get_user_rating() -> int:
    """Prompts the user for a rating and validates it."""
    while True:
        try:
            rating_str = console.input(
                "[bold]Rating (1:Again, 2:Hard, 3:Good, 4:Easy): [/bold]"
            )
            rating = int(rating_str)
            if 1 <= rating <= 4:
                return rating
            else:
                console.print(
                    "[bold red]Invalid rating. Please enter a number between 1 and 4.[/bold red]"
                )
        except (ValueError, TypeError):
            console.print("[bold red]Invalid input. Please enter a number.[/bold red]")


def _display_card(card: Card) -> int:
    """
    Displays the front and back of a card, waiting for user input.
    Returns the response time in milliseconds.
    """
    console.print(Panel(card.front, title="Front", border_style="green"))
    start_time = time.time()
    console.input("[italic]Press Enter to see the back...[/italic]")
    end_time = time.time()
    console.print(Panel(card.back, title="Back", border_style="blue"))
    return int((end_time - start_time) * 1000)


def start_review_flow(manager: ReviewSessionManager) -> None:
    """
    Manages the command-line review session flow.

    Args:
        manager: An instance of ReviewSessionManager.
    """
    console.print("[bold cyan]Starting review session...[/bold cyan]")
    manager.start_session()

    due_cards_count = len(manager.review_queue)
    if due_cards_count == 0:
        console.print("[bold yellow]No cards are due for review.[/bold yellow]")
        console.print("[bold cyan]Review session finished.[/bold cyan]")
        return

    reviewed_count = 0
    while (card := manager.get_next_card()) is not None:
        reviewed_count += 1
        console.rule(f"[bold]Card {reviewed_count} of {due_cards_count}[/bold]")

        resp_ms = _display_card(card)
        rating = _get_user_rating()
        
        updated_card = manager.submit_review(
            card_uuid=card.uuid, rating=rating, resp_ms=resp_ms
        )

        if updated_card and updated_card.next_due_date:
            days_until_due = (updated_card.next_due_date - date.today()).days
            due_date_str = updated_card.next_due_date.strftime('%Y-%m-%d')
            console.print(
                f"[green]Reviewed.[/green] Next due in [bold]{days_until_due} days[/bold] on {due_date_str}."
            )
        else:
            console.print("[bold red]Error submitting review. Card will be reviewed again later.[/bold red]")
        console.print("") # Add a blank line for spacing

    console.print("[bold cyan]Review session finished. Well done![/bold cyan]")
