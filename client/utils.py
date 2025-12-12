"""Utility functions for the Snorefox Med API client."""
from collections.abc import Callable
from typing import Any

from snorefox_med.client.snorefox_med_client import client

PAGE_SIZE = 1000


def get_concatenated_obj_list_from_all_pages(
    get_function: Callable[..., Any],
    api_client: client.AuthenticatedClient,
    **kwargs: Any,  # noqa: ANN401
) -> list[Any]:
    """Get the concatenated list from all pages of a list-based GET API response."""
    limit = kwargs.pop("limit", PAGE_SIZE)
    response = get_function(client=api_client, limit=limit, **kwargs)
    concatenated_obj_list: list[Any] = response.list_
    kwargs.pop("page", None)
    while response.current_page < response.pages:
        response = get_function(client=api_client, page=response.current_page + 1, limit=limit, **kwargs)
        concatenated_obj_list += response.list_

    return concatenated_obj_list
