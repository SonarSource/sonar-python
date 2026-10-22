from __future__ import annotations
from typing import TYPE_CHECKING
import httpx

if TYPE_CHECKING:
    from httpx import AsyncClient


async def parameter_response_hook(response: httpx.Response) -> None:  # Noncompliant
    print(response.status_code)


def install_async_client_hook(client: AsyncClient) -> None:
    previous_hooks = client.event_hooks
    previous_response_hooks = previous_hooks["response"]
    client.event_hooks = {
        **previous_hooks,
        "response": [*previous_response_hooks, parameter_response_hook],
    }


async def sync_client_hook(response) -> None:  # Noncompliant
    print(response.status_code)


def install_sync_client_hook(client: httpx.Client) -> None:
    client.event_hooks = {"response": [sync_client_hook]}


async def unknown_client_hook(response) -> None:  # Noncompliant
    print(response.status_code)


def install_unknown_client_hook(client) -> None:
    client.event_hooks = {"response": [unknown_client_hook]}


async def dynamic_key_hook(response) -> None:  # Noncompliant
    print(response.status_code)


def install_dynamic_key_hook(key: str) -> None:
    client = httpx.AsyncClient()
    client.event_hooks = {key: [dynamic_key_hook]}


async def overwritten_hook(response) -> None:  # Noncompliant
    print(response.status_code)


def install_overwritten_hook() -> None:
    client = httpx.AsyncClient()
    client.event_hooks = {"response": [overwritten_hook]}
    client.event_hooks = {"response": []}


async def receiver_alias_hook(response) -> None:  # Noncompliant
    print(response.status_code)


def install_receiver_alias_hook() -> None:
    client = httpx.AsyncClient()
    alias = client
    alias.event_hooks = {"response": [receiver_alias_hook]}


async def escaped_local_client_hook(response) -> None:  # Noncompliant
    print(response.status_code)


def install_escaped_local_client_hook() -> None:
    client = httpx.AsyncClient()
    client.event_hooks = {"response": [escaped_local_client_hook]}
    consume(client)


async def assigned_response_hook(response) -> None:
    print(response.status_code)


async def send_request() -> None:
    client = httpx.AsyncClient()
    previous_hooks = client.event_hooks
    previous_response_hooks = previous_hooks["response"]
    client.event_hooks = {
        **previous_hooks,
        "response": [*previous_response_hooks, assigned_response_hook],
    }
    await client.get("https://example.org")


async def assigned_request_hook(request) -> None:  # Noncompliant
    print(request.url)


request_client = httpx.AsyncClient()
request_client.event_hooks = {"request": [assigned_request_hook], "unrelated": []}


async def duplicate_key_hook(response) -> None:  # Noncompliant
    print(response.status_code)


duplicate_client = httpx.AsyncClient()
duplicate_client.event_hooks = {"response": [duplicate_key_hook], "response": []}


async def later_unpack_hook(response) -> None:  # Noncompliant
    print(response.status_code)


unpack_client = httpx.AsyncClient()
unpack_client.event_hooks = {"response": [later_unpack_hook], **other_hooks}


async def later_dynamic_key_hook(response) -> None:  # Noncompliant
    print(response.status_code)


dynamic_client = httpx.AsyncClient()
dynamic_client.event_hooks = {"response": [later_dynamic_key_hook], unknown_key: []}


async def escaped_receiver_hook(response) -> None:  # Noncompliant
    print(response.status_code)


escaped_client = httpx.AsyncClient()
other_client = escaped_client
escaped_client.event_hooks = {"response": [escaped_receiver_hook]}


async def argument_receiver_hook(response) -> None:  # Noncompliant
    print(response.status_code)


argument_client = httpx.AsyncClient()
argument_client.event_hooks = {"response": [argument_receiver_hook]}
modify_client(argument_client)


async def subscript_overwrite_hook(response) -> None:  # Noncompliant
    print(response.status_code)


subscript_client = httpx.AsyncClient()
subscript_client.event_hooks = {"response": [subscript_overwrite_hook]}
subscript_client.event_hooks["response"] = []


async def augmented_overwrite_hook(response) -> None:  # Noncompliant
    print(response.status_code)


augmented_client = httpx.AsyncClient()
augmented_client.event_hooks = {"response": [augmented_overwrite_hook]}
augmented_client.event_hooks["response"] += []


async def deleted_hook(response) -> None:  # Noncompliant
    print(response.status_code)


deleted_client = httpx.AsyncClient()
deleted_client.event_hooks = {"response": [deleted_hook]}
del deleted_client.event_hooks["response"]


async def mutating_method_hook(response) -> None:  # Noncompliant
    print(response.status_code)


mutating_client = httpx.AsyncClient()
mutating_client.event_hooks = {"response": [mutating_method_hook]}
mutating_client.event_hooks.clear()


async def chained_assignment_hook(response) -> None:  # Noncompliant
    print(response.status_code)


chained_client = httpx.AsyncClient()
unrelated_hooks = chained_client.event_hooks = {"response": [chained_assignment_hook]}


class CustomAsyncClient(httpx.AsyncClient):
    pass


async def subclass_property_hook(response) -> None:  # Noncompliant
    print(response.status_code)


subclass_client = CustomAsyncClient()
subclass_client.event_hooks = {"response": [subclass_property_hook]}


async def rebound_hook(response) -> None:  # Noncompliant
    print(response.status_code)


rebound_client = httpx.AsyncClient()
rebound_hook = unrelated_callback
rebound_client.event_hooks = {"response": [rebound_hook]}


async def nested_hook(response) -> None:  # Noncompliant
    print(response.status_code)


outer_client = httpx.AsyncClient()


def deferred_registration() -> None:
    outer_client.event_hooks = {"response": [nested_hook]}


from httpx import AsyncClient as HttpxClient


async def imported_constructor_hook(response) -> None:  # Noncompliant
    print(response.status_code)


imported_client = HttpxClient()
imported_client.event_hooks = {"request": [], "response": [imported_constructor_hook]}


async def final_duplicate_hook(response) -> None:  # Noncompliant
    print(response.status_code)


final_client = httpx.AsyncClient()
final_client.event_hooks = {"response": [], "res" "ponse": [final_duplicate_hook]}


async def factory_receiver_hook(response) -> None:  # Noncompliant
    print(response.status_code)


def make_client() -> httpx.AsyncClient:
    return httpx.AsyncClient()


factory_client = make_client()
factory_client.event_hooks = {"response": [factory_receiver_hook]}


async def rebound_receiver_hook(response) -> None:  # Noncompliant
    print(response.status_code)


rebound_receiver = httpx.AsyncClient()
rebound_receiver = httpx.Client()
rebound_receiver.event_hooks = {"response": [rebound_receiver_hook]}


async def indirect_list_hook(response) -> None:  # Noncompliant
    print(response.status_code)


indirect_list_client = httpx.AsyncClient()
indirect_list_client.event_hooks = {"response": [wrap(indirect_list_hook)]}


async def starred_hook(response) -> None:  # Noncompliant
    print(response.status_code)


starred_client = httpx.AsyncClient()
starred_client.event_hooks = {"response": [*starred_hook]}


async def indirect_dictionary_hook(response) -> None:  # Noncompliant
    print(response.status_code)


indirect_dictionary_client = httpx.AsyncClient()
indirect_dictionary_client.event_hooks = wrap({"response": [indirect_dictionary_hook]})


async def creation_alias_hook(response) -> None:  # Noncompliant
    print(response.status_code)


creation_alias = creation_client = httpx.AsyncClient()
creation_client.event_hooks = {"response": [creation_alias_hook]}


async def private_mutation_hook(response) -> None:  # Noncompliant
    print(response.status_code)


private_client = httpx.AsyncClient()
private_client.event_hooks = {"response": [private_mutation_hook]}
private_client._event_hooks = {}


async def bound_method_escape_hook(response) -> None:  # Noncompliant
    print(response.status_code)


bound_method_client = httpx.AsyncClient()
bound_method_client.event_hooks = {"response": [bound_method_escape_hook]}
bound_method_alias = bound_method_client.get.__self__
bound_method_alias.event_hooks = {}


async def captured_method_hook(response) -> None:  # Noncompliant
    print(response.status_code)


captured_method_client = httpx.AsyncClient()
captured_method_client.event_hooks = {"response": [captured_method_hook]}
modify_method(captured_method_client.get)


async def interpolated_key_hook(response) -> None:  # Noncompliant
    print(response.status_code)


interpolated_client = httpx.AsyncClient()
interpolated_client.event_hooks = {"response": [interpolated_key_hook], f"{key}": []}


async def escaped_key_hook(response) -> None:  # Noncompliant
    print(response.status_code)


escaped_key_client = httpx.AsyncClient()
escaped_key_client.event_hooks = {"response": [escaped_key_hook], "\x72esponse": []}


async def bytes_key_hook(response) -> None:  # Noncompliant
    print(response.status_code)


bytes_client = httpx.AsyncClient()
bytes_client.event_hooks = {b"response": [bytes_key_hook]}


async def formatted_literal_key_hook(response) -> None:  # Noncompliant
    print(response.status_code)


formatted_literal_client = httpx.AsyncClient()
formatted_literal_client.event_hooks = {f"response": [formatted_literal_key_hook]}


async def template_key_hook(response) -> None:  # Noncompliant
    print(response.status_code)


template_client = httpx.AsyncClient()
template_client.event_hooks = {"response": [template_key_hook], t"{key}": []}


async def annotated_client_hook(response) -> None:  # Noncompliant
    print(response.status_code)


annotated_client: httpx.AsyncClient = httpx.AsyncClient()
annotated_client.event_hooks = {"response": [annotated_client_hook]}


async def tuple_client_hook(response) -> None:  # Noncompliant
    print(response.status_code)


(tuple_client, other_client) = httpx.AsyncClient()
tuple_client.event_hooks = {"response": [tuple_client_hook]}


async def reused_for_client_hook(response):  # Noncompliant
    print(response.status_code)


def configure_reused_for_client():
    client = httpx.AsyncClient()
    for enabled in (True, False):
        previous_hooks = client.event_hooks
        previous_hooks["response"].clear()
        if enabled:
            client.event_hooks = {"response": [reused_for_client_hook]}


async def reused_while_client_hook(response):  # Noncompliant
    print(response.status_code)


def configure_reused_while_client():
    client = httpx.AsyncClient()
    remaining = 2
    while remaining:
        previous_hooks = client.event_hooks
        previous_hooks["response"].clear()
        if remaining == 2:
            client.event_hooks = {"response": [reused_while_client_hook]}
        remaining -= 1


async def reused_async_for_client_hook(response):  # Noncompliant
    print(response.status_code)


async def configure_reused_async_for_client(settings):
    client = httpx.AsyncClient()
    async for enabled in settings:
        previous_hooks = client.event_hooks
        previous_hooks["response"].clear()
        if enabled:
            client.event_hooks = {"response": [reused_async_for_client_hook]}


async def nested_loop_client_hook(response):  # Noncompliant
    print(response.status_code)


def configure_nested_loop_client():
    for unused in range(2):
        client = httpx.AsyncClient()
        for enabled in (True, False):
            previous_hooks = client.event_hooks
            previous_hooks["response"].clear()
            if enabled:
                client.event_hooks = {"response": [nested_loop_client_hook]}


async def conditional_creation_hook(response):  # Noncompliant
    print(response.status_code)


def configure_conditional_creation():
    for enabled in (True, False):
        if enabled:
            client = httpx.AsyncClient()
        previous_hooks = client.event_hooks
        previous_hooks["response"].clear()
        if enabled:
            client.event_hooks = {"response": [conditional_creation_hook]}


async def reversed_creation_hook(response):  # Noncompliant
    print(response.status_code)


def configure_reversed_creation():
    for enabled in (False, True):
        if enabled:
            client.event_hooks = {"response": [reversed_creation_hook]}
        client = httpx.AsyncClient()


async def fresh_for_client_hook(response):
    print(response.status_code)


def configure_fresh_for_client():
    for enabled in (True, False):
        client = httpx.AsyncClient()
        previous_hooks = client.event_hooks
        if enabled:
            client.event_hooks = {"response": [fresh_for_client_hook]}
        previous_hooks["response"].clear()


async def read_before_creation_hook(response):  # Noncompliant
    print(response.status_code)


def configure_read_before_creation():
    for enabled in (True, False):
        if not enabled:
            previous_hooks = client.event_hooks
            previous_hooks["response"].clear()
        client = httpx.AsyncClient()
        if enabled:
            client.event_hooks = {"response": [read_before_creation_hook]}


async def fresh_while_client_hook(response):
    print(response.status_code)


def configure_fresh_while_client():
    remaining = 2
    while remaining:
        client = httpx.AsyncClient()
        previous_hooks = client.event_hooks
        client.event_hooks = {"response": [fresh_while_client_hook]}
        previous_hooks["response"].clear()
        remaining -= 1


async def unrelated_loop_client_hook(response):
    print(response.status_code)


def configure_unrelated_loop_client():
    client = httpx.AsyncClient()
    for value in range(2):
        print(value)
    client.event_hooks = {"response": [unrelated_loop_client_hook]}


async def known_methods_hook(response):
    print(response.status_code)


async def use_known_client_methods():
    client = httpx.AsyncClient()
    client.event_hooks = {"response": [known_methods_hook]}
    request = client.build_request("GET", "https://example.org")
    await client.send(request)
    await client.request("GET", "https://example.org")
    await client.get("https://example.org")
    await client.post("https://example.org")
    await client.put("https://example.org")
    await client.patch("https://example.org")
    await client.delete("https://example.org")
    await client.head("https://example.org")
    await client.options("https://example.org")
    async with client.stream("GET", "https://example.org") as response:
        print(response.status_code)
    await client.aclose()


async def unknown_method_hook(response):  # Noncompliant
    print(response.status_code)


unknown_method_client = httpx.AsyncClient()
unknown_method_client.event_hooks = {"response": [unknown_method_hook]}
unknown_method_client.reset_event_hooks()


async def private_method_hook(response):  # Noncompliant
    print(response.status_code)


private_method_client = httpx.AsyncClient()
private_method_client.event_hooks = {"response": [private_method_hook]}
private_method_client._send_single_request(request)


async def class_overwritten_hook(response) -> None:  # Noncompliant
    print(response.status_code)


class Clients:
    client = httpx.AsyncClient()
    client.event_hooks = {"response": [class_overwritten_hook]}


Clients.client.event_hooks = {}


async def class_escaped_hook(response) -> None:  # Noncompliant
    print(response.status_code)


class EscapedClients:
    client = httpx.AsyncClient()
    client.event_hooks = {"response": [class_escaped_hook]}


modify_client(EscapedClients.client)


async def nested_class_hook(response) -> None:  # Noncompliant
    print(response.status_code)


def configure_nested_class() -> None:
    class NestedClients:
        client = httpx.AsyncClient()
        client.event_hooks = {"response": [nested_class_hook]}


async def local_dynamic_key_hook(response) -> None:  # Noncompliant
    print(response.status_code)


def configure_local_dynamic_key(key: str) -> None:
    client = httpx.AsyncClient()
    client.event_hooks = {"response": [local_dynamic_key_hook], key: []}


async def local_final_duplicate_hook(response) -> None:
    print(response.status_code)


def configure_local_final_duplicate() -> None:
    client = httpx.AsyncClient()
    client.event_hooks = {"response": [], "res" "ponse": [local_final_duplicate_hook]}
