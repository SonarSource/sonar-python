def noncompliant_simple():
    with open("a.txt") as a:  # Noncompliant {{Combine these nested "with" statements into a single "with" with multiple contexts.}}
        with open("b.txt") as b:
#   ^^^^@-1
#       ^^^^@-1< {{Nested "with" statement}}
            process(a, b)


def noncompliant_without_as():
    with open("a.txt"):  # Noncompliant
        with open("b.txt"):
#   ^^^^@-1
#       ^^^^@-1< {{Nested "with" statement}}
            process()


def noncompliant_multiple_contexts_inner():
    with open("a.txt") as a:  # Noncompliant
        with open("b.txt") as b, open("c.txt") as c:
#   ^^^^@-1
#       ^^^^@-1< {{Nested "with" statement}}
            process(a, b, c)


def noncompliant_multiple_contexts_outer():
    with open("a.txt") as a, open("b.txt") as b:  # Noncompliant
        with open("c.txt") as c:
#   ^^^^@-1
#       ^^^^@-1< {{Nested "with" statement}}
            process(a, b, c)


def noncompliant_inline_inner_body():
    with open("a.txt") as a:  # Noncompliant
        with open("b.txt") as b: process(a, b)
#   ^^^^@-1
#       ^^^^@-1< {{Nested "with" statement}}


def noncompliant_deeply_nested():
    # One issue only: primary on outermost, secondaries on each nested with.
    with open("a.txt") as a:  # Noncompliant
        with open("b.txt") as b:
            with open("c.txt") as c:
                process(a, b, c)
#   ^^^^@-3
#       ^^^^@-3< {{Nested "with" statement}}
#           ^^^^@-3< {{Nested "with" statement}}


def noncompliant_four_levels():
    with open("a.txt") as a:  # Noncompliant
        with open("b.txt") as b:
            with open("c.txt") as c:
                with open("d.txt") as d:
                    process(a, b, c, d)
#   ^^^^@-4
#       ^^^^@-4< {{Nested "with" statement}}
#           ^^^^@-4< {{Nested "with" statement}}
#               ^^^^@-4< {{Nested "with" statement}}


def noncompliant_async_nested():
    async def inner():
        async with open_async("a.txt") as a:  # Noncompliant
            async with open_async("b.txt") as b:
#             ^^^^@-1
#                 ^^^^@-1< {{Nested "with" statement}}
                await process(a, b)


def noncompliant_async_deeply_nested():
    async def inner():
        async with open_async("a.txt") as a:  # Noncompliant
            async with open_async("b.txt") as b:
                async with open_async("c.txt") as c:
                    await process(a, b, c)
#             ^^^^@-3
#                 ^^^^@-3< {{Nested "with" statement}}
#                     ^^^^@-3< {{Nested "with" statement}}


def compliant_already_combined():
    with open("a.txt") as a, open("b.txt") as b:
        process(a, b)


def compliant_inner_not_only_statement():
    with open("a.txt") as a:
        prepare(a)
        with open("b.txt") as b:
            process(a, b)


def compliant_statement_after_inner_with():
    with open("a.txt") as a:
        with open("b.txt") as b:
            process(a, b)
        cleanup(a)


def compliant_async_with_nested_sync_with():
    # Mixed async/sync cannot be combined into one multi-item with.
    async def inner():
        async with open_async("a.txt") as a:
            with open("b.txt") as b:
                await process(a, b)


def compliant_sync_with_nested_async_with():
    # Mixed sync/async cannot be combined into one multi-item with.
    async def inner():
        with open("a.txt") as a:
            async with open_async("b.txt") as b:
                await process(a, b)


def noncompliant_async_with_nested_sync_chain():
    # Async outer stays; the sole-nested sync chain below still combines.
    async def inner():
        async with open_async("a.txt") as a:
            with open("b.txt") as b:  # Noncompliant
                with open("c.txt") as c:
#           ^^^^@-1
#               ^^^^@-1< {{Nested "with" statement}}
                    await process(a, b, c)


def noncompliant_sync_chain_then_async():
    # Sync chain combines; mixed async innermost stays nested.
    async def inner():
        with open("a.txt") as a:  # Noncompliant
            with open("b.txt") as b:
#       ^^^^@-1
#           ^^^^@-1< {{Nested "with" statement}}
                async with open_async("c.txt") as c:
                    await process(a, b, c)


async def noncompliant_asyncio_timeout():
    import asyncio
    async with open_async("a.txt") as a:  # Noncompliant
        async with asyncio.timeout(1.0):
#         ^^^^@-1
#             ^^^^@-1< {{Nested "with" statement}}
            await process(a)
    async with asyncio.timeout(1.0):  # Noncompliant
        async with open_async("a.txt") as a:
#         ^^^^@-1
#             ^^^^@-1< {{Nested "with" statement}}
            await process(a)


async def noncompliant_asyncio_timeout_at():
    import asyncio
    async with open_async("a.txt") as a:  # Noncompliant
        async with asyncio.timeout_at(1.0):
#         ^^^^@-1
#             ^^^^@-1< {{Nested "with" statement}}
            await process(a)


async def noncompliant_trio_scopes():
    import trio
    async with open_async("a.txt") as a:  # Noncompliant
        async with trio.fail_after(1.0):
#         ^^^^@-1
#             ^^^^@-1< {{Nested "with" statement}}
            await process(a)
    async with open_async("a.txt") as a:  # Noncompliant
        async with trio.fail_at(1.0):
#         ^^^^@-1
#             ^^^^@-1< {{Nested "with" statement}}
            await process(a)
    async with open_async("a.txt") as a:  # Noncompliant
        async with trio.move_on_after(1.0):
#         ^^^^@-1
#             ^^^^@-1< {{Nested "with" statement}}
            await process(a)
    async with open_async("a.txt") as a:  # Noncompliant
        async with trio.move_on_at(1.0):
#         ^^^^@-1
#             ^^^^@-1< {{Nested "with" statement}}
            await process(a)


async def noncompliant_anyio_scopes():
    import anyio
    async with open_async("a.txt") as a:  # Noncompliant
        async with anyio.fail_after(1.0):
#         ^^^^@-1
#             ^^^^@-1< {{Nested "with" statement}}
            await process(a)
    async with open_async("a.txt") as a:  # Noncompliant
        async with anyio.move_on_after(1.0):
#         ^^^^@-1
#             ^^^^@-1< {{Nested "with" statement}}
            await process(a)
    async with open_async("a.txt") as a:  # Noncompliant
        async with anyio.CancelScope():
#         ^^^^@-1
#             ^^^^@-1< {{Nested "with" statement}}
            await process(a)


async def compliant_timeout_wraps_subset_of_body():
    import asyncio
    async with open_async("a.txt") as a:
        await prepare(a)
        async with asyncio.timeout(1.0):
            await process(a)
        await cleanup(a)


def compliant_single_with():
    with open("a.txt") as a:
        process(a)


def noncompliant_with_variable_cm(lock):
    with lock:  # Noncompliant
        with open("b.txt") as b:
#   ^^^^@-1
#       ^^^^@-1< {{Nested "with" statement}}
            process(b)


def noncompliant_multiline_outer_item():
    with open(  # Noncompliant
#   ^^^^
        "a.txt"
    ) as a:
        with open("b.txt") as b:
#       ^^^^< {{Nested "with" statement}}
            process(a, b)


def noncompliant_comment_before_outer():
    # A comment above the outer with does not document the nesting.
    with open("a.txt") as a:  # Noncompliant
        with open("b.txt") as b:
#   ^^^^@-1
#       ^^^^@-1< {{Nested "with" statement}}
            process(a, b)


def compliant_comment_between():
    with open("a.txt") as a:
        # Add a comment explaining why these with statements are not merged. For example:
        # These contexts can't be merged as Python 3.0 has no multi-item with
        with open("b.txt") as b:
            process(a, b)


def noncompliant_comment_breaks_outer_pair_only():
    with open("a.txt") as a:
        # These contexts can't be merged as Python 3.0 has no multi-item with
        with open("b.txt") as b:  # Noncompliant
            with open("c.txt") as c:
#       ^^^^@-1
#           ^^^^@-1< {{Nested "with" statement}}
                process(a, b, c)


def noncompliant_long_combined_line():
    with open("very_long_filename_aaaaaaaaaaaaaaaaaaaaaaaaaaaa.txt") as a:  # Noncompliant
        with open("another_very_long_filename_bbbbbbbbbbbbbbbbbbbb.txt") as b:
#   ^^^^@-1
#       ^^^^@-1< {{Nested "with" statement}}
            process(a, b)


def compliant_with_only_in_try():
    try:
        with open("b.txt") as b:
            process(b)
    except OSError:
        pass


def process(*args):
    pass


def prepare(a):
    pass


def cleanup(a):
    pass


def open_async(path):
    pass
