
def asyncio_section():
    import asyncio

    SHARED_CONDITION_ASYNCIO = False
    condition_met_asyncio = False
    x_asyncio = 0
    my_flag_asyncio = False
    dict_state_asyncio = {"status": "pending"}

    async def asyncio_noncompliant_simple_not_operator():
        nonlocal SHARED_CONDITION_ASYNCIO
        while not SHARED_CONDITION_ASYNCIO: # Noncompliant
            await asyncio.sleep(0.01)

    async def asyncio_noncompliant_simple_equality_check_false():
        nonlocal condition_met_asyncio
        while condition_met_asyncio == False: # Noncompliant
            await asyncio.sleep(0.01)

    async def asyncio_noncompliant_simple_equality_check_is_false():
        nonlocal condition_met_asyncio
        while condition_met_asyncio is False: # Noncompliant
            await asyncio.sleep(0.01)

    async def asyncio_noncompliant_direct_variable_true_implicit_poll():
        nonlocal my_flag_asyncio
        my_flag_asyncio = True
        while my_flag_asyncio: # Noncompliant
            await asyncio.sleep(0.01)
            my_flag_asyncio = False

    async def asyncio_noncompliant_loop_variable_comparison_potentially_external():
        nonlocal x_asyncio
        while x_asyncio < 10: # Noncompliant
            await asyncio.sleep(0.01)

    async def asyncio_noncompliant_while_true_with_internal_check():
        nonlocal SHARED_CONDITION_ASYNCIO
        while True: # FN because of the if
            if SHARED_CONDITION_ASYNCIO:
                break
            await asyncio.sleep(0.01)

    async def asyncio_noncompliant_imported_sleep():
        from asyncio import sleep
        nonlocal my_flag_asyncio
        while not my_flag_asyncio: # Noncompliant
            await sleep(0.01)

    async def asyncio_noncompliant_aliased_imported_sleep():
        from asyncio import sleep as async_sleep
        nonlocal my_flag_asyncio
        while not my_flag_asyncio: # Noncompliant
            await async_sleep(0.01)

    async def asyncio_noncompliant_aliased_module_sleep():
        import asyncio as my_asyncio
        global my_flag_asyncio
        while not my_flag_asyncio: # Noncompliant
            await my_asyncio.sleep(0.01)

    async def asyncio_noncompliant_dict_value_check():
        global dict_state_asyncio
        while dict_state_asyncio["status"] == "pending": # Noncompliant
            await asyncio.sleep(0.01)

    # Compliant Examples
    async def asyncio_compliant_event():
        event = asyncio.Event()
        await event.wait()

    async def asyncio_compliant_sleep_outside_while():
        await asyncio.sleep(0.01)

    async def asyncio_compliant_sleep_in_for_loop():
        for _i in range(3):
            await asyncio.sleep(0.01)

    async def asyncio_compliant_while_loop_without_async_sleep():
        nonlocal SHARED_CONDITION_ASYNCIO
        while not SHARED_CONDITION_ASYNCIO:
            if True:
                SHARED_CONDITION_ASYNCIO = True

    async def asyncio_compliant_while_true_no_sleep_just_break():
        nonlocal SHARED_CONDITION_ASYNCIO
        while True:
            if SHARED_CONDITION_ASYNCIO:
                break

    def asyncio_compliant_sync_function_with_time_sleep():
        import time
        sync_condition = False
        while not sync_condition:
            time.sleep(0.01)
            sync_condition = True

    pseudo_global = False

    async def asyncio_compliant_while_true_with_internal_check():
        while not pseudo_global:  # Noncompliant
            await asyncio.sleep(0.01)


# --- Trio Cases ---
def trio_section():
    import trio

    SHARED_CONDITION_TRIO = False
    my_flag_trio = False

    async def trio_noncompliant_simple_not_operator():
        nonlocal SHARED_CONDITION_TRIO
        while not SHARED_CONDITION_TRIO: # Noncompliant
            await trio.sleep(0.01)

    async def trio_noncompliant_while_true_with_internal_check():
        nonlocal SHARED_CONDITION_TRIO
        while True: # FN because of the if
            if SHARED_CONDITION_TRIO:
                break
            await trio.sleep(0.01)

    async def trio_noncompliant_imported_sleep():
        from trio import sleep
        nonlocal my_flag_trio
        while not my_flag_trio: # Noncompliant
            await sleep(0.01)

    async def trio_noncompliant_aliased_imported_sleep():
        from trio import sleep as trio_s
        nonlocal my_flag_trio
        while not my_flag_trio: # Noncompliant
            await trio_s(0.01)

    async def trio_noncompliant_aliased_module_sleep():
        import trio as my_trio
        nonlocal my_flag_trio
        while not my_flag_trio: # Noncompliant
            await my_trio.sleep(0.01)

    # Compliant Examples
    async def trio_compliant_event():
        event = trio.Event()
        await event.wait()

    async def trio_compliant_sleep_outside_while():
        await trio.sleep(0.01)

    async def trio_compliant_sleep_in_for_loop():
        for _i in range(3):
            await trio.sleep(0.01)


# --- AnyIO Cases ---
def anyio_section():
    import anyio

    SHARED_CONDITION_ANYIO = False
    my_flag_anyio = False

    async def anyio_noncompliant_simple_not_operator():
        nonlocal SHARED_CONDITION_ANYIO
        while not SHARED_CONDITION_ANYIO: # Noncompliant
            await anyio.sleep(0.01)

    async def anyio_noncompliant_while_true_with_internal_check():
        nonlocal SHARED_CONDITION_ANYIO
        while True: # FN because of the if
            if SHARED_CONDITION_ANYIO:
                break
            await anyio.sleep(0.01)

    async def anyio_noncompliant_imported_sleep():
        from anyio import sleep
        nonlocal my_flag_anyio
        while not my_flag_anyio: # Noncompliant
            await sleep(0.01)

    async def anyio_noncompliant_aliased_imported_sleep():
        from anyio import sleep as anyio_s
        nonlocal my_flag_anyio
        while not my_flag_anyio: # Noncompliant
            await anyio_s(0.01)

    async def anyio_noncompliant_aliased_module_sleep():
        import anyio as my_anyio
        nonlocal my_flag_anyio
        while not my_flag_anyio: # Noncompliant
            await my_anyio.sleep(0.01)

    # Compliant Examples
    async def anyio_compliant_event():
        event = anyio.Event()
        await event.wait()

    async def anyio_compliant_sleep_outside_while():
        await anyio.sleep(0.01)

    async def anyio_compliant_sleep_in_for_loop():
        for _i in range(3):
            await anyio.sleep(0.01)


def edge_cases_section():
    import asyncio

    async def _while_loop_internal_counter():
        internal_counter = 0
        while internal_counter < 3:
            await asyncio.sleep(0.01)
            internal_counter += 1

    async def while_true_periodic_task_no_external_condition():
        while True:
            await asyncio.sleep(1)


    is_ready_for_fn_test_1 = False
    async def check_ready_with_sleep_inside():
        nonlocal is_ready_for_fn_test_1
        import asyncio
        if not is_ready_for_fn_test_1:
            await asyncio.sleep(0.001)
            return False
        return True

    async def sleep_hidden_in_awaited_function_in_condition():
        while not await check_ready_with_sleep_inside(): # Too hard to detect
            pass

    is_ready_for_fn_test_2 = False
    async def perform_check_and_maybe_sleep_in_body():
        import asyncio
        if not is_ready_for_fn_test_2:
            await asyncio.sleep(0.001)
            return False
        return True

    is_ready_flag_for_internal_poll_fn = False
    async def operation_that_polls_internally_with_sleep_fn():
        nonlocal is_ready_flag_for_internal_poll_fn
        import asyncio
        while not is_ready_flag_for_internal_poll_fn: # Noncompliant
            await asyncio.sleep(0.001)
        is_ready_flag_for_internal_poll_fn = False

def other_compliant_sleep_uses_section():
    import asyncio

    async def compliant_fixed_delay_task():
        await asyncio.sleep(0.1)

    async def compliant_rate_limiting_in_while_true():
        processed_items = 0
        while processed_items < 3:
            processed_items +=1
            await asyncio.sleep(0.05)

    async def compliant_async_for_loop_with_sleep():
        async def dummy_async_generator():
            for i in range(3):
                yield i
                await asyncio.sleep(0.01)

        async for _item in dummy_async_generator():
            pass

    async def compliant_try_except_retry_with_fixed_attempts_and_sleep():
        attempts = 0
        max_attempts = 3
        success = False
        while attempts < max_attempts and not success:
            try:
                if attempts == 1:
                    success = True
                else:
                    raise ConnectionError("Failed")
            except ConnectionError:
                attempts += 1
                if attempts < max_attempts:
                    await asyncio.sleep(0.1)
            else:
                break

GLOB = False
async def smth_async(): ...
async def coverage_await_not_call_expr():
    import asyncio
    while not GLOB:  # Noncompliant
        something = smth_async()
        await something
        await asyncio.sleep(0.01)
