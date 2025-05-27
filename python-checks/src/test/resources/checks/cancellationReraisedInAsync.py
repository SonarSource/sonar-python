import asyncio
import trio
import anyio
from contextlib import suppress

async def asyncio_compliant_examples():
    # Compliant - re-raising CancelledError
    async def func1():
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            # Clean up code here
            raise

    # Compliant - specific handler before general handler
    async def func2():
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            raise
        except Exception:
            return None

    # Compliant - different pattern but still re-raising
    async def func3():
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError as e:
            # Clean up code here
            raise e

    # Compliant - handling in finally block instead
    async def func4():
        try:
            await asyncio.sleep(1)
        finally:
            # Clean up code here
            pass

    # Compliant - not catching exceptions at all
    async def func5():
        await asyncio.sleep(1)
        return True

    # Compliant - catching specific non-cancellation exceptions
    async def func6():
        try:
            await asyncio.sleep(1)
        except ValueError:
            return None


async def asyncio_noncompliant_examples():
    # catching CancelledError without re-raising
    async def func1():
        # ^[sc=5;ec=9]> {{This function is async.}}
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:  # Noncompliant {{Ensure that the asyncio.CancelledError exception is re-raised after your cleanup code.}}
            #  ^^^^^^^^^^^^^^^^^^^^^^
            pass

    # catching and returning
    async def func2():
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError: # Noncompliant
            return None

    # catching and logging only
    async def func3():
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError as e: # Noncompliant
            print(f"Cancelled: {e}")

    # catching and raising different exception
    async def func4():
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError: # Noncompliant
            raise RuntimeError("Operation cancelled")

    # catching in generic Exception handler
    async def func5():
        try:
            await asyncio.sleep(1)
        except Exception: # Compliant
            return None

    async def func6():
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError: # Noncompliant
            return
            raise RuntimeError("Operation cancelled")

async def trio_compliant_examples():
    # Compliant - re-raising Cancelled
    async def func1():
        try:
            await trio.sleep(1)
        except trio.Cancelled:
            # Clean up code here
            raise

    # Compliant - specific handler before general handler
    async def func2():
        try:
            await trio.sleep(1)
        except trio.Cancelled:
            raise
        except Exception:
            return None

    # Compliant - different pattern but still re-raising
    async def func3():
        try:
            await trio.sleep(1)
        except trio.Cancelled as e:
            # Clean up code here
            raise e


async def trio_noncompliant_examples():
    # catching Cancelled without re-raising
    async def func1():
        try:
            await trio.sleep(1)
        except trio.Cancelled: # Noncompliant
            # Clean up code here
            pass

    # catching and returning
    async def func2():
        try:
            await trio.sleep(1)
        except trio.Cancelled: # Noncompliant
            return None

    # catching in generic Exception handler
    async def func3():
        try:
            await trio.sleep(1)
        except Exception: # Compliant
            return None

async def anyio_compliant_examples():
    # Compliant - re-raising Cancelled
    async def func1():
        try:
            await anyio.sleep(1)
        except anyio.get_cancelled_exc_class():
            # Clean up code here
            raise

    # Compliant - specific handler before general handler
    async def func2():
        try:
            await anyio.sleep(1)
        except anyio.get_cancelled_exc_class():
            raise
        except Exception:
            return None


async def anyio_noncompliant_examples():
    # catching Cancelled without re-raising
    async def func1():
        try:
            await anyio.sleep(1)
        except anyio.get_cancelled_exc_class():  # Noncompliant {{Ensure that the cancellation exception is re-raised after your cleanup code.}}
            # Clean up code here
            pass

    # catching and returning
    async def func2():
        try:
            await anyio.sleep(1)
        except anyio.get_cancelled_exc_class(): # Noncompliant
            return None


async def edge_case_tests():
    # Compliant - using contextlib.suppress for other exceptions but not cancellation
    async def func1():
        with suppress(ValueError, TypeError):
            await asyncio.sleep(1)

    # Compliant - handling cancellation in a with block that re-raises
    async def func2():
        class CancellationHandler:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                if isinstance(exc_val, asyncio.CancelledError):
                    # Cleanup
                    return False  # Re-raises the exception
                return False

        async with CancellationHandler():
            await asyncio.sleep(1)

    # using contextlib.suppress on cancellation exceptions
    async def func3():
        with suppress(asyncio.CancelledError):
            await asyncio.sleep(1)

    # using a with block that swallows cancellation
    async def func4():
        class BadCancellationHandler:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                if isinstance(exc_val, asyncio.CancelledError):
                    # Cleanup
                    return True  # Suppresses the exception
                return False

        async with BadCancellationHandler():  # Too complex
            await asyncio.sleep(1)
    # The except is a tuple
    async def func5():
        try:
            await asyncio.sleep(1)
        except (asyncio.CancelledError, ValueError): # FN
            pass


async def coverage():
    async def blank_except():
        try:
            await asyncio.sleep(1)
        except:
            pass

    def not_async():
        try:
            asyncio.sleep(1)
        except:
            pass

    async def unrelated_exception_returned():
        try:
            await asyncio.sleep(1)
        except get_some_exception():
            pass

    async def reraise_not_a_name():
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError as e:  # Noncompliant
            raise some(e)
