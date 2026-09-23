from nonContextManagerImported import create_session, create_session_async, raw_generator


def use_imported_contextmanager():
    # Cross-module: the @contextmanager decorator must be detected on the imported function.
    with create_session() as session:  # Compliant
        print(session)
    with raw_generator():  # Noncompliant
        pass


async def use_imported_async_contextmanager():
    async with create_session_async() as session:  # Compliant
        print(session)
