async def async_lambda_handler(event, context):  # Noncompliant {{Remove the `async` keyword from this AWS Lambda handler definition.}}
#^[sc=1;ec=5]
    result = some_logic()
    return {"status": result}

def some_logic():
    return "some result"


def lambda_handler(event, context):  
    import asyncio
    
    result = asyncio.run(not_a_lambda_handler())
    return {"status": result}

async def not_a_lambda_entry_point():  # Compliant - not a lambda handler
    pass
