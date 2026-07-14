import typing
from fastapi import FastAPI, APIRouter, Depends, Security, Path
from typing import Annotated
from pydantic import BaseModel
from dataclasses import dataclass
import typing_extensions

app = FastAPI()
router = APIRouter()

@app.get("/items/{item_id}")
def noncompliant_missing_path_param():  # Noncompliant {{Add path parameter "item_id" to the function signature.}}
    return {"message": "Hello"}

@app.get("/users/{user_id}/items/{item_id}")
def noncompliant_missing_one_of_multiple_params(user_id: int):  # Noncompliant {{Add path parameter "item_id" to the function signature.}}
    return {"user_id": user_id}

@app.get("/items/{item_id}")
def noncompliant_positional_only_param(item_id: int, /):  # Noncompliant {{Path parameter "item_id" should not be positional-only.}}
    return {"item_id": item_id}

@app.get("/things/{thing_id}")
async def noncompliant_async_missing_param(query: str):  # Noncompliant 
    return {"query": query}

@app.get("/users/{user_id}/posts/{post_id}")
def noncompliant_multiple_positional_only(user_id: int, post_id: int, /):  # Noncompliant 2
    return {"user_id": user_id}

@router.put("/items/{item_id}")
def noncompliant_router_missing_param():  # Noncompliant 
    return {"updated": True}

@app.post("/items/{item_id}")
def noncompliant_post_missing():  # Noncompliant 
    pass

@app.put("/items/{item_id}")
def noncompliant_put_missing():  # Noncompliant 
    pass

@app.delete("/items/{item_id}")
def noncompliant_delete_missing():  # Noncompliant 
    pass

@app.patch("/items/{item_id}")
def noncompliant_patch_missing():  # Noncompliant 
    pass

@app.options("/items/{item_id}")
def noncompliant_options_missing():  # Noncompliant 
    pass

@app.head("/items/{item_id}")
def noncompliant_head_missing():  # Noncompliant 
    pass

@app.trace("/items/{item_id}")
def noncompliant_trace_missing():  # Noncompliant 
    pass

@app.get("/items/{item_id:int}")
def noncompliant_with_converter_missing():  # Noncompliant 
    pass

@app.get("/users/{user_id}/items/{item_id}")
def noncompliant_mixed_positional_only_and_missing(user_id: int, /):  # Noncompliant 2
    pass

@app.get("/a/{x}/b/{y}/c/{z}")
def noncompliant_all_params_missing():  # Noncompliant 3
    pass

@app.get(path="/items/{item_id}")
def noncompliant_path_keyword_missing():  # Noncompliant 
    return {}

@app.get("/items/{item_id}")
def compliant_basic(item_id: int):
    return {"item_id": item_id}

@app.get("/users/{user_id}/items/{item_id}")
def compliant_multiple_params(user_id: int, item_id: int):
    return {"user_id": user_id, "item_id": item_id}

@app.get("/users/{user_id}/items/{item_id}")
def compliant_reordered_params(item_id: int, user_id: int):
    return {"user_id": user_id, "item_id": item_id}

@app.get("/things/{thing_id}")
async def compliant_async_with_query(thing_id: int, query: str):
    return {"thing_id": thing_id, "query": query}

@app.get("/items")
def compliant_static_route():
    return {"items": []}

@router.put("/items/{item_id}")
def compliant_router_with_param(item_id: int):
    return {"updated": True}

@app.get("/items/{item_id}")
def compliant_keyword_only(*, item_id: int):
    return {"item_id": item_id}

@app.post("/items/{item_id}")
def compliant_post(item_id: int):
    pass

@app.put("/items/{item_id}")
def compliant_put(item_id: int):
    pass

@app.delete("/items/{item_id}")
def compliant_delete(item_id: int):
    pass

@app.get("/items/{item_id}")
def compliant_with_default(item_id: int = 1):
    return {"item_id": item_id}

@app.get("/items/{item_id:int}")
def compliant_with_converter(item_id: int):
    return {"item_id": item_id}

@app.get("/items/{item_id}")
def no_issue_bailout_with_kwargs(**kwargs):
    return kwargs

@app.get("/items/{item_id}")
def no_issue_bailout_with_args_kwargs(*args, **kwargs):
    return {"args": args, "kwargs": kwargs}

@app.get(status_code=200, path="/items/{item_id}")
def compliant_path_keyword(item_id: int):
    return {"item_id": item_id}

# --- Edge cases ---

def some_other_decorator(path):
    def wrapper(func):
        return func
    return wrapper

@some_other_decorator("/items/{item_id}")
def compliant_not_fastapi_decorator():
    pass

@app.get("/static")
def compliant_no_path_params():
    pass

@app.get("/items")
def compliant_extra_query_params(query: str):
    return {"query": query}

path = "/items/{item_id}"
@app.get(path)
def noncompliant_dynamic_path(): # Noncompliant
    pass

@app.get
def compliant_decorator_without_call():
    pass

@app.get("")
def compliant_empty_path():
    pass


@app.get(True)
def compliant_path_is_not_a_string():
    pass

# --- starlette route syntax ---
# FastAPI builts upon starlette.
# It documents support for "path convert" 
# See https://fastapi.tiangolo.com/tutorial/path-params/?h=convert#path-convertor
# But starlette supports also other kinds of coverters (int, float, uuid, etc.)
# See https://starlette.dev/routing/

@app.get("/items/{item_id:int}")
def noncompliant_path_param_with_int_converter():  # Noncompliant
    pass

@app.get("/items/{item_id:path}")
def noncompliant_path_param_with_path_converter():  # Noncompliant
    pass

@app.get("/items/{item_id : path}")
def compliant_ignore_invalid_starlette_route_0():
    pass

@app.get("/items/{ item_id }")
def compliant_ignore_invalid_starlette_route_1():
    pass

# --- Depends() path parameter delegation ---

def get_item(item_id: int):
    return {"item_id": item_id}

def get_item_positional_only(item_id: int, /):
    return {"item_id": item_id}

@router.get("/items/{item_id}")
def compliant_depends_default_value(item=Depends(get_item)):
    return item

@router.get("/items/{item_id}")
def compliant_depends_positional_only_dependency(item=Depends(get_item_positional_only)):
    # Theoretically it is also valuable to raise on positional-only parameters
    # in dependency callables. But let's not do that for now to limit the scope
    # of our work. And, perhaps not using positional-only parameters can be
    # an entirely different rule, that is orthogonal to path parameters.
    return item

@router.get("/items/{item_id}")
def compliant_depends_dependency_keyword(item=Depends(dependency=get_item)):
    return item

@app.get("/items/{item_id}")
def no_issue_bailout_unresolved_dependency(item=Depends(unknown_dependency)):
    # `unknown_dependency` cannot be resolved. Bail out to avoid FPs.
    pass

@app.get("/users/{user_id}/items/{item_id}")
def noncompliant_positional_only_and_missing_bailout(user_id: int, /, item=Depends(unknown_dependency)):  # Noncompliant {{Path parameter "user_id" should not be positional-only.}}
    # We don't raise for `item_id`. Bail out to avoid FPs.
    # But `user_id` is positional and this shoudl be raised even though we cannot resolve `unknown_dependency`
    pass

def dependency_factory():
    return dep_without_item_id

@app.get("/items/{item_id}")
def no_issue_bailout_dynamic_dependency(item=Depends(dependency_factory())):
    # We don't try to analyze the dynamic behavior of `dependency_factory`.
    # Bail out to avoid FPs.
    pass

def dependency_marker_factory():
    return Depends(get_item)

@app.get("/items/{item_id}")
def no_issue_bailout_dynamic_default_value_dependency(item=dependency_marker_factory()):
    # The default value may still evaluate to a valid FastAPI dependency marker.
    # Bail out rather than report a FP.
    pass

@app.get("/items/{item_id}")
def compliant_depends_annotated(item: Annotated[dict, Depends(get_item)]):
    return item

ItemDependency = Annotated[dict, Depends(get_item)]

@app.get("/items/{item_id}")
def compliant_depends_annotated_type_alias(item: ItemDependency):
    pass

def get_dependency_annotation():
  return Annotated[dict, Depends(dep_without_item_id)]

DynamicDependency = get_dependency_annotation()

@app.get("/items/{item_id}")
def no_issue_bailout_runtime_type_alias(item: DynamicDependency):
    # Type alias is not. Bail out to avoid FPs.
    pass

CyclicDependencyA = CyclicDependencyB
CyclicDependencyB = CyclicDependencyA

@app.get("/items/{item_id}")
def no_issue_bailout_cyclic_type_alias(item: CyclicDependencyA):
    pass

@app.get("/items/{item_id}")
def compliant_depends_annotated_from_typing_extension(item: typing_extensions.Annotated[dict, Depends(get_item)]):
    return item

@app.get("/users/{user_id}/items/{item_id}")
def compliant_depends_partial(user_id: int, item=Depends(get_item)):
    return {"user_id": user_id, "item": item}

async def get_item_async(item_id: int):
    return {"item_id": item_id}

@router.get("/items/{item_id}")
def compliant_depends_async_dependency(item=Depends(get_item_async)):
    return item

def dep_without_item_id(name: str):
    return name

@app.get("/items/{item_id}")
def noncompliant_depends_param_not_in_dep(item=Depends(dep_without_item_id)):  # Noncompliant {{Add path parameter "item_id" to the function signature.}}
    return item

@app.get("/items/{item_id}")
def compliant_depends_annotated_qualified(item: typing.Annotated[dict, Depends(get_item)]):
    return item

@app.get("/items/{item_id}")
def compliant_security_dependency(item=Security(get_item)):
    # Security works in the same way of Depends.
    # See https://fastapi.tiangolo.com/reference/dependencies/?h=depends+securi#fastapi.Security
    pass

# --- Multiple dependencies in Annotated ---
# Note: FastAPI only documents Annotated with a single Depends(). It is thus an invalid use of FastAPI.
# ruff bails out in this situation. 
# We currently just pick the first Depends.

@app.get("/items/{item_id}")
def compliant_multiple_depends_metadata_union(item: Annotated[dict, Depends(dep_without_item_id), Depends(get_item)]):
    pass


@app.get("/items/{item_id}")
def noncompliant_multiple_depends_metadata_no_match(item: Annotated[dict, Depends(dep_without_item_id), Depends(dep_without_item_id)]):  # Noncompliant
    pass


# --- Nested/recursive Depends() ---

def get_item_wrapper(item=Depends(get_item)):
    return item

@router.get("/items/{item_id}")
def compliant_nested_depends(item=Depends(get_item_wrapper)):
    return item

def get_item_annotated_wrapper(item: Annotated[dict, Depends(get_item)]):
    return item

@app.get("/items/{item_id}")
def compliant_nested_depends_annotated(item=Depends(get_item_annotated_wrapper)):
    return item

def dep_level2_no_id(name: str):
    return name

def dep_level1_no_id(x=Depends(dep_level2_no_id)):
    return x

@app.get("/items/{item_id}")
def noncompliant_nested_depends_not_covering(item=Depends(dep_level1_no_id)):  # Noncompliant {{Add path parameter "item_id" to the function signature.}}
    return item

def wrapper(dep=Depends(unknown_dependency)):
    return dep


@app.get("/items/{item_id}")
def no_issue_bailout_unknown_nested_dependency(item=Depends(wrapper)):
    # wrapper depends on `unknown_dependency`, which we cannot resolve.
    # Bail out to avoid FPs.
    pass

# --- Circular dependency detection (cycle in Depends chain) ---
# dep_b_circular references dep_a_circular (defined below)
def dep_b_circular(dep=Depends(dep_a_circular)):
    return dep

def dep_a_circular(item_id: int, dep=Depends(dep_b_circular)):
    return item_id

@app.get("/items/{item_id}")
def compliant_circular_depends(dep=Depends(dep_a_circular)):
    return dep

# --- QualifiedExpression as Depends argument ---
class ItemDepsHolder:
    @staticmethod
    def get_item(item_id: int):
        return item_id

@app.get("/items/{item_id}")
def compliant_qualified_expr_depends(dep=Depends(ItemDepsHolder.get_item)):
    return dep

# --- Non-Name/non-QualifiedExpression as Depends argument ---
@app.get("/items/{item_id}")
def noncompliant_lambda_depends(dep=Depends(lambda: None)):  # Noncompliant {{Add path parameter "item_id" to the function signature.}}
    return dep


@app.get("/items/{item_id}")
def compliant_lambda_depends_with_path_param(dep=Depends(lambda item_id: item_id)):
    return dep

@app.get("/items/{item_id}")
def noncompliant_depends_without_target(dep=Depends()):  # Noncompliant
    pass


# --- Dependencies specified in path operation decorators ---
# See https://fastapi.tiangolo.com/tutorial/dependencies/dependencies-in-path-operation-decorators/

@app.get("/items/{item_id}", dependencies=[Depends(get_item)])
def compliant_route_dependency_covers_path_param():
    pass


@app.get("/items/{item_id}", dependencies=[Depends(dep_without_item_id)])
def noncompliant_route_dependency_does_not_cover_path_param():  # Noncompliant
    pass

dependencies = [Depends(unknown_dependency)]

@app.get("/items/{item_id}", dependencies=dependencies)
def no_issue_bailout_route_dependencies_alias_with_unresolved_dependency():
    pass


@app.get("/items/{item_id}", dependencies=unknown_dependencies)
def no_issue_bailout_unresolved_route_dependencies_argument():
    pass


@app.get("/items/{item_id}", dependencies=get_route_dependencies())
def no_issue_bailout_dynamic_route_dependencies_argument():
    pass


@app.get("/items/{item_id}", dependencies=[dependency_marker_factory()])
def no_issue_bailout_dynamic_route_dependency_list_entry():
    pass


@app.get("/items/{item_id}", dependencies=[Depends(unknown_dependency)])
def no_issue_bailout_dependency_unresolved():
    pass


@app.get("/items/{item_id}", dependencies=[unknown_dependency])
def no_issue_bailout_unresolved_direct_route_dependency():
    pass


@app.get("/items/{item_id}", dependencies=[*dependencies])
def no_issue_bailout_on_dependencies_sequence_unpacking():
    pass


@app.get("/items/{item_id}", dependencies=[object()])
def no_issue_bailout_invalid_route_dependency_list_entry_object():
    # Not a dependency entry we support. And in this case it's clearly wrong
    # FastAPI usage. We bail out.
    pass

@app.get("/items/{item_id}", dependencies=[42])
def no_issue_bailout_invalid_route_dependency_list_entry_integer():
    # Not a dependency entry we support. And in this case it's clearly  wrong
    # FastAPI usage. We bail out.
    pass


# --- Dependencies specified in FastAPI/APIRouter constructors ---
# See https://fastapi.tiangolo.com/tutorial/dependencies/global-dependencies/

router_with_dependency = APIRouter(dependencies=[Depends(get_item)])

@router_with_dependency.get("/items/{item_id}")
def compliant_router_dependency_covers_path_param():
    pass


router_without_path_dependency = APIRouter(dependencies=[Depends(dep_without_item_id)])

@router_without_path_dependency.get("/items/{item_id}")
def noncompliant_router_dependency_does_not_cover_path_param():  # Noncompliant
    pass


app_with_dependency = FastAPI(dependencies=[Depends(get_item)])

@app_with_dependency.get("/items/{item_id}")
def compliant_app_dependency_covers_path_param():
    pass


FastAPIAlias = FastAPI
app_with_aliased_fastapi_dependency = FastAPIAlias(dependencies=[Depends(get_item)])

@app_with_aliased_fastapi_dependency.get("/items/{item_id}")
def compliant_aliased_app_dependency_covers_path_param():
    pass


# --- Path aliases ---
# See docs on `alias` param at https://fastapi.tiangolo.com/reference/parameters/?h=Path+reference#fastapi.Path

@app.get("/items/{item-id}")
def compliant_literal_path_alias(item_id: Annotated[int, Path(alias="item-id")]):
    pass


@app.get("/items/{item_id}")
def noncompliant_path_alias_replaces_parameter_name(item_id: Annotated[int, Path(alias="other_id")]):  # Noncompliant {{Add path parameter "item_id" to the function signature.}}
    pass


@app.get("/items/{item-id}")
def compliant_default_value_path_alias(item_id: int = Path(alias="item-id")):
    pass


@app.get("/items/{item_id}")
def compliant_default_value_path_without_alias(item_id: int = Path()):
    pass


@app.get("/items/{item_id}")
def noncompliant_default_value_path_without_alias(other_id: int = Path()):  # Noncompliant
    pass

ITEM_ID_ALIAS = "item-id"

@app.get("/items/{item-id}")
def compliant_constant_path_alias(item_id: Annotated[int, Path(alias=ITEM_ID_ALIAS)]):
    pass


@app.get("/items/{item-id}")
def no_issue_bailout_unknown_path_alias(item_id: Annotated[int, Path(alias=get_alias_unresolved())]):
    pass


@app.get("/items/{item-id}")
def no_issue_bailout_unknown_default_value_path_alias(item_id: int = Path(alias=get_alias_unresolved())):
    pass


def get_item_with_alias(item_id: Annotated[int, Path(alias="item-id")]):
    return item_id


@app.get("/items/{item-id}")
def compliant_dependency_has_path_alias(item=Depends(get_item_with_alias)):
    pass


def get_item_with_default_alias(item_id: int = Path(alias="item-id")):
    return item_id


@app.get("/items/{item-id}")
def compliant_dependency_default_value_path_alias(item=Depends(get_item_with_default_alias)):
    pass

# --- Class dependencies ---
# --- Callable dependency instances ---
# See https://fastapi.tiangolo.com/tutorial/dependencies/classes-as-dependencies/

class ItemQuery:
    def __init__(self, item_id: int):
        pass


@app.get("/items/{item_id}")
def compliant_class_dependency(query: Annotated[ItemQuery, Depends(ItemQuery)]):
    pass


@app.get("/items/{item_id}")
def compliant_class_dependency_default_value(query: ItemQuery = Depends(ItemQuery)):
    pass


class EmptyItemQuery:
    pass


@app.get("/items/{item_id}")
def noncompliant_class_dependency_without_path_param_in_init(query: Annotated[EmptyItemQuery, Depends(EmptyItemQuery)]):  # Noncompliant
    pass


class PydanticItemQueryWithoutPathParam(BaseModel):
    name: str


@app.get("/items/{item_id}")
def no_issue_bailout_pydantic_dependency(query: Annotated[PydanticItemQueryWithoutPathParam, Depends(PydanticItemQueryWithoutPathParam)]):
    pass


@dataclass
class DataclassItemQueryWithoutPathParam:
    name: str


@app.get("/items/{item_id}")
def no_issue_bailout_dataclass_dependency(query: Annotated[DataclassItemQueryWithoutPathParam, Depends(DataclassItemQueryWithoutPathParam)]):
    pass


def generated_constructor(cls):
    return cls


@generated_constructor
class DecoratedItemQuery:
    item_id: int


@app.get("/items/{item_id}")
def no_issue_bailout_decorated_class_without_explicit_init(query: Annotated[DecoratedItemQuery, Depends(DecoratedItemQuery)]):
    pass


class ItemChecker:
    def __call__(self, item_id: int):
        return item_id


checker = ItemChecker()


@app.get("/items/{item_id}")
def compliant_callable_instance_dependency(item=Depends(checker)):
    pass


@app.get("/items/{item_id}")
def compliant_callable_constructor_instance_dependency(item=Depends(ItemChecker())):
    pass


class DependencyWithoutCall:
    pass


dependency_without_call = DependencyWithoutCall()


@app.get("/items/{item_id}")
def noncompliant_dependency_instance_without_call(item=Depends(dependency_without_call)):  # Noncompliant {{Add path parameter "item_id" to the function signature.}}
    pass


class InheritedCallableDependency(ItemChecker):
    pass


inherited_callable_dependency = InheritedCallableDependency()


@app.get("/items/{other_id}")
def no_issue_bailout_inherited_callable_dependency(item=Depends(inherited_callable_dependency)):
    pass


class ItemCheckerClassReference:
    def __call__(self, item_id: int):
        return item_id


@app.get("/items/{item_id}")
def noncompliant_class_dependency_without_init_path_param(item=Depends(ItemCheckerClassReference)):  # Noncompliant {{Add path parameter "item_id" to the function signature.}}
    # FastAPI inspects __init__ when Depends receives a class, not the __call__ method used by instances.
    pass


# --- Shortcut when class callable is used as dependency ---
# See https://fastapi.tiangolo.com/tutorial/dependencies/classes-as-dependencies/?h=shortcut#type-annotation-vs-depends

@app.get("/items/{item_id}")
def compliant_depends_shortcut_annotated(query: Annotated[ItemQuery, Depends()]):
    pass


@app.get("/items/{item_id}")
def compliant_depends_shortcut_default(query: ItemQuery = Depends()):
    pass


# --- Known FPs: include_router(..., dependencies=...) is currently out of scope ---

def require_item_access(item_id: int):
    return item_id

router_included_with_dependency = APIRouter()
@router_included_with_dependency.get("/items/{item_id}")
def known_fp_app_include_router_dependency():  # Noncompliant
    # Known FP: `app.include_router(..., dependencies=[Depends(require_item_access)])`
    # covers `item_id`, but include_router dependencies are currently not modeled by this rule.
    pass

app_with_include_router_dependency = FastAPI()
app_with_include_router_dependency.include_router(router_included_with_dependency, dependencies=[Depends(require_item_access)])

parent_router_with_include_router_dependency = APIRouter()
child_router_included_with_dependency = APIRouter()
parent_router_with_include_router_dependency.include_router(child_router_included_with_dependency, dependencies=[Depends(require_item_access)])

@child_router_included_with_dependency.get("/items/{item_id}/issues")
def known_fp_router_include_router_dependency():  # Noncompliant
    # Known FP: `parent_router.include_router(..., dependencies=[Depends(require_item_access)])`
    # covers `item_id`, but nested include_router dependencies are currently not modeled by this rule.
    pass
