from collections.abc import Mapping
from typing import Any


def mapping_argument(mapping: Mapping[str, Any]) -> dict[str, Any]:
  return dict(mapping)


mapping_argument({"a": 1})
annotated_mapping: Mapping[str, Any] = {"a": 1}
mapping_argument(annotated_mapping)
