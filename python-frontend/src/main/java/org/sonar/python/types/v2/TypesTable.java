package org.sonar.python.types.v2;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.python.types.pytype.PyTypeTable;
import org.sonar.python.types.v2.converter.PyTypeToPythonTypeConverter;

public class TypesTable {

  private final Map<String, PythonType> declaredTypesTable;
  private final PyTypeTable pyTypeTable;
  private final PyTypeToPythonTypeConverter converter;

  public TypesTable(PyTypeTable pyTypeTable, PyTypeToPythonTypeConverter converter) {
    this.pyTypeTable = pyTypeTable;
    this.converter = converter;
    declaredTypesTable = new HashMap<>();
  }

  public PythonType getTypeForName(String fileName, Name name) {
    return pyTypeTable.getTypeFor(fileName, name)
      .map(pyTypeInfo -> {
        var type = declaredTypesTable.computeIfAbsent(pyTypeInfo.baseType().toString(), (typeName) -> converter.convert(pyTypeInfo.baseType()));
        return (PythonType) new ObjectType(type, List.of());
      }).orElse(new UnionType(List.of()));

  }

}
