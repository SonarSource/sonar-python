package org.sonar.python.types.pytype;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import org.sonar.plugins.python.api.tree.Name;

public class PyTypeTable {

  private final Map<String, List<PyTypeInfo>> files;
  private final Map<TypePositionKey, PyTypeInfo> typesByPosition;
  private final HashMap<TypePositionKey, List<PyTypeInfo>> multipleTypesByPosition;

  public PyTypeTable(Map<String, List<PyTypeInfo>> files) {
    this.files = files;

    typesByPosition = files.entrySet()
      .stream()
      .flatMap(entry -> {
        var file = entry.getKey();
        return entry.getValue()
          .stream()
          .map(typeInfo -> Map.entry(
            new TypePositionKey(file, typeInfo.startLine(), typeInfo.startCol(), typeInfo.text()),
            typeInfo));
      })
      .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue, (v1, v2) -> v1));

    multipleTypesByPosition = files
      .entrySet()
      .stream()
      .flatMap(entry -> {
        var file = entry.getKey();
        return entry.getValue()
          .stream()
          .map(typeInfo -> Map.entry(
            new TypePositionKey(file, typeInfo.startLine(), typeInfo.startCol(), typeInfo.text()),
            typeInfo));
      })
      .collect(Collectors.groupingBy(Map.Entry::getKey, HashMap::new, Collectors.mapping(Map.Entry::getValue, Collectors.toList())));
  }

  public Optional<PyTypeInfo> getTypeFor(String fileName, Name name) {
    var token = name.firstToken();
    return getTypeFor(fileName, token.line(), token.column(), name.name(), "Variable");
  }

  public Optional<PyTypeInfo> getTypeFor(String fileName, int line, int column, String name, String kind) {
    TypePositionKey typePositionKey = new TypePositionKey(fileName, line, column, name);
    return Optional.ofNullable(multipleTypesByPosition.get(typePositionKey))
      .filter(Predicate.not(List::isEmpty))
      .map(types -> types.stream()
        .filter(t -> kind.equals(t.syntaxRole()))
        .findFirst()
        .orElse(types.get(0)));
  }

  private record TypePositionKey(String fileName, int line, int column, String name) {
  }


}
