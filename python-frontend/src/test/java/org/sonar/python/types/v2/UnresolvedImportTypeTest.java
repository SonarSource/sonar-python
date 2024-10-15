package org.sonar.python.types.v2;

import java.util.List;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.v2.ProjectLevelTypeTable;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.semantic.v2.TypeInferenceV2Test.inferTypes;


class UnresolvedImportTypeTest {
  @Test
  @Disabled("unknown.submodule is not correctly resolved")
  void imported_unknown_submodule() {
    FileInput fileInput = inferTypesWithNoResolution("""
      import unknown.submodule
      unknown.submodule
      """);
    var unknownSubmoduleType = ((UnresolvedImportType) ((ExpressionStatement) fileInput.statements().statements().get(1)).expressions().get(0).typeV2());
    assertThat(unknownSubmoduleType.importPath()).isEqualTo("unknown.submodule");
  }


  @Test
  void imported_unknown() {
    FileInput fileInput = inferTypesWithNoResolution("""
      import unknown
      unknown
      """);
    var unknownType = ((UnresolvedImportType) ((ExpressionStatement) fileInput.statements().statements().get(1)).expressions().get(0).typeV2());
    assertThat(unknownType.importPath()).isEqualTo("unknown");
  }

  @Test
  void imported_unknown2() {
    FileInput fileInput = inferTypesWithNoResolution("""
      import xml as a
      a
      """);
    var etreeType = ((UnresolvedImportType) ((ExpressionStatement) fileInput.statements().statements().get(1)).expressions().get(0).typeV2());
    assertThat(etreeType.importPath()).isEqualTo("xml");
  }

  @Test
  void imported_flow_union() {
    FileInput fileInput = inferTypesWithNoResolution("""
      from something import acos, atan
      if cond:
        x = acos
      else:
        x = atan
      x
      acos
      atan
      """);
    var xType = ((UnionType) ((ExpressionStatement) fileInput.statements().statements().get(2)).expressions().get(0).typeV2());
    var acosType = ((UnresolvedImportType) ((ExpressionStatement) fileInput.statements().statements().get(3)).expressions().get(0).typeV2());
    var atanType = ((UnresolvedImportType) ((ExpressionStatement) fileInput.statements().statements().get(4)).expressions().get(0).typeV2());

    assertThat(xType.candidates()).extracting(PythonType::unwrappedType).containsExactlyInAnyOrder(acosType, atanType);
    assertThat(acosType.importPath()).isEqualTo("something.acos");
    assertThat(atanType.importPath()).isEqualTo("something.atan");
  }

  @Test
  void import_from_as() {
    FileInput fileInput = inferTypesWithNoResolution("""
      from something import a_func as f
      f
      """);
    var fType = ((UnresolvedImportType) ((ExpressionStatement) fileInput.statements().statements().get(1)).expressions().get(0).typeV2());
    assertThat(fType).isInstanceOf(UnknownType.class);
  }

  @Test
  void imported_call() {
    FileInput fileInput = inferTypesWithNoResolution("""
      from something import a_func
      x = a_func(1)
      x
      """);
    var xType = ((ExpressionStatement) fileInput.statements().statements().get(2)).expressions().get(0).typeV2();
    assertThat(xType).isInstanceOf(UnknownType.class);
  }

  private static FileInput inferTypesWithNoResolution(String lines) {
    var typeTable = new TestProjectLevelTypeTable(ProjectLevelSymbolTable.empty());
    return inferTypes(lines, typeTable);
  }

  private static class TestProjectLevelTypeTable extends ProjectLevelTypeTable {

    public TestProjectLevelTypeTable(ProjectLevelSymbolTable projectLevelSymbolTable) {
      super(projectLevelSymbolTable);
    }

    @Override
    public PythonType getType(List<String> typeFqnParts) {
      return PythonType.UNKNOWN;
    }
  }
}
