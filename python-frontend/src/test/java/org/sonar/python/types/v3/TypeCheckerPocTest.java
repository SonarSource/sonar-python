package org.sonar.python.types.v3;

import org.junit.jupiter.api.Test;
import org.sonar.python.semantic.v2.ClassTypeBuilder;
import org.sonar.python.semantic.v2.ObjectTypeBuilder;

import static org.assertj.core.api.Assertions.assertThat;

class TypeCheckerPocTest {

  @Test
  void simpleClassType() {
    var typeChecker = new TypeCheckerPoc.UnspecializedTypeCheckerBuilder()
      .with(TypeCheckerPoc.isClass())
      .build();

    var classType = new ClassTypeBuilder().withName("MyClass").build();
    var objectType = new ObjectTypeBuilder().build();

    assertThat(typeChecker.isTrue(classType)).isTrue();
    assertThat(typeChecker.isTrue(objectType)).isFalse();
  }
}
