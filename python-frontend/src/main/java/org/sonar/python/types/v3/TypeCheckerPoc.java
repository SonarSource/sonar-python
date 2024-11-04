package org.sonar.python.types.v3;

import java.util.List;
import java.util.function.Predicate;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.TypeSource;

public class TypeCheckerPoc {
  void test() {
    typeChecker()
      .with(isInstanceOf("AClass"))
      .with(typeSource(TypeSource.EXACT));
  }

  // METHODS

  static InnerPredicateBuilder<UnspecializedTypeCheckerBuilder, ObjectTypeBuilder> isInstanceOf(String type) {
    return null;
  }

  static InnerPredicateBuilder<ObjectTypeBuilder, ObjectTypeBuilder> typeSource(TypeSource typeSource) {
    return builder -> {
      return null;
    };
  }

  static InnerPredicateBuilder<UnspecializedTypeCheckerBuilder, ClassTypeBuilder> inheritsFrom(String type) {
    return builder -> {
      return null;
    };
  }

  static UnspecializedTypeCheckerBuilder typeChecker() {
    return null;
  }

  // INTERFACES
  interface TypeCheckerBuilder<T extends TypeCheckerBuilder<T>> {
    <O extends TypeCheckerBuilder<O>> O with(InnerPredicateBuilder<T, O> predicate);
  }
  interface InnerPredicateBuilder<I extends TypeCheckerBuilder<? extends I>, O extends TypeCheckerBuilder<? extends  O>> {
    O construct(I input);

    default InnerPredicateBuilder<I, O> anyMatch() {
      return this;
    }
  }

  // CLASSES

  abstract class AbstractTypeCheckerBuilder<T extends AbstractTypeCheckerBuilder<T>> implements TypeCheckerBuilder<T> {
    private List<Predicate<PythonType>> predicates;

    public void addPredicate(Predicate<PythonType> predicate) {
      predicates.add(predicate);
    }

    @Override
    public <O extends TypeCheckerBuilder<O>> O with(InnerPredicateBuilder<T, O> predicate) {
      return null;
    }
  }

  class UnspecializedTypeCheckerBuilder extends AbstractTypeCheckerBuilder<UnspecializedTypeCheckerBuilder> {
  }

  class ObjectTypeBuilder extends AbstractTypeCheckerBuilder<ObjectTypeBuilder>  {
  }

  class ClassTypeBuilder extends AbstractTypeCheckerBuilder<ClassTypeBuilder>  {

    public FunctionTypeBuilder withMethod(String name) {
      return null;
    }
  }
  class FunctionTypeBuilder extends AbstractTypeCheckerBuilder<FunctionTypeBuilder>  {

  }
}
