package org.sonar.python.types.v3;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import org.sonar.python.types.v2.ClassType;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.TriBool;

public class TypeCheckerPoc {
  // METHODS
  static InnerPredicateBuilder<UnspecializedTypeCheckerBuilder, ClassTypeBuilder> isClass() {
    return builder -> {
      builder.addPredicate(type -> type instanceof ClassType ? TriBool.TRUE : TriBool.FALSE);
      return new ClassTypeBuilder(builder);
    };
  }

  // INTERFACES
  interface TypeCheckerBuilder<SELF extends TypeCheckerBuilder<SELF>> {
    <O extends TypeCheckerBuilder<O>> O with(InnerPredicateBuilder<SELF, O> predicate);

    TypeChecker build();
  }
  interface InnerPredicateBuilder<I extends TypeCheckerBuilder<? extends I>, O extends TypeCheckerBuilder<? extends  O>> {
    O construct(I input);

    default InnerPredicateBuilder<I, O> anyMatch() {
      return this;
    }
  }

  // CLASSES

  abstract static class AbstractTypeCheckerBuilder<SELF extends AbstractTypeCheckerBuilder<SELF>> implements TypeCheckerBuilder<SELF> {
    private List<Function<PythonType, TriBool>> predicates = new ArrayList<>();

    protected AbstractTypeCheckerBuilder(AbstractTypeCheckerBuilder<?> input) {
      this.predicates = new ArrayList<>(input.predicates);
    }

    private AbstractTypeCheckerBuilder() {
    }

    public void addPredicate(Function<PythonType, TriBool> predicate) {
      predicates.add(predicate);
    }

    @Override
    public <O extends TypeCheckerBuilder<O>> O with(InnerPredicateBuilder<SELF, O> predicate) {
      // TODO fix generics (if possible)
      return predicate.construct((SELF) this);
    }

    public TypeChecker build() {
      return new TypeChecker(predicates);
    }
  }

  static class UnspecializedTypeCheckerBuilder extends AbstractTypeCheckerBuilder<UnspecializedTypeCheckerBuilder> {
  }

  static class ObjectTypeBuilder extends AbstractTypeCheckerBuilder<ObjectTypeBuilder> {
    public ObjectTypeBuilder(AbstractTypeCheckerBuilder<?> builder) {
      super(builder);
    }
  }

  static class ClassTypeBuilder extends AbstractTypeCheckerBuilder<ClassTypeBuilder> {
    public ClassTypeBuilder(AbstractTypeCheckerBuilder<?> builder) {
      super(builder);
    }
  }

  static class TypeChecker {
    private final List<Function<PythonType, TriBool>> predicates;

    public TypeChecker(List<Function<PythonType, TriBool>> predicates) {
      this.predicates = new ArrayList<>(predicates);
    }

    boolean isTrue(PythonType pythonType) {
      return predicates.stream().allMatch(predicate -> predicate.apply(pythonType) == TriBool.TRUE);
    }
  }
}
