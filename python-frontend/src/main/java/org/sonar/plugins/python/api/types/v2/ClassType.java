/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.plugins.python.api.types.v2;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import org.sonar.api.Beta;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.plugins.python.api.TriBool;

/**
 * ClassType
 */
@Beta
public final class ClassType implements PythonType {

  private final String name;
  private final String fullyQualifiedName;
  private final Set<Member> members;
  private final List<PythonType> attributes;
  private final List<TypeWrapper> superClasses;
  private final List<PythonType> metaClasses;
  private final boolean hasDecorators;
  private final boolean isGeneric;
  private final LocationInFile locationInFile;

  public ClassType(
    String name,
    String fullyQualifiedName,
    Set<Member> members,
    List<PythonType> attributes,
    List<TypeWrapper> superClasses,
    List<PythonType> metaClasses,
    boolean hasDecorators,
    boolean isGeneric,
    @Nullable LocationInFile locationInFile) {
    this.name = name;
    this.fullyQualifiedName = fullyQualifiedName;
    this.members = members;
    this.attributes = attributes;
    this.superClasses = superClasses;
    this.metaClasses = metaClasses;
    this.hasDecorators = hasDecorators;
    this.isGeneric = isGeneric;
    this.locationInFile = locationInFile;
  }

  public ClassType(String name, String fullyQualifiedName) {
    this(name, fullyQualifiedName, new HashSet<>(), new ArrayList<>(), new ArrayList<>(), new ArrayList<>(), false, false, null);
  }

  @Override
  public Optional<String> displayName() {
    return Optional.of("type");
  }

  @Override
  public Optional<String> instanceDisplayName() {
    var splits = name.split("\\.");
    if (splits.length > 0) {
      return Optional.of(splits[splits.length - 1]);
    }
    return Optional.of(name);
  }

  @Override
  public TriBool isCompatibleWith(PythonType another) {
    if (another instanceof SelfType selfType) {
      return this.isCompatibleWith(selfType.innerType());
    }
    if (another instanceof ObjectType objectType) {
      return this.isCompatibleWith(objectType.type());
    }
    if (another instanceof UnionType unionType) {
      return unionType.candidates().stream()
        .map(this::isCompatibleWith)
        .reduce(TriBool.FALSE, TriBool::or);
    }
    if (another instanceof FunctionType functionType) {
      return this.isCompatibleWith(functionType.returnType());
    }
    if (another instanceof ClassType classType) {
      if ((this.hasDecorators() && this.isUserDefinedType()) || (classType.hasDecorators() && classType.isUserDefinedType())) {
        return TriBool.UNKNOWN;
      }
      var isASubClass = this.isASubClassFrom(classType);
      var areAttributeCompatible = this.areAttributesCompatible(classType);
      var isDuckTypeCompatible = !this.members.isEmpty() && this.members.containsAll(classType.members);
      boolean isCompatible = Objects.equals(this, another)
        || "builtins.object".equals(classType.name())
        || isDuckTypeCompatible
        || (isASubClass && areAttributeCompatible);
      return isCompatible ? TriBool.TRUE : TriBool.FALSE;
    }
    return TriBool.UNKNOWN;
  }

  @Beta
  public boolean isASubClassFrom(ClassType other) {
    return superClasses().stream().anyMatch(superClass -> superClass.type().isCompatibleWith(other).isTrue());
  }

  @Beta
  public boolean areAttributesCompatible(ClassType other) {
    return attributes.stream().allMatch(attr -> other.attributes.stream().anyMatch(otherAttr -> attr.isCompatibleWith(otherAttr).isTrue()));
  }

  @Override
  public String key() {
    return Optional.of(attributes())
      .stream()
      .flatMap(Collection::stream)
      .map(PythonType::key)
      .collect(Collectors.joining(",", name() + "[", "]"));
  }

  @Override
  public Optional<PythonType> resolveMember(String memberName) {
    return resolveMember(memberName, new HashSet<>());
  }

  private Optional<PythonType> resolveMember(String memberName, Set<PythonType> visited) {
    visited.add(this);
    return localMember(memberName)
      .or(() -> inheritedMember(memberName, visited));
  }

  Optional<PythonType> localMember(String memberName) {
    return members.stream()
      .filter(m -> m.name().equals(memberName))
      .map(Member::type)
      .findFirst();
  }

  public Optional<PythonType> inheritedMember(String memberName) {
    return inheritedMember(memberName, new HashSet<>());
  }

  private Optional<PythonType> inheritedMember(String memberName, Set<PythonType> visited) {
    return superClasses().stream()
      .map(TypeWrapper::type)
      .filter(Predicate.not(visited::contains))
      .map(t -> {
        visited.add(t);
        if (t instanceof ClassType superClassType) {
          return superClassType.resolveMember(memberName, visited);
        }
        return t.resolveMember(memberName);
      })
      .filter(Optional::isPresent)
      .map(Optional::get)
      .findFirst();
  }

  public boolean hasUnresolvedHierarchy() {
    return superClasses().stream().anyMatch(s -> {
        if (s.type() instanceof ClassType parentClassType) {
          return parentClassType.hasUnresolvedHierarchy();
        }
        return true;
      }
    );
  }

  /**
   * Computes the C3 linearization (Method Resolution Order) for this class.
   *
   * <p>Returns {@code Optional.empty()} if:
   * <ul>
   *   <li>the type hierarchy is unresolved ({@link #hasUnresolvedHierarchy()} is {@code true}), or</li>
   *   <li>the MRO cannot be computed because the base class ordering creates a conflict
   *       (i.e. what Python raises as a {@code TypeError} at class definition time).</li>
   * </ul>
   *
   * <p>The returned list starts with {@code this} and ends with the most-distant ancestor.
   *
   * @see <a href="https://docs.python.org/3/howto/mro.html">Python MRO documentation</a>
   */
  public Optional<List<ClassType>> mro() {
    if (hasUnresolvedHierarchy()) {
      return Optional.empty();
    }
    return ClassTypeMroUtils.compute(this);
  }

  /**
   * Returns whether C3 linearization would succeed at runtime for a hypothetical class whose direct
   * bases are exactly {@code bases} in order. Uses Python's runtime view of built-in containers:
   * typeshed-only inheritance edges from {@code dict}/{@code list}/{@code set}/… to their
   * {@code collections.abc} / {@code typing} ABCs (which at runtime are
   * {@code abc.ABCMeta.register()} virtual subclasses, not real bases) are ignored. This way, valid
   * Python code such as {@code class N(MutableMapping, dict): pass} is not flagged.
   *
   * <p>Precondition: every element of {@code bases} is non-null and fully resolved (no unresolved
   * hierarchy), matching what {@link #mro()} requires for each base.
   *
   * <p>If any base has an empty {@link #mro()} because an ancestor's hierarchy is invalid, this
   * returns {@code true} — the failure is not attributable to the merge of these bases alone.
   */
  public static boolean wouldHaveValidMro(List<ClassType> bases) {
    return ClassTypeMroUtils.wouldHaveValidMro(bases);
  }

  /**
   * Returns {@code true} for built-in containers whose typeshed declaration includes fictional
   * ABC inheritance that does not exist at runtime (e.g. {@code dict}, {@code list}, {@code set}).
   * See {@link #wouldHaveValidMro(List)}.
   */
  public boolean isVirtualAbcSubclassingBuiltin() {
    return ClassTypeMroUtils.isVirtualAbcSubclassingBuiltin(this);
  }

  @Override
  public TriBool hasMember(String memberName) {
    // a ClassType is an object of class type, it has the same members as those present on any type
    if ("__call__".equals(memberName)) {
      return TriBool.TRUE;
    }
    if (resolveMember(memberName).filter(t -> t != PythonType.UNKNOWN).isPresent()) {
      return TriBool.TRUE;
    }
    if (hasUnresolvedHierarchy()) {
      return TriBool.UNKNOWN;
    }
    // TODO: Not correct, we should look at what the actual type is instead (SONARPY-1666)
    return TriBool.UNKNOWN;
  }

  public boolean hasMetaClass() {
    return !this.metaClasses.isEmpty() ||
      this.superClasses()
        .stream()
        .map(TypeWrapper::type)
        .filter(ClassType.class::isInstance)
        .map(ClassType.class::cast)
        .anyMatch(ClassType::hasMetaClass);
  }

  public TriBool instancesHaveMember(String memberName) {
    if (hasUnresolvedHierarchy() || hasMetaClass() || (hasDecorators() && isUserDefinedType())) {
      return TriBool.UNKNOWN;
    }
    if ("NamedTuple".equals(this.name)) {
      // TODO: instances of NamedTuple are type
      return TriBool.TRUE;
    }
    if ("function".equals(this.name) && "__call__".equals(memberName)) {
      // __call__ is not formally defined on function, but is present
      return TriBool.TRUE;
    }
    return resolveMember(memberName).isPresent() ? TriBool.TRUE : TriBool.FALSE;
  }

  @Override
  public Optional<LocationInFile> definitionLocation() {
    return Optional.ofNullable(this.locationInFile);
  }

  private boolean isUserDefinedType() {
    // assumes if definedLocation is not present that the type is loaded from typeshed
    return definitionLocation().isPresent();
  }

  @Override
  public String toString() {
    return "ClassType[%s]".formatted(name);
  }

  @Override
  public String name() {
    return name;
  }

  public String fullyQualifiedName() {
    return fullyQualifiedName;
  }

  public Set<Member> members() {
    return members;
  }

  public List<PythonType> attributes() {
    return attributes;
  }

  public List<TypeWrapper> superClasses() {
    return superClasses;
  }

  public List<PythonType> metaClasses() {
    return metaClasses;
  }

  public boolean hasDecorators() {
    return hasDecorators;
  }

  public boolean isGeneric() {
    return isGeneric;
  }
}
