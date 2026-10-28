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
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import javax.annotation.CheckForNull;
import org.sonar.python.types.v2.TypeUtils;

/**
 * C3 linearization (Method Resolution Order) for {@link ClassType}.
 *
 * <p>Built-in containers listed in {@link #VIRTUAL_ABC_SUBCLASSING_BUILTIN_FQNS} have typeshed
 * declarations that lie about their bases (e.g. {@code class dict(MutableMapping, Generic)}),
 * but at runtime they only inherit from {@code object} and the ABC relationship is a
 * virtual-subclass registration via {@code abc.ABCMeta.register()}. {@link #compute(ClassType)}
 * preserves the typeshed view (used by {@link ClassType#mro()}), while
 * {@link #wouldHaveValidMro(List)} honours the runtime view (used to detect real MRO conflicts).
 */
final class ClassTypeMroUtils {

  private static final Set<String> VIRTUAL_ABC_SUBCLASSING_BUILTIN_FQNS = Set.of(
    "dict", "list", "tuple", "set", "frozenset", "str", "bytes", "bytearray", "collections.deque"
  );

  private static final int MAX_MRO_ALTERNATIVES = 32;

  private ClassTypeMroUtils() {
  }

  static Optional<List<ClassType>> compute(ClassType cls) {
    return compute(cls, new HashMap<>(), new HashSet<>(), false, Map.of());
  }

  /**
   * Keep every complete C3 order when a base is a union of class declarations.
   * A repeated union denotes the same choice throughout a diamond. Unsupported
   * bases, an invalid order, or too many combinations leave lookup conservative.
   */
  static Optional<List<List<ClassType>>> computeAlternatives(ClassType cls) {
    List<UnionType> unions = new ArrayList<>();
    Map<TypeWrapper, PythonType> resolvedBases = new IdentityHashMap<>();
    if (!collectBaseAlternatives(cls, Collections.newSetFromMap(new IdentityHashMap<>()),
      Collections.newSetFromMap(new IdentityHashMap<>()), unions, resolvedBases) || unions.isEmpty()) {
      return Optional.empty();
    }
    List<Map<UnionType, ClassType>> selections = List.of(new IdentityHashMap<>());
    for (UnionType union : unions) {
      if (selections.size() > MAX_MRO_ALTERNATIVES / union.candidates().size()) {
        return Optional.empty();
      }
      selections = expandSelections(selections, union);
    }
    List<List<ClassType>> results = new ArrayList<>();
    for (Map<UnionType, ClassType> selection : selections) {
      // Deferred wrappers may create a fresh union on every read. Use the exact
      // base values collected above throughout each complete linearization.
      Map<TypeWrapper, PythonType> bases = new IdentityHashMap<>();
      resolvedBases.forEach((wrapper, type) -> bases.put(wrapper, type instanceof UnionType union ? selection.get(union) : type));
      var order = compute(cls, new HashMap<>(), new HashSet<>(), false, bases);
      if (order.isEmpty()) {
        return Optional.empty();
      }
      results.add(order.get());
    }
    return Optional.of(results);
  }

  private static List<Map<UnionType, ClassType>> expandSelections(List<Map<UnionType, ClassType>> selections, UnionType union) {
    List<Map<UnionType, ClassType>> expanded = new ArrayList<>();
    for (Map<UnionType, ClassType> selection : selections) {
      for (PythonType candidate : union.candidates()) {
        Map<UnionType, ClassType> alternative = new IdentityHashMap<>(selection);
        alternative.put(union, (ClassType) candidate);
        expanded.add(alternative);
      }
    }
    return expanded;
  }

  private static boolean collectBaseAlternatives(
    PythonType type, Set<ClassType> classes, Set<UnionType> visited, List<UnionType> unions, Map<TypeWrapper, PythonType> resolvedBases) {
    if (type instanceof ClassType cls) {
      return !classes.add(cls) || cls.superClasses().stream()
        .allMatch(wrapper -> collectBaseAlternatives(resolvedBases.computeIfAbsent(wrapper, TypeWrapper::type), classes, visited, unions, resolvedBases));
    }
    if (type instanceof UnionType union && !union.candidates().isEmpty()
      && union.candidates().stream().allMatch(ClassType.class::isInstance)) {
      if (!visited.add(union)) {
        return true;
      }
      unions.add(union);
      return union.candidates().stream().allMatch(base -> collectBaseAlternatives(base, classes, visited, unions, resolvedBases));
    }
    return false;
  }

  static Optional<PythonType> inheritedMember(List<List<ClassType>> orders, String memberName) {
    var members = orders.stream().map(order -> order.stream().skip(1)
      .map(cls -> cls.localMember(memberName)).flatMap(Optional::stream).findFirst()).toList();
    if (members.stream().allMatch(Optional::isEmpty)) {
      return Optional.empty();
    }
    // A member missing from one possible hierarchy must not become definite.
    return Optional.of(UnionType.or(members.stream().map(member -> member.orElse(PythonType.UNKNOWN))
      .map(type -> TypeUtils.map(type, TypeUtils::resolved)).toList()));
  }

  /**
   * Returns whether C3 would succeed at runtime for a hypothetical class whose direct bases are
   * exactly {@code bases} in order. Backs {@link ClassType#wouldHaveValidMro(List)}.
   */
  static boolean wouldHaveValidMro(List<ClassType> bases) {
    Map<ClassType, Optional<List<ClassType>>> cache = new HashMap<>();
    List<List<ClassType>> lists = new ArrayList<>();
    for (ClassType base : bases) {
      Optional<List<ClassType>> mro = compute(base, cache, new HashSet<>(), true, Map.of());
      if (mro.isEmpty()) {
        return true;
      }
      lists.add(new ArrayList<>(mro.get()));
    }
    lists.add(new ArrayList<>(bases));
    return c3Merge(lists) != null;
  }

  static boolean isVirtualAbcSubclassingBuiltin(ClassType cls) {
    String fqn = cls.fullyQualifiedName();
    return fqn != null && VIRTUAL_ABC_SUBCLASSING_BUILTIN_FQNS.contains(fqn);
  }

  /**
   * Recursively computes the C3 MRO for {@code cls}. {@code cache} memoises results across the
   * traversal (callers may pass a fresh map); {@code visiting} guards against inheritance cycles.
   */
  private static Optional<List<ClassType>> compute(
    ClassType cls,
    Map<ClassType, Optional<List<ClassType>>> cache,
    Set<ClassType> visiting,
    boolean stripVirtualAbcInheritance,
    Map<TypeWrapper, PythonType> selection
  ) {
    if (cache.containsKey(cls)) {
      return cache.get(cls);
    }
    if (stripVirtualAbcInheritance && isVirtualAbcSubclassingBuiltin(cls)) {
      // Runtime view: ignore the typeshed-declared parents but still keep {@code object} as the tail,
      // so C3 rejects orderings like {@code class X(object, dict)} (object must come after dict).
      // {@code object} is, by construction, the last element of {@code cls}'s non-stripped MRO.
      Optional<List<ClassType>> fullMro = compute(cls, new HashMap<>(), new HashSet<>(), false, selection);
      List<ClassType> mro = fullMro.map(m -> List.of(cls, m.get(m.size() - 1))).orElseGet(() -> List.of(cls));
      Optional<List<ClassType>> result = Optional.of(mro);
      cache.put(cls, result);
      return result;
    }
    if (!visiting.add(cls)) {
      // Cycle detected — treat as failure.
      return Optional.empty();
    }

    // ClassType.mro() rejects unresolved ancestry before calling this helper.
    // the resulting partial order must not be used as a complete MRO.
    List<ClassType> parentTypes = cls.superClasses().stream()
      .map(wrapper -> selection.containsKey(wrapper) ? selection.get(wrapper) : wrapper.type())
      .filter(ClassType.class::isInstance)
      .map(ClassType.class::cast)
      .toList();

    List<List<ClassType>> lists = new ArrayList<>();
    for (ClassType parent : parentTypes) {
      Optional<List<ClassType>> parentMro = compute(parent, cache, visiting, stripVirtualAbcInheritance, selection);
      if (parentMro.isEmpty()) {
        // A parent's MRO is invalid — Python would have raised TypeError there, not here.
        visiting.remove(cls);
        cache.put(cls, Optional.empty());
        return Optional.empty();
      }
      lists.add(new ArrayList<>(parentMro.get()));
    }
    lists.add(new ArrayList<>(parentTypes));

    List<ClassType> merged = c3Merge(lists);
    Optional<List<ClassType>> result;
    if (merged == null) {
      result = Optional.empty();
    } else {
      merged.add(0, cls);
      result = Optional.of(merged);
    }

    visiting.remove(cls);
    cache.put(cls, result);
    return result;
  }

  /** Returns the C3 merge of {@code lists}, or {@code null} if no valid merge exists. */
  @CheckForNull
  private static List<ClassType> c3Merge(List<List<ClassType>> lists) {
    List<ClassType> result = new ArrayList<>();
    while (true) {
      lists.removeIf(List::isEmpty);
      if (lists.isEmpty()) {
        return result;
      }
      ClassType candidate = findFirstValidMergeHead(lists);
      if (candidate == null) {
        return null;
      }
      result.add(candidate);
      removeHeadFromLists(lists, candidate);
    }
  }

  /** Returns the first list head that doesn't appear as a non-head in any list, or {@code null}. */
  @CheckForNull
  private static ClassType findFirstValidMergeHead(List<List<ClassType>> lists) {
    for (List<ClassType> list : lists) {
      ClassType head = list.get(0);
      if (!headAppearsAsNonHeadInAnyList(head, lists)) {
        return head;
      }
    }
    return null;
  }

  private static boolean headAppearsAsNonHeadInAnyList(ClassType head, List<List<ClassType>> lists) {
    for (List<ClassType> other : lists) {
      for (int k = 1; k < other.size(); k++) {
        if (other.get(k) == head) {
          return true;
        }
      }
    }
    return false;
  }

  private static void removeHeadFromLists(List<List<ClassType>> lists, ClassType chosen) {
    for (List<ClassType> list : lists) {
      if (!list.isEmpty() && list.get(0) == chosen) {
        list.remove(0);
      }
    }
  }
}
