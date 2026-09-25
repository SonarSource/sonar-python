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
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import javax.annotation.CheckForNull;

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

  private ClassTypeMroUtils() {
  }

  static Optional<List<ClassType>> compute(ClassType cls) {
    return compute(cls, new HashMap<>(), new HashSet<>(), false);
  }

  /**
   * Returns whether C3 would succeed at runtime for a hypothetical class whose direct bases are
   * exactly {@code bases} in order. Backs {@link ClassType#wouldHaveValidMro(List)}.
   */
  static boolean wouldHaveValidMro(List<ClassType> bases) {
    Map<ClassType, Optional<List<ClassType>>> cache = new HashMap<>();
    List<List<ClassType>> lists = new ArrayList<>();
    for (ClassType base : bases) {
      Optional<List<ClassType>> mro = compute(base, cache, new HashSet<>(), true);
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
    boolean stripVirtualAbcInheritance
  ) {
    if (cache.containsKey(cls)) {
      return cache.get(cls);
    }
    if (stripVirtualAbcInheritance && isVirtualAbcSubclassingBuiltin(cls)) {
      // Runtime view: ignore the typeshed-declared parents but still keep {@code object} as the tail,
      // so C3 rejects orderings like {@code class X(object, dict)} (object must come after dict).
      // {@code object} is, by construction, the last element of {@code cls}'s non-stripped MRO.
      Optional<List<ClassType>> fullMro = compute(cls, new HashMap<>(), new HashSet<>(), false);
      List<ClassType> mro = fullMro.map(m -> List.of(cls, m.get(m.size() - 1))).orElseGet(() -> List.of(cls));
      Optional<List<ClassType>> result = Optional.of(mro);
      cache.put(cls, result);
      return result;
    }
    if (!visiting.add(cls)) {
      // Cycle detected — treat as failure.
      return Optional.empty();
    }

    // Non-ClassType superclasses are already filtered out by hasUnresolvedHierarchy()
    List<ClassType> parentTypes = cls.superClasses().stream()
      .map(TypeWrapper::type)
      .filter(ClassType.class::isInstance)
      .map(ClassType.class::cast)
      .toList();

    List<List<ClassType>> lists = new ArrayList<>();
    for (ClassType parent : parentTypes) {
      Optional<List<ClassType>> parentMro = compute(parent, cache, visiting, stripVirtualAbcInheritance);
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
