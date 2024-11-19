from unusedFunctionParameterImported import ImportedParent, ParentWithDuplicatedParent

class ChildFromImported(ImportedParent):

    # SONARPY-1829 `method_defined_in_child_class_only` is not considered a member of ImportedParent class, thus S1172 is raised
    def method_defined_in_child_class_only(self, a): # Noncompliant
        #                                        ^
        return compute()

    def method_defined_in_child_class_only_and_not_used(self, a): # Noncompliant
        #                                                     ^
        return compute()

class ChildFromDuplicated(ParentWithDuplicatedParent):

    def do_something(self, a): # FN SONARPY-1829 ChildFromDuplicated has an unresolved type hierarchy, because of the duplicated parent classes
        return compute()

