from unusedFunctionParameterImported import ImportedParent
class ChildFromImported(ImportedParent):

    # SONARPY-2327 `method_defined_in_child_class_only` is not considered a member of ImportedParent class, thus S1172 is raised
    def method_defined_in_child_class_only(self, a): # Noncompliant
        #                                        ^
        return compute()

    def method_defined_in_child_class_only_and_not_used(self, a): # Noncompliant
        #                                                     ^
        return compute()
