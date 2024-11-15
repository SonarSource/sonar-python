class ImportedParent:
    def using_child_method(self):
        return self.method_defined_in_child_class_only(1,2)
