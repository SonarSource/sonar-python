class DataFrame:
    def values_as_list(self):
        return self.values

    def aliased_values_as_list(self):
        dataframe = self
        return dataframe.values
