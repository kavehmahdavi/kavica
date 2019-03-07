import numpy as np
import six
import pandas as pd

from kavica.transformer.TransformingFunctions import *
from kavica.transformer.BatchTransformer import TransformerFunction

__all__ = ['VerticalTransformer']


# TODO: It is needed to revised
class VerticalTransformer(TransformerFunction):
    """
    Transform is a tuple that is included:
            (str(new name), func(transform function), list(column/s),bool(replace)
           1- the transformation function
           2- the column/s that have to be transferred
           3- the name of the new column
           4- the add the main column or replace it with the transformed
    """

    def __init__(self, transformers, rest='transit'):
        super(VerticalTransformer, self).__init__()
        self.transformers = transformers  # [('name','transformation', [columns]),replace]
        self.rest = rest  # transit | cast
        self._remainder = None
        self._columns = []
        self.data = pd.DataFrame()
        self.X = pd.DataFrame()
        self.Y = pd.DataFrame()
        self.iterator = None
        self.yIndex = None

    @property
    def _transformers(self):
        return [(name, trans, reconstructor) for name, trans, _, reconstructor in self.transformers]

    @property
    def named_transformers_(self):
        """Access the fitted transformer by name.
        Read-only attribute to access any transformer by given name.
        Keys are transformer names and values are the fitted transformer
        objects.
        """
        # Use Bunch object to improve autocomplete
        return (dict([(name, trans) for name, trans, _, _ in self.transformers]))

    # checking the type of the data column that will be transformed
    def _get_column(self, X, key):

        def _check_key_type(key, superclass):

            if isinstance(key, superclass):
                return True
            if isinstance(key, slice):
                return (isinstance(key.start, (superclass, type(None))) and
                        isinstance(key.stop, (superclass, type(None))))
            if isinstance(key, list):
                return all(isinstance(x, superclass) for x in key)
            if hasattr(key, 'dtype'):
                if superclass is int:
                    return key.dtype.kind == 'i'
                else:
                    # superclass = six.string_types
                    return key.dtype.kind in ('O', 'U', 'S')
            return False

        if callable(key):
            key = key(X)

        # check whether we have string column names or integers
        if _check_key_type(key, int):
            column_names = False
        elif _check_key_type(key, six.string_types):
            column_names = True
        elif hasattr(key, 'dtype') and np.issubdtype(key.dtype, np.bool_):
            # boolean mask
            column_names = False
            if hasattr(X, 'loc'):
                # pandas boolean masks don't work with iloc, so take loc path
                column_names = True
        else:
            raise ValueError("No valid specification of the columns. Only a "
                             "scalar, list or slice of all integers or all "
                             "strings, or boolean mask is allowed")

        if column_names:
            if hasattr(X, 'loc'):
                # pandas dataframes
                return X.loc[:, key]
            else:
                raise ValueError("Specifying the columns using strings is only "
                                 "supported for pandas DataFrames")
        else:
            if hasattr(X, 'iloc'):
                return X.iloc[:, key]
            else:
                # numpy arrays, sparse arrays
                return X[:, key]

    def _check_key_type(self, key, superclass):
        """
        Check that scalar, list or slice is of a certain type.
        This is only used in _get_column and _get_column_indices to check
        if the `key` (column specification) is fully integer or fully string-like.
        Parameters
        ----------
        key : scalar, list, slice, array-like
            The column specification to check
        superclass : int or six.string_types
            The type for which to check the `key`
        """
        if isinstance(key, superclass):
            return True
        if isinstance(key, slice):
            return (isinstance(key.start, (superclass, type(None))) and
                    isinstance(key.stop, (superclass, type(None))))
        if isinstance(key, list):
            return all(isinstance(x, superclass) for x in key)
        if hasattr(key, 'dtype'):
            if superclass is int:
                return key.dtype.kind == 'i'
            else:
                # superclass = six.string_types
                return key.dtype.kind in ('O', 'U', 'S')

        return False

    # checking the name of the transformed data
    def _validate_names(self, names, X):
        invalid_names = None
        if len(set(names)) != len(names):
            raise ValueError('Provided names are not unique: '
                             '{0!r}'.format(list(names)))

        if not all(name for name in names):
            raise ValueError('All the transformation are needed to have name'
                             .format())

        if isinstance(X, pd.DataFrame):
            if X.columns.values.dtype != np.int64:
                invalid_names = set(names).intersection(X.columns.values)
            else:
                raise ValueError('The constructor arguments is {} and '
                                 ' It should not assinge a name to it.'
                                 .format(X.columns.values.dtype))

        elif isinstance(X, np.ndarray):
            if X.dtype.names:
                invalid_names = set(names).intersection(X.dtype.names)
            else:
                raise ValueError('The constructor arguments is {} and '
                                 ' It should not assign a name to it'
                                 .format('int64'))

        if invalid_names:
            raise ValueError('Estimator names conflict with constructor '
                             'arguments: {0!r}'.format(sorted(invalid_names)))

        invalid_names = [name for name in names if '__' in name]
        if invalid_names:
            raise ValueError('Estimator names must not contain __: got '
                             '{0!r}'.format(invalid_names))

    def _get_column_indices(self, X, key):
        """
        Get feature column indices for input data X and key.
        For accepted values of `key`, see the docstring of _get_column
        """
        n_columns = X.shape[1]

        if callable(key):
            key = key(X)

        if self._check_key_type(key, int):
            if isinstance(key, int):
                return [key]
            elif isinstance(key, slice):
                return list(range(n_columns)[key])
            else:
                return list(key)

        elif self._check_key_type(key, six.string_types):
            try:
                all_columns = list(X.columns)
            except AttributeError:
                raise ValueError("Specifying the columns using strings is only "
                                 "supported for pandas DataFrames")
            if isinstance(key, six.string_types):
                columns = [key]
            elif isinstance(key, slice):
                start, stop = key.start, key.stop
                if start is not None:
                    start = all_columns.index(start)
                if stop is not None:
                    # pandas indexing with strings is endpoint included
                    stop = all_columns.index(stop) + 1
                else:
                    stop = n_columns + 1
                return list(range(n_columns)[slice(start, stop)])
            else:
                columns = list(key)

            return [all_columns.index(col) for col in columns]

        elif hasattr(key, 'dtype') and np.issubdtype(key.dtype, np.bool_):
            # boolean mask
            return list(np.arange(n_columns)[key])
        else:
            raise ValueError("No valid specification of the columns. Only a "
                             "scalar, list or slice of all integers or all "
                             "strings, or boolean mask is allowed")

    def _validate_rest(self, X, Y=None):

        if self.rest not in ('transit', 'cast'):
            raise ValueError(
                "The rest column needs to be one of 'attach', 'detach',"
                " or estimator. '%s' was passed instead" %
                self.rest)

        n_columns = X.shape[1]
        cols = []
        yIndex = []

        for _, _, columns, _ in self.transformers:
            cols.extend(self._get_column_indices(X, columns))

        remaining_idx = sorted(list(set(range(n_columns)) - set(cols))) or None

        if Y:
            self.yIndex = self._get_column_indices(X, Y)
            if self.yIndex[0] in remaining_idx:
                remaining_idx.remove(self.yIndex[0])
                yIndex = self.yIndex
            else:
                yIndex = []
        self._remainder = ('rest', self.rest, remaining_idx, yIndex)
        print(self._remainder)

    def _validate_transformers(self, X):

        if not self.transformers:
            return

        names, transformers, _, reconstructor = zip(*self.transformers)

        # validate names
        self._validate_names(names, X)

        # validate reconstruction
        for t in reconstructor:
            if t in ('replace', 'add'):
                continue
            if (not (hasattr(t, "fit") or hasattr(t, "fit_transform")) or not
            hasattr(t, "transform")):
                raise TypeError("All estimators should implement fit and "
                                "transform, or can be 'replace' or 'save' "
                                "specifiers. '%s' (type %s) doesn't." %
                                (t, type(t)))

    def _validate_column_callables(self, X):
        """
        Converts callable column specifications.
        """
        columns = []
        for _, _, column, _ in self.transformers:
            if callable(column):
                column = column(X)
            elif column is not int:
                column = self._get_column_indices(X, column)
            columns.extend(column)
        self._columns = columns

    def __transform_generator(self, X):
        for trasition, column in zip(self.transformers, self._columns):
            names, transformers, _, reconstructor, column = (*trasition, column)
            yield (column,
                   X.iloc[:, column].name,
                   transformers,
                   names,
                   reconstructor,
                   np.array(X.iloc[:, column]),
                   X.iloc[:, column].dtype.name,)

    def fiting(self, X, Y=None):

        if isinstance(X, pd.DataFrame):
            pass
        elif isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        elif isinstance(X, list):
            X = pd.DataFrame(X)

        self.X = pd.DataFrame(index=X.index)

        self._validate_column_callables(X)
        self._validate_rest(X, Y)
        self._validate_transformers(X)

        # initiate the output X,Y
        if self.rest == 'transit':
            self.X = X.iloc[:, self._remainder[2]]
            if Y:
                self.Y = pd.DataFrame(index=X.index)
                if self._remainder[3]:
                    self.Y = X.iloc[:, self._remainder[3]]
        self.iterator = self.__transform_generator(X)

        return self

    def transform(self):
        for transform in self.iterator:
            transformedItem = self._transform(transform[5].reshape(1, -1), func=transform[2])
            if transform[4] == 'add':
                self.X[transform[1]] = transform[5]
            elif transform[4] == 'replace':
                pass
            else:
                raise TypeError("It is {} that is not replace or add".format(transform[4]))
            self.X[transform[3]] = transformedItem[0]


def main():
    data = np.array([(1, 9, 6, 13, 1, 72, 4),
                     (1, 9, 0, 13, 1, 12, 4),
                     (2, 2, 45, 23, 24, 13, 16),
                     (3, 12, 0, 9, 5, 20, 89)])
    datanp = np.array([(1, 9, 6),
                       (1, 9, 0),
                       (2, 2, 45),
                       (3, 12, 0)],
                      dtype={'col1': ('i1', 0, 'title 1'),
                             'col2': ('f4', 1, 'title 2'),
                             'col3': ('f4', 1, 'title 3')})

    headers = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    df = pd.DataFrame(data, columns=headers)
    data1 = [1, 9, 6, 13, 1, 72, 4]
    data2 = np.array([1, 9, 6, 13, 1, 72, 4])

    tr = [('T', testfunction, [4], 'add'), ('T1', testfunction, [1], 'replace')]  # slice(2, 4)
    t1 = VerticalTransformer(tr)

    a = [[1, 2, 4], [5, 3, 6], [7, 8, 9]]
    t1.fiting(df)
    t1.transform()
    print(t1.X)


if __name__ == '__main__':
    main()
