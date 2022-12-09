#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" K nearest nigher method

Author: Kaveh Mahdavi <kavehmahdavi74@gmail.com>
License: BSD 3 clause
last update: 29/09/2022
"""
import pandas as pd
import numpy as np
import math
import sys
import warnings
from scipy.spatial import distance_matrix
from kavica.distance_measure import euclidean_distance
from kavica.graph_data_structur import AdjacencyList
from kavica.imputation.base import compatible_data_structure


class KNN(object):
    """ K-nearest neighbors base class

    """

    def __init__(self, k_neighbors=None, metric='Euclidean', data=None, graph=True):
        self.neighbors = k_neighbors
        self.metric = metric
        self.origin_data = data
        if graph:
            if k_neighbors is None:
                self.knnGraph = None
            else:
                self.knnGraph = AdjacencyList(k_smallest_edges=k_neighbors)
        else:
            warnings.warn("The data structure is not canonical graph, but a KNN Matrix constructor.", UserWarning)

    def _optimal__k(self):
        """ Computes the optimal k value

        Returns:
            An int indicates the optimal k value
        """
        k = round(math.sqrt(self.origin_data.shape[0]))
        if math.fmod(k, 2) == 1:
            return k
        else:
            return k + 1

    @staticmethod
    def progress_bar(counter, total, bar_len=40, process_id=1, status='', functionality='Forming adjacency matrix'):
        """ Text progressbar

        The ProgressBar class manages the current progressbar, and the format of the line
        is given by a number of widgets.

        Args:
            counter (int): indicates the dynamic progressed value
            total (int): indicates the total tasks that will be done.
            bar_len (int): indicates the progressbar steps size
            process_id (int): indicates the specific process id among the multiprocess
            status (str): indicates the text explanation of the progressbar
            functionality (str): explains the associated functionality

        Returns:

        """
        filled_len = int(round(bar_len * counter / float(total)))
        percents = round(100.0 * counter / float(total), 1)
        bar = '|' * filled_len + '-' * (bar_len - filled_len)
        sys.stdout.write(
            '\r\033[1;36;m[%s] <%s> chunk_id <%s> %s%s ...%s' % (bar,
                                                                 functionality,
                                                                 process_id,
                                                                 percents,
                                                                 '%',
                                                                 status))

    def _get_k_euclidean_neighbors(self, adjacency_matrix, k=None):
        """ Compute the k neighbors

        It computes the neighbors in the euclidean feature space

        Args:
            adjacency_matrix (AdjacencyList object): includes the adjacency matrix
            k (int): indicates the number of the nearest neighbors

        Returns:
            Self, it is applicable as a chain
        """
        k += 1  # it is used instead of omitting the <i,i> neighboring

        # Add to the KNN graph.
        top_indexes = adjacency_matrix.apply(lambda x: pd.Series(x.nsmallest(k).index))
        top_indexes = top_indexes.to_dict('list')

        adjacency_matrix = adjacency_matrix.to_dict('list')

        # TODO: Combine with filter and ensemble
        # TODO: Use multiprocess here

        progress_line_len = len(top_indexes)
        for key_node, value_nodes in top_indexes.items():
            self.progress_bar(key_node, progress_line_len, 1, status=str(key_node) + "/" + str(progress_line_len))
            for value_node in value_nodes:
                if key_node != value_node:
                    self.knnGraph.add_edge(str(key_node), str(value_node), adjacency_matrix[key_node][value_node])
                else:
                    pass

        return self

    def _get_neighbors(self):
        """ Computes the KNN neighbors

        Returns:
            Self, it is applicable as a chain
        """
        # Add vertexes to the graph.
        _node_sList = self.origin_data.index

        for _node in _node_sList.tolist():
            self.knnGraph.add_vertex(str(_node))

        # fixme: use a distributed computing here

        # K_nearest neighbors are added as edges to the graph.
        if self.metric == "Euclidean":
            _d = pd.DataFrame(distance_matrix(self.origin_data, self.origin_data))
            self._get_k_euclidean_neighbors(adjacency_matrix=_d, k=self.neighbors)
        else:
            _adjacency_matrix_columns = _node_sList
            _adjacency_matrix_rows = _node_sList
            _total_lines_number = len(_adjacency_matrix_rows)
            # TODO: distribute the df and calculat the KNN partiality than integrate.
            for _row in _adjacency_matrix_rows:
                _adjacency_matrix_columns = _adjacency_matrix_columns.drop(_row)
                self.progress_bar(int(_row), _total_lines_number, 1, status=str(_row) + "/" + str(_total_lines_number))
                for j in _adjacency_matrix_columns:
                    distance = euclidean_distance(pd.to_numeric(self.origin_data.loc[j]),
                                                  pd.to_numeric(self.origin_data.loc[_row]))
                    self.knnGraph.add_edge(_row, j, distance)
        return self

    def print_knn(self):
        """ Print the knn graph

        Returns:
            A boolean which is True
        """
        for _vertex_item in self.knnGraph:
            for _edge_item in _vertex_item.get_connections():
                vid = _vertex_item.get_id()
                wid = _edge_item.get_id()
                print('( %s , %s, %.4f)' % (vid, wid, _vertex_item.get_weight(_edge_item)))

        for _vertex_item in self.knnGraph:
            print('Vertex Item_dict[%s]=%s' % (_vertex_item.get_id(), self.knnGraph.vert_dict[_vertex_item.get_id()]))
        return True

    def graph_to_matrix(self, binary=True):
        """ Convert the knn graph to a matrix data structure

        Args:
            binary (boolean): If True it computes a binary matrix. Else, it uses the euclidean distance

        Returns:
            A numpy ndarray includes the knn matrix.
        """
        matrix_shape = (self.knnGraph.num_vertices, self.knnGraph.num_vertices)
        knn_matrix = np.zeros(matrix_shape)
        if binary:
            for _vertex in self.knnGraph.__iter__():
                for _neighbor in _vertex.adjacent:
                    knn_matrix[int(_vertex.id), int(_neighbor.id)] = 1
        else:
            for _vertex in self.knnGraph.__iter__():
                for _neighbor in _vertex.adjacent:
                    knn_matrix[int(_vertex.get_id()), int(_neighbor.get_id())] = _vertex.get_weight(_neighbor)
        return knn_matrix

    def fit(self, dataset, header=True, index=True, adjacency_matrix=False, draw=False):
        """ Fit the knn data.

        Args:
            dataset (numpy.ndarray): indicates the dataset
            header (boolean): If True the dataset first row assume as header.
            index (boolean): If True the dataset first column assume as index.
            adjacency_matrix (boolean):If True it returns an adjacency matrix
            draw (boolean): If True it draws a graph plot.

        Returns:
            self : object Fitted estimator.
        """
        # Todo: preprocess and clean the data
        self.origin_data = compatible_data_structure(data=dataset, header=header, index=index)

        # Todo: check the sparsity
        # Estimate the optimal k value
        if self.neighbors is None:
            self.neighbors = self._optimal__k()

        self.knnGraph = AdjacencyList(k_smallest_edges=self.neighbors)
        self._get_neighbors()

        if draw:
            self.knnGraph.plot_graph()
        if adjacency_matrix:
            return self.graph_to_matrix()
        else:
            return self


class KnnMatrix(KNN):
    """ KNN matrix class

    It is used the matrix data structure to represent the KNN graph
    """

    def __init__(self, k_neighbors=None, metric='Euclidean', data=None, graph=True):
        super(KnnMatrix, self).__init__(k_neighbors, metric, data, graph)
        self.knn_matrix = None

    def _get_k_euclidean_neighbors(self, adjacency_matrix, k=None, graph=True):
        """ Compute the k neighbors

        It computes the neighbors in the euclidean feature space

        Args:
            adjacency_matrix (AdjacencyList object): includes the adjacency matrix
            k (int): indicates the number of the nearest neighbors
            graph (boolean): if True it computes the graph copy of the matrix

        Returns:
            Self, it is applicable as a chain
        """
        # Todo: If we use method=min/stochastic in rank, it should be better but more expensive.
        mask = adjacency_matrix.rank(method='first', axis=1) <= k + 1
        self.knn_matrix = adjacency_matrix.where(mask, 0)
        if graph:
            super(KnnMatrix, self)._get_k_euclidean_neighbors(adjacency_matrix, k)
        return self

    def _get_neighbors(self):
        """ Computes the KNN neighbors

        Returns:
            Self, it is applicable as a chain
        """
        # Add the vertex to the graph.
        nodes_list = self.origin_data.index
        for node in nodes_list.tolist():
            self.knnGraph.add_vertex(str(node))

        # K nearest neighbors are added as edges to the graph.
        # fixme: use a distributed computing here
        if self.metric == "Euclidean":
            d = pd.DataFrame(distance_matrix(self.origin_data, self.origin_data))
            self._get_k_euclidean_neighbors(adjacency_matrix=d, k=self.neighbors)
        else:
            adjacency_matrix_columns = nodes_list
            adjacency_matrix_rows = nodes_list
            total_lines_number = len(adjacency_matrix_rows)
            # TODO: distribute the df and calculat the KNN partiality than integrate.
            for i in adjacency_matrix_rows:
                adjacency_matrix_columns = adjacency_matrix_columns.drop(i)
                self.progress_bar(int(i), total_lines_number, 1, status=str(i) + "/" + str(total_lines_number))
                for j in adjacency_matrix_columns:
                    distance = euclidean_distance(pd.to_numeric(self.origin_data.loc[j]),
                                                  pd.to_numeric(self.origin_data.loc[i]))
                    self.knnGraph.add_edge(i, j, distance)
        return self

    def graph_to_matrix(self, binary=True):
        """ Computes the adjacency matrix
        Args:
            binary (boolean): If True it computes a binary matrix only

        Returns:
            An adjacency matrix object
        """
        # Todo: add the binary computation into it
        return self.knn_matrix.values

    # todo
    def matrix_to_graph(self):
        pass


def _test_main():
    data0 = np.array([("ind", "F1", "F2", "F3", "F4", "F5", "F6"),
                      (0, 12, 0, 9, 5, 20, 89),
                      (1, 1, 1, 1, 1, 1, 1),
                      (2, 2, 2, 2, 2, 2, 2),
                      (3, 4, 45, 23, 24, 19, 16),
                      (4, 2, 44, 23, 22, 13, 11),
                      (5, 2, 4, 3, 2, 1, 1),
                      (6, 1, 1, 1, 1, 1, 1),
                      (7, 2, 2, 2, 2, 2, 2),
                      (8, 2, 45, 23, 24, 13, 16)])

    data2 = np.array(
        [["ind", "F1", "F2", "F3", "F4", "F5", "F6"],
         [0, 12, 0, 9, 5, 20, 89],
         [1, 1, 1, 1, 1, 1, 1],
         [2, 2, 2, 2, 2, 2, 2],
         [3, 4, 45, 23, 24, 19, 16],
         [4, 2, 44, 23, 22, 13, 11],
         [5, 2, 4, 3, 2, 1, 1],
         [6, 1, 1, 1, 1, 1, 1],
         [7, 2, 2, 2, 2, 2, 2],
         [8, 2, 45, 23, 24, 13, 16]])

    data1 = np.array(
        [['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10'],
         [0., -1.96234408, -1.30622481, -1.26386223, -1.3921304, -0.44232587, -1.35910009, -1.125, 0., -0.5],
         [1., 0.3540203, 0.88895015, 0.92448665, 0.82306145, 1.76930347, 0.89244255, 1.375, 0., -0.5],
         [2., 0.8636969, 0.77108165, 0.70248698, 0.82306145, 0.29488391, 0.7644938, 0.125, 0., 2.],
         [3., 0.39059454, -1.13773329, -1.17869255, -1.0443889, -0.44232587, -1.07832366, -1.125, 0., -0.5],
         [4., 0.35403234, 0.78392629, 0.81558115, 0.78847904, -1.17953565, 0.78048739, 0.75, 0., -0.5]])

    draft = KNN(k_neighbors=4)
    draft.fit(data1, adjacency_matrix=True, draw=False)
    output1 = draft.graph_to_matrix(binary=False)
    output1 = np.round(output1, decimals=1)
    output2 = draft.graph_to_matrix(binary=False)
    output2 = np.round(output2, decimals=1)
    print(draft.graph_to_matrix(binary=False))
    draft.print_knn()


if __name__ == '__main__':
    _test_main()
