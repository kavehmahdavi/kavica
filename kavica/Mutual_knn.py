# -*- coding: utf-8 -*-
""" K nearest nigher method"""

# Author: Kaveh Mahdavi <kavehmahdavi74@gmail.com>
# License: BSD 3 clause

from kavica.distance_measure import euclideanDistance
from kavica.graph_data_structur import AdjacencyList
from kavica.imputation.mice import data_structure_Compatibilization
import pandas as pd
import numpy as np
import math
import sys
import warnings
from scipy.spatial import distance_matrix


class KNN(object):

    def __init__(self, k_nighbors=None, metric='Euclidean', data=None, graph=True):
        self.k_nighbors = k_nighbors
        self.metric = metric
        self.originData = data
        if graph:
            if k_nighbors is None:
                self.knnGraph = None
            else:
                self.knnGraph = AdjacencyList(kSmallestEdges=k_nighbors)
        else:
            warnings.warn("The data structure is not sufficient graph. It is a KNN Matrix constructor.",
                          UserWarning)

    def estimate_k(self):
        k = round(math.sqrt(self.originData.shape[0]))
        if math.fmod(k, 2) == 1:
            return k
        else:
            return k + 1

    @staticmethod
    def progress_bar(counter, total, process_id=1, status='', functionality='Forming adjacency matrix'):
        bar_len = 40
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

        k += 1  # it is used instead of omitting the <i,i> neighboring
        # Add to the KNN graph.
        top_indexes = adjacency_matrix.apply(lambda x: pd.Series(x.nsmallest(k).index))
        top_indexes = top_indexes.to_dict('list')

        adjacency_matrix = adjacency_matrix.to_dict('list')

        progress_line_len = len(top_indexes)
        # TODO: Combine with filter and ensemble
        # TODO: Use multiprocess here

        for key_node, value_nodes in top_indexes.items():
            self.progress_bar(key_node, progress_line_len, 1,
                              status=str(key_node) + "/" + str(progress_line_len))
            for value_node in value_nodes:
                if key_node != value_node:
                    self.knnGraph.add_edge(str(key_node), str(value_node), adjacency_matrix[key_node][value_node])
                else:
                    pass

        return self

    def _get_neighbors(self):

        # Add the vertex to the graph.
        nodesList = self.originData.index

        for node in nodesList.tolist():
            self.knnGraph.add_vertex(str(node))

        # fixme: use a distributed computing here
        # K nearest neighbors are added as edges to the graph.

        if self.metric == "Euclidean":
            D = pd.DataFrame(distance_matrix(self.originData, self.originData))
            self._get_k_euclidean_neighbors(adjacency_matrix=D, k=self.k_nighbors)

        else:
            adjacencyMatrixColumns = nodesList
            adjacencyMatrixRows = nodesList
            totalLinesNumber = len(adjacencyMatrixRows)
            # TODO: distribute the df and calculat the KNN partiality than integrate.
            for i in adjacencyMatrixRows:
                adjacencyMatrixColumns = adjacencyMatrixColumns.drop(i)
                self.progress_bar(int(i), totalLinesNumber, 1, status=str(i) + "/" + str(totalLinesNumber))
                for j in adjacencyMatrixColumns:
                    distance = euclideanDistance(pd.to_numeric(self.originData.loc[j]),
                                                 pd.to_numeric(self.originData.loc[i]))
                    self.knnGraph.add_edge(i, j, distance)

    def knnPrint(self):
        for vertexItem in self.knnGraph:
            for edgeItem in vertexItem.get_connections():
                vid = vertexItem.get_id()
                wid = edgeItem.get_id()
                print('( %s , %s, %.4f)' % (vid, wid, vertexItem.get_weight(edgeItem)))

        for vertexItem in self.knnGraph:
            print('Vertex Item_dict[%s]=%s' % (vertexItem.get_id(),
                                               self.knnGraph.vert_dict[vertexItem.get_id()]))

    def graph_to_matrix(self, binary=True):
        matrixShape = (self.knnGraph.num_vertices, self.knnGraph.num_vertices)

        knnMatrix = np.zeros(matrixShape)

        if binary:
            for vertex in self.knnGraph.__iter__():
                for nighbor in vertex.adjacent:
                    knnMatrix[int(vertex.id), int(nighbor.id)] = 1
        else:
            for vertex in self.knnGraph.__iter__():
                for nighbor in vertex.adjacent:
                    knnMatrix[int(vertex.get_id()),
                              int(nighbor.get_id())] = vertex.get_weight(nighbor)

        return knnMatrix

    def fit(self, dataset, header=True, index=True, adjacencyMatrix=False, draw=False):

        # Todo: pre_possessing the data
        self.originData = data_structure_Compatibilization(data=dataset,
                                                           header=header,
                                                           index=index)

        # Todo: check the sparsity
        # Estimate the K
        if self.k_nighbors is None:
            self.k_nighbors = self.estimate_k()
        else:
            pass

        self.knnGraph = AdjacencyList(kSmallestEdges=self.k_nighbors)
        self._get_neighbors()

        if draw:
            self.knnGraph.plot_graph()
        if adjacencyMatrix:
            return self.graph_to_matrix()


class KNNMatrix(KNN):

    def __init__(self, k_nighbors=None, metric='Euclidean', data=None, graph=True):
        super(KNNMatrix, self).__init__(k_nighbors, metric, data, graph)
        self.knnMatrix = None

    def _get_k_euclidean_neighbors(self, adjacency_matrix, k=None, graph=True):

        # Todo: If we use method=min/stochastic in rank, it should be better but more expensive.

        mask = adjacency_matrix.rank(method='first', axis=1) <= k + 1
        self.knnMatrix = adjacency_matrix.where(mask, 0)
        if graph:
            super(KNNMatrix, self)._get_k_euclidean_neighbors(adjacency_matrix, k)

        return self

    def _get_neighbors(self):
        # Add the vertex to the graph.
        nodesList = self.originData.index
        for node in nodesList.tolist():
            self.knnGraph.add_vertex(str(node))

        # fixme: use a distributed computing here
        # K nearest neighbors are added as edges to the graph.

        if self.metric == "Euclidean":
            D = pd.DataFrame(distance_matrix(self.originData, self.originData))
            self._get_k_euclidean_neighbors(adjacency_matrix=D, k=self.k_nighbors)
        else:
            adjacencyMatrixColumns = nodesList
            adjacencyMatrixRows = nodesList
            totalLinesNumber = len(adjacencyMatrixRows)
            # TODO: distribute the df and calculat the KNN partiality than integrate.
            for i in adjacencyMatrixRows:
                djacencyMatrixColumns = adjacencyMatrixColumns.drop(i)
                self.progress_bar(int(i), totalLinesNumber, 1, status=str(i) + "/" + str(totalLinesNumber))
                for j in djacencyMatrixColumns:
                    distance = euclideanDistance(pd.to_numeric(self.originData.loc[j]),
                                                 pd.to_numeric(self.originData.loc[i]))
                    self.knnGraph.add_edge(i, j, distance)

    def graph_to_matrix(self, binary=True):
        return self.knnMatrix.values

    def matrix_to_graph(self):
        pass


'''
def main():
    data1 = np.array([("ind", "F1", "F2", "F3", "F4", "F5", "F6"),
                      (0, 12, 0, 9, 5, 20, 89),
                      (1, 1, 1, 1, 1, 1, 1),
                      (2, 2, 2, 2, 2, 2, 2),
                      (3, 4, 45, 23, 24, 19, 16),
                      (4, 2, 44, 23, 22, 13, 11),
                      (5, 2, 4, 3, 2, 1, 1),
                      (6, 1, 1, 1, 1, 1, 1),
                      (7, 2, 2, 2, 2, 2, 2),
                      (8, 2, 45, 23, 24, 13, 16)])

    data11 = np.array(
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
         [0., -1.96234408, -1.30622481, -1.26386223, -1.39021304, -0.44232587, -1.35910009, -1.125, 0., -0.5],
         [1., 0.3540203, 0.88895015, 0.92448665, 0.82306145, 1.76930347, 0.89244255, 1.375, 0., -0.5],
         [2., 0.8636969, 0.77108165, 0.70248698, 0.82306145, 0.29488391, 0.7644938, 0.125, 0., 2.],
         [3., 0.39059454, -1.13773329, -1.17869255, -1.0443889, -0.44232587, -1.07832366, -1.125, 0., -0.5],
         [4., 0.35403234, 0.78392629, 0.81558115, 0.78847904, -1.17953565, 0.78048739, 0.75, 0., -0.5]])

    test = KNN(k_nighbors=4)
    test.fit(data1, adjacencyMatrix=True, draw=False)
    output1 = test.graph_to_matrix(binary=False)
    output1 = np.round(output1, decimals=1)
    output2 = test.graph_to_matrix(binary=False)
    output2 = np.round(output2, decimals=1)
    print(test.graph_to_matrix(binary=False))


if __name__ == '__main__':
    main()
'''
