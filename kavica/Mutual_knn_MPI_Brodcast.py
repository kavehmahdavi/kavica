# -*- coding: utf-8 -*-
# run: mpirun -np 4 python3 Mutual_knn_MPI_Brodcast.py
""" K nearest nigher method"""

# Author: Kaveh Mahdavi <kavehmahdavi74@gmail.com>
# License: BSD 3 clause

# TODO: divide teh data file based on CPU or Timestamps.
# fixme: when it is run from shell is ok, but in package
from distance_measure import euclideanDistance
from graph_data_structur import AdjacencyList
from imputation.base import data_structure_Compatibilization
import pandas as pd
import numpy as np
import pickle
from Mutual_knn import KNN as baseKNN
import warnings


class KNNBrodcast(baseKNN):

    def __init__(self, k_nighbors=None, metric='Euclidiean', data=None):
        super(KNNBrodcast, self).__init__(k_nighbors, metric, data)

    def _get_neighbors(self, subinstances=None):
        # Subinstances is list of the indexes. it has to convert to pandas.core.indexes.numeric.Float64Index
        # Initiate the instances for MPI.
        nodesList = self.originData.index
        if subinstances is None:
            adjacencyMatrixColumns = nodesList
            adjacencyMatrixRows = nodesList
        else:
            adjacencyMatrixColumns = nodesList
            adjacencyMatrixRows = pd.Index(subinstances)
        # Add the vertex to the graph.
        for node in nodesList.tolist():
            self.knnGraph.add_vertex(int(node))

        # fixme: use a distributed computing here
        # K nearest neighbors are added as edges to the graph.
        for i in adjacencyMatrixRows:
            adjacencyMatrixColumns = adjacencyMatrixColumns.drop(i)
            for j in adjacencyMatrixColumns:
                distance = euclideanDistance(pd.to_numeric(self.originData.loc[j]),
                                             pd.to_numeric(self.originData.loc[i]))
                self.knnGraph.add_edge(i, j, distance)

    # TODO: rewrite it
    def _graph_to_matrix(self, binary=True):
        # FIXME: it has problem when we have different array indexing. It just work when the index is a sequence.
        matrixShape = (self.knnGraph.num_vertices, self.knnGraph.num_vertices)
        knnMatrix = np.zeros(matrixShape)
        if binary:
            for vertex in self.knnGraph.__iter__():
                for nighbor in vertex.adjacent:
                    knnMatrix[int(vertex.id) - 1, int(nighbor.id) - 1] = 1
        else:
            for vertex in self.knnGraph.__iter__():
                for nighbor in vertex.adjacent:
                    knnMatrix[int(vertex.get_id()) - 1,
                              int(nighbor.get_id()) - 1] = vertex.get_weight(nighbor)

        return knnMatrix

    def fit(self, dataset, header=False, index=True,
            adjacencyMatrix=False, draw=False, subinstances=None):

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
        self._get_neighbors(subinstances)

        if draw:
            self.knnGraph.plot_graph()
        if adjacencyMatrix:
            return self._graph_to_matrix()


def data_reform(data, hasIndex=True, hasHeader=True, reindex=True):
    if hasHeader:
        data = np.delete(data, 0, 0)
        warnings.warn("The data frame header is eliminated.", UserWarning)
    else:
        print('The data dose not have header.')
    if hasIndex:
        if reindex:
            indexes = np.array(list(range(0, data.shape[0])))
            data[:, 0] = indexes
            warnings.warn("The data frame is reindexed.", UserWarning)
        elif len(np.unique(data[:, 0])) != len(data[:, 0]):
            raise ValueError('The data index is not unique. Refine the data or try reindexing.')
        else:
            warnings.warn("The original row index is used.", UserWarning)
    else:
        indexes = np.array(list(range(0, data.shape[0])))
        data[:, 0] = indexes
        warnings.warn("The data is indexed now.", UserWarning)
    return data.astype(np.float64)


def main():
    data = np.array([("ind", "F1", "F2", "F3", "F4", "F5", "F6"),
                     (1, 1, 1, 1, 1, 1, 1),
                     (2, 2, 2, 2, 2, 2, 2),
                     (3, 4, 45, 23, 24, 19, 16),
                     (4, 2, 44, 23, 22, 13, 11),
                     (6, 2, 4, 3, 2, 1, 1),
                     (6, 1, 1, 1, 1, 1, 1),
                     (7, 2, 2, 2, 2, 2, 2),
                     (8, 2, 45, 23, 24, 13, 16),
                     (9, 12, 0, 9, 5, 20, 89)])
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    if rank == 0:
        data = data_reform(data)

        # distributed the subinstances list among the cores
        split = np.array_split(data[:, 0], size, axis=0)  # Split input array by the number of available cores

        split_sizes = []
        for i in range(0, len(split), 1):
            split_sizes = np.append(split_sizes, len(split[i]))

        split_sizes = split_sizes * data.shape[1]
        displacements = np.insert(np.cumsum(split_sizes), 0, 0)[0:-1]
        split_sizes = split_sizes / data.shape[1]
        displacements = (displacements / data.shape[1])
        print("Input data split into vectors of sizes %s" % (split_sizes))
        print("Input data split with displacements of %s" % (displacements))

    else:
        # Create variables on other cores
        data = None
        split_sizes = None
        displacements = None

    # Fixme: it dose not work properly with -np 1, probably the problem is splitting
    split_sizes = comm.scatter(split_sizes, root=0)
    displacements = comm.scatter(displacements, root=0)
    subInstances = list(range(int(displacements), int(displacements + split_sizes)))

    data = comm.bcast(data, root=0)  # Broadcast split array to other cores

    knnRank = KNNBrodcast()
    knnRank.fit(data, adjacencyMatrix=False, draw=False, subinstances=subInstances)
    output = knnRank

    # TODO: gather the knnGraph from all nodes and integrate and return the final one.
    comm.Barrier()
    output = comm.gather(output, root=0)
    if rank == 0:
        for coreGraph in output[1:]:
            output[0].knnGraph.merga(coreGraph.knnGraph)
        # The final graph is saved as pickle in a file.
        knnObjectFile = 'knn.pkl'
        fileObject = open(knnObjectFile, 'wb')
        pickle.dump(output[0], fileObject)
        fileObject.close()

        # load the object from the file into var b
        fileObject = open('knn.pkl', 'rb')
        b = pickle.load(fileObject)
        b.knnPrint()
        fileObject.close()

    else:
        assert output is None


if __name__ == '__main__':
    main()
