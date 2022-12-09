#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Adjacency List data structure


It is an implementation of the adjacency list data structure based on graph data structure

Author: Kaveh Mahdavi <kavehmahdavi74@yahoo.com>
License: BSD 3 clause
last update: 30/09/2022
"""
import matplotlib.pyplot as plt
import networkx as nx


class Vertex:
    """ Base class of graph

    """

    def __init__(self, node):
        self.id = node
        self.adjacent = {}
        self.max_cost = {}
        self.number_of_edges = 0

    def __str__(self):
        return str(self.id) + ' adjacent: ' + str([x.id for x in self.adjacent])

    def add_neighbor(self, neighbor, weight=0):
        """ Assassinate a neighbor to hte vertex

        Args:
            neighbor (int): indicates the neighbor index
            weight (float): indicates the weight of the edge

        Returns:
            Self, it is applicable as a chain
        """
        self.adjacent[neighbor] = weight
        self.number_of_edges += 1
        if not bool(self.max_cost):
            self.max_cost['neighbor'] = neighbor
            self.max_cost['distance'] = weight
        elif self.max_cost['distance'] < weight:
            self.max_cost['neighbor'] = neighbor
            self.max_cost['distance'] = weight
        return self

    def remove_neighbor(self, neighbor):
        """ Eliminate an edge to the vertex

        Args:
            neighbor (int): indicates the neighbor index in the list

        Returns:
            Self, it is applicable as a chain
        """
        if neighbor not in self.adjacent:
            raise ValueError('Error: There is not any connection to {}'.format(neighbor))
        else:
            del self.adjacent[neighbor]
            self.number_of_edges -= 1

            # Update the max_cost
            if neighbor == self.max_cost['neighbor']:
                maxKey, maxValue = max(self.adjacent.items(), key=lambda x: x[1])
                self.max_cost['neighbor'] = maxKey
                self.max_cost['distance'] = maxValue
        return self

    def get_connections(self):
        """ Compute the list of the adjacent vertexes

        Returns:
            A lis t of the adjacent vertexes
        """
        return self.adjacent.keys()

    def get_id(self):
        """ Reuters the vertex id

        Returns:
            An int indicate the vertex id
        """
        return self.id

    def get_weight(self, neighbor):
        """ Reuters the edge weight to a neighbor

        Returns:
            An float indicates the weight
        """
        return self.adjacent[neighbor]

    def update_weight(self, neighbor, new_weight):
        """ Updates an edges weight

        Args:
            neighbor (int): indicate the neighbor index
            new_weight (float): indicates the edge weight

        Returns:
            Self, it is applicable as a chain
        """
        self.adjacent[neighbor] = new_weight
        return self

    def neighbor_ids(self):
        """ Computes a list of the neighbors indexes

        Returns:
            A list of the indexes
        """
        return list([x.id for x in self.adjacent])


class AdjacencyList:
    def __init__(self, vertex=None, k_smallest_edges=5):
        self.vert_dict = {}
        self.num_vertices = 0
        self.k_smallest_edges = k_smallest_edges
        if vertex is not None:
            for vertex_item in vertex:
                self.add_vertex(vertex_item)

    def __iter__(self):
        return iter(self.vert_dict.values())

    def add_vertex(self, node):
        """ Adds a vertex to the list/graph

        Args:
            node(int/str): indicates the vertex id

        Returns:
            Self, it is applicable as a chain
        """
        self.num_vertices = self.num_vertices + 1
        new_vertex = Vertex(node)
        self.vert_dict[node] = new_vertex
        return new_vertex

    def get_vertex(self, node):
        """ Returns the vertex object

        Args:
            node (int/str): indicates the vertex id

        Returns:
            Self, it is applicable as a chain
        """
        if node in self.vert_dict:
            return self.vert_dict[node]
        else:
            return self

    def add_edge(self, _source, _target, _weight):
        """ Adds an edge to the graph

        Args:
            _source (int/str): indicates the source vertex
            _target (int/str): indicates the target vertex
            _weight (float): indicates the edge weight

        Returns:
            Self, it is applicable as a chain
        """
        if _source not in self.vert_dict:
            self.add_vertex(_source)
        if _target not in self.vert_dict:
            self.add_vertex(_target)
        if self.vert_dict[_source].number_of_edges < self.k_smallest_edges:
            self.vert_dict[_source].add_neighbor(self.vert_dict[_target], _weight)
        elif _weight < self.vert_dict[_source].max_cost['distance']:
            self.vert_dict[_source].remove_neighbor(self.vert_dict[_source].max_cost['neighbor'])
            self.vert_dict[_source].add_neighbor(self.vert_dict[_target], _weight)
        return self

    def update_edge(self, _source, _target, _weight=None, smallest=True):
        """ Updates an edge weight

        Args:
            _source (int/str): indicates the source vertex
            _target (int/str): indicates the target vertex
            _weight (float): indicates the edge weight
            smallest (boolean): If True it changes the weight if it is less than its existing weight

        Returns:
            Self, it is applicable as a chain
        """
        if _source not in self.vert_dict:
            raise ValueError('There is not any node {}'.format(_source))
        if _target not in self.vert_dict[_source].neighbor_ids():
            raise ValueError('There is not any connection between {} and {}'.format(_source, _target))
        if smallest:
            if _weight < self.vert_dict[_source].get_weight(self.vert_dict[_target]):
                self.vert_dict[_source].update_weight(self.vert_dict[_target], _weight)
        else:
            self.vert_dict[_source].update_weight(self.vert_dict[_target], _weight)
        return self

    def merge(self, y, intersect='add'):
        """ Merges to adjacency list object

        Args:
            y (AdjacencyList object): indicates another AdjacencyList object
            intersect (staring): If there is an intersect edge, what should do ('add'/'update'/'knn')

        Returns:
            Self, it is applicable as a chain
        """
        # Adding the new vertexes
        vertexes_x = set(self.vert_dict.keys())
        vertexes_y = set(y.vert_dict.keys())
        difference_set = vertexes_y - vertexes_x
        if difference_set:
            for newVertex in difference_set:
                self.add_vertex(str(newVertex))

        # adding the new edges
        edges_x = {}
        edges_y = {}
        for v in self.vert_dict.values():
            for w in v.get_connections():
                vid = v.get_id()
                wid = w.get_id()
                edges_x.update({(vid, wid): v.get_weight(w)})
        for v in y:
            for w in v.get_connections():
                vid = v.get_id()
                wid = w.get_id()
                edges_y.update({(vid, wid): v.get_weight(w)})

        # Update the intersected edges
        # TODO: other options for intersection have to been added "update" and "knn".
        if intersect == 'add':
            for intersectedEdge in set(edges_x.keys()).intersection(set(edges_y.keys())):
                # If new cost is smaller, it will be updated.
                self.update_edge(intersectedEdge[0], intersectedEdge[1], edges_y[intersectedEdge])
        elif intersect == 'knn':
            raise ValueError('{} intersection method has not been defined yet.')
        elif intersect == 'update':
            raise ValueError('{} intersection method has not been defined yet.')
        for newEdge in set(set(edges_y.keys() - edges_x.keys())):
            self.add_edge(newEdge[0], newEdge[1], edges_y[newEdge])
        return self

    def to_networkx(self):
        """ Converts to the NetworkX object

        Returns:
            A NetworkX object
        """
        _nwx_graph = nx.Graph()
        for v in self.__iter__():
            for w in v.get_connections():
                vid = v.get_id()
                wid = w.get_id()
                _nwx_graph.add_edge(str(vid), str(wid), weight=v.get_weight(w))
        return _nwx_graph

    def plot_graph(self):
        """ Plots the NetworkX graph

        Returns:
            Self, it is applicable as a chain
        """
        _nwx_graph = self.to_networkx()
        _long_edges = [(u, v) for (u, v, d) in _nwx_graph.edges(data=True) if d['weight'] > 0.5]
        _short_edges = [(u, v) for (u, v, d) in _nwx_graph.edges(data=True) if d['weight'] <= 0.5]

        # positions for all nodes
        pos = nx.spring_layout(_nwx_graph)

        # draws nodes
        nx.draw_networkx_nodes(_nwx_graph, pos, node_size=700)

        # draws edges
        nx.draw_networkx_edges(_nwx_graph, pos, edgelist=_long_edges, width=6)
        nx.draw_networkx_edges(_nwx_graph, pos, edgelist=_short_edges, width=6, alpha=0.5, edge_color='b',
                               style='dashed')

        # adds labels
        nx.draw_networkx_labels(_nwx_graph, pos, font_size=20, font_family='sans-serif')

        plt.axis('off')
        plt.savefig("weighted_graph.png")  # save as png
        plt.show()  # display
        return self

    def get_vertices(self):
        """ Computes the vertexes list

        Returns:
            A list of the vertexes
        """
        return self.vert_dict.keys()


def __test():
    g = AdjacencyList(vertex=[])

    g.add_vertex('a')
    g.add_vertex('b')
    g.add_vertex('c')
    g.add_vertex('d')
    g.add_vertex('e')
    g.add_vertex('f')

    g.add_edge('a', 'b', 7)
    g.add_edge('a', 'c', 9)
    g.add_edge('a', 'f', 14)
    g.add_edge('b', 'c', 10)
    g.add_edge('b', 'd', 15)
    g.add_edge('b', 'f', 1)

    h = AdjacencyList(vertex=[])

    h.add_vertex('a')
    h.add_vertex('b')
    h.add_vertex('c')
    h.add_vertex('d')
    h.add_vertex('e')
    h.add_vertex('f')
    h.add_vertex('k')

    h.add_edge('a', 'b', 3)
    h.add_edge('b', 'd', 11)
    h.add_edge('c', 'f', 2)
    h.add_edge('d', 'e', 6)
    h.add_edge('e', 'f', 9)
    h.add_edge('f', 'b', 14)
    h.add_edge('k', 'b', 1)
    g.merge(h)

    for v in g:
        for w in v.get_connections():
            vid = v.get_id()
            wid = w.get_id()
            print('( %s , %s, %3d)' % (vid, wid, v.get_weight(w)))

    for v in g:
        print('g.vert_dict[%s]=%s' % (v.get_id(), g.vert_dict[v.get_id()]))

    print("=" * 50)

    for v in h:
        for w in v.get_connections():
            vid = v.get_id()
            wid = w.get_id()
            print('( %s , %s, %3d)' % (vid, wid, v.get_weight(w)))

    for v in h:
        print('g.vert_dict[%s]=%s' % (v.get_id(), h.vert_dict[v.get_id()]))

    g.plot_graph()

if __name__ == '__main__':
    __test()
