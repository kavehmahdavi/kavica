# -*- coding: utf-8 -*-
""" Adjacency List data structure"""

# Author: Kaveh Mahdavi <kavehmahdavi74@gmail.com>
# License: BSD 3 clause

import matplotlib.pyplot as plt
import networkx as nx


class Vertex:

    def __init__(self, node):
        self.id = node
        self.adjacent = {}
        self.max_cost = {}
        self.numberOfEdges = 0

    def __str__(self):
        return str(self.id) + ' adjacent: ' + str([x.id for x in self.adjacent])

    def add_neighbor(self, neighbor, weight=0):
        self.adjacent[neighbor] = weight
        self.numberOfEdges += 1
        if not bool(self.max_cost):
            self.max_cost['neighbor'] = neighbor
            self.max_cost['distance'] = weight
        elif self.max_cost['distance'] < weight:
            self.max_cost['neighbor'] = neighbor
            self.max_cost['distance'] = weight

    def remove_neighbor(self, neighbor):
        if neighbor not in self.adjacent:
            raise ValueError('There is not any connection to {}'.format(neighbor))
        else:
            del self.adjacent[neighbor]
            self.numberOfEdges -= 1

            # Update the max_cost
            if neighbor == self.max_cost['neighbor']:
                maxKey, maxValue = max(self.adjacent.items(), key=lambda x: x[1])
                self.max_cost['neighbor'] = maxKey
                self.max_cost['distance'] = maxValue

    def get_connections(self):
        return self.adjacent.keys()

    def get_id(self):
        return self.id

    def get_weight(self, neighbor):
        return self.adjacent[neighbor]

    def update_weight(self, neighbor, newWeight):
        self.adjacent[neighbor] = newWeight

    def neighborIds(self):
        return list([x.id for x in self.adjacent])


class AdjacencyList:
    def __init__(self, vertex=None, kSmallestEdges=5):
        self.vert_dict = {}
        self.num_vertices = 0
        self.kSmallestEdges = kSmallestEdges
        if vertex is not None:
            for vertexItem in vertex:
                self.add_vertex(vertexItem)

    def __iter__(self):
        return iter(self.vert_dict.values())

    def add_vertex(self, node):
        self.num_vertices = self.num_vertices + 1
        new_vertex = Vertex(node)
        self.vert_dict[node] = new_vertex
        return new_vertex

    def get_vertex(self, n):
        if n in self.vert_dict:
            return self.vert_dict[n]
        else:
            return None

    def add_edge(self, frm, to, cost):
        if frm not in self.vert_dict:
            self.add_vertex(frm)
        if to not in self.vert_dict:
            self.add_vertex(to)
        if self.vert_dict[frm].numberOfEdges < self.kSmallestEdges:
            self.vert_dict[frm].add_neighbor(self.vert_dict[to], cost)
        elif cost < self.vert_dict[frm].max_cost['distance']:
            self.vert_dict[frm].remove_neighbor(self.vert_dict[frm].max_cost['neighbor'])
            self.vert_dict[frm].add_neighbor(self.vert_dict[to], cost)

    def update_edge(self, frm, to, newCost=None, smallest=True):
        if frm not in self.vert_dict:
            raise ValueError('There is not any node {}'.format(frm))
        if to not in self.vert_dict[frm].neighborIds():
            raise ValueError('There is not any connection between {} and {}'.format(frm, to))
        if smallest:
            if newCost < self.vert_dict[frm].get_weight(self.vert_dict[to]):
                self.vert_dict[frm].update_weight(self.vert_dict[to], newCost)
        else:
            self.vert_dict[frm].update_weight(self.vert_dict[to], newCost)

    def merga(self, y=None, intersect='add'):
        # Parameter intersect:staring ('add'/'update'/'knn'). If there is an intersect edge, what should do.
        # TODO: other options for adding the edge have to been added.
        # Adding the new vertexes
        vertexces_x = set(self.vert_dict.keys())
        vertexces_y = set(y.vert_dict.keys())
        defertialvertexcesSet = vertexces_y - vertexces_x
        if defertialvertexcesSet:
            for newVertex in defertialvertexcesSet:
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

        if intersect:
            for intersectedEdge in set(edges_x.keys()).intersection(set(edges_y.keys())):
                # If new cost is smaller, it will be updated.
                self.update_edge(intersectedEdge[0], intersectedEdge[1], edges_y[intersectedEdge])
        for newEdge in set(set(edges_y.keys() - edges_x.keys())):
            self.add_edge(newEdge[0], newEdge[1], edges_y[newEdge])

    def toNx(self):
        G = nx.Graph()
        for v in self.__iter__():
            for w in v.get_connections():
                vid = v.get_id()
                wid = w.get_id()
                G.add_edge(str(vid), str(wid), weight=v.get_weight(w))
        return G

    def plot_graph(self):
        G = self.toNx()
        elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 0.5]
        esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= 0.5]

        pos = nx.spring_layout(G)  # positions for all nodes

        # nodes
        nx.draw_networkx_nodes(G, pos, node_size=700)

        # edges
        nx.draw_networkx_edges(G, pos, edgelist=elarge,
                               width=6)
        nx.draw_networkx_edges(G, pos, edgelist=esmall,
                               width=6, alpha=0.5, edge_color='b', style='dashed')

        # labels
        nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')

        plt.axis('off')
        plt.savefig("weighted_graph.png")  # save as png
        plt.show()  # display

    def get_vertices(self):
        return self.vert_dict.keys()


if __name__ == '__main__':
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
    g.merga(h)

    for v in g:
        for w in v.get_connections():
            vid = v.get_id()
            wid = w.get_id()
            print('( %s , %s, %3d)' % (vid, wid, v.get_weight(w)))

    for v in g:
        print('g.vert_dict[%s]=%s' % (v.get_id(), g.vert_dict[v.get_id()]))

    '''
    print("="*50)

    for v in h:
        for w in v.get_connections():
            vid = v.get_id()
            wid = w.get_id()
            print('( %s , %s, %3d)' % (vid, wid, v.get_weight(w)))

    for v in h:
        print('g.vert_dict[%s]=%s' % (v.get_id(), h.vert_dict[v.get_id()]))
    '''
