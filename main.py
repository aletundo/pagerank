#!/usr/bin/env python3

import operator
import argparse
import logging
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
LOGGER = logging.getLogger(__name__)


def init_argument_parser():
    parser = argparse.ArgumentParser(description='PageRank')

    parser.add_argument('--nodes', default=25, type=int,
                        help='The number of nodes of the generated graph')
    parser.add_argument('--edge_probability', default=0.125, type=float,
                        help='The edge probability of the generated graph')
    parser.add_argument('--seed', default=100, type=int,
                        help='The random seed used during the graph generation and for plots layout')
    parser.add_argument('--dumping_factor', default=0.85, type=float,
                        help='The dumping factor used by the algorithm')
    parser.add_argument('--iterations', default=100, type=int,
                        help='The number of iterations of the algorithm')
    return parser


def init_hyperlink_matrix(graph):
    nodes_list = list(graph.nodes)
    nodes_num = len(nodes_list)
    H = np.zeros([nodes_num, nodes_num])

    for n, ngbrs in graph.adj.items():
        ngbrs_num = len(list(ngbrs.items()))
        for ngbr in ngbrs.items():
            H[ngbr[0], n] = 1 / ngbrs_num
    return H


def apply_power_method(H, iterations):
    LOGGER.info("Initializing importance vector with random values")
    I = np.random.rand(H.shape[0])
    I = I / np.linalg.norm(I)
    initial_eigenvalue = I.T @ H @ I / (I.T @ I)

    LOGGER.info("Initial I = {}".format(initial_eigenvalue))

    old_eigenvalue = initial_eigenvalue

    LOGGER.info("Starting computation ({} iterations)".format(iterations))
    for i in range(iterations):
        I = np.dot(H, I)
        I = I / np.linalg.norm(I)
        eigenvalue = I.T @ H @ I / (I.T @ I)

        error = abs(eigenvalue - old_eigenvalue)
        LOGGER.info("Iteration {}, new eigenvalue = {}, error = {}".format(i, eigenvalue, error))

        old_eigenvalue = eigenvalue
    return I


def init_matrix(graph, dumping_factor):
    LOGGER.info("Initializing the hyperlink matrix")
    H = init_hyperlink_matrix(graph)

    LOGGER.info("Transforming H into a stochastic matrix")
    col_sum = np.sum(H, axis=0)
    for s in range(col_sum.shape[0]):
        if col_sum[s] != 0.:
            continue
        H[:, s] = 1 / float(H.shape[0])

    LOGGER.info("Applying dumping factor")
    M = dumping_factor * H + (1 - dumping_factor) / H.shape[0]
    return M


def compare_algorithm(graph, dumping_factor, iterations, I, seed):
    LOGGER.info("Comparing naive implementation with NetworkX PageRank")
    pagerank = nx.pagerank(graph, alpha=dumping_factor, max_iter=iterations)
    I_networkx = [pagerank[key] for key in pagerank]

    I = I.tolist()
    naive_pagerank = {}
    for i in range(len(I)):
        naive_pagerank[i] = I[i]

    sorted_networkx_pagerank = sorted(pagerank.items(), key=operator.itemgetter(1))
    sorted_naive_pagerank = sorted(naive_pagerank.items(), key=operator.itemgetter(1))

    LOGGER.info("Naive PageRank results:")
    LOGGER.info([page[0] for page in sorted_naive_pagerank])

    LOGGER.info("NetworkX PageRank results:")
    LOGGER.info([page[0] for page in sorted_networkx_pagerank])

    if graph.number_of_nodes() <= 50:
        show_plots(graph, I, I_networkx, seed)


def show_plots(G, I, I_networkx, seed):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title("Naive implementation")
    nx.draw(G, ax=ax1, cmap=plt.get_cmap('Blues'), node_color=I, with_labels=True, font_weight='bold',
            pos=nx.spring_layout(G, seed=seed))
    ax2.set_title("NetworkX")
    nx.draw(G, ax=ax2, cmap=plt.get_cmap('Blues'), node_color=I_networkx, with_labels=True, font_weight='bold',
            pos=nx.spring_layout(G, seed=seed))
    plt.show()


def main():
    parser = init_argument_parser()
    args = parser.parse_args()

    LOGGER.info(
        "Creating a directed random graph with {} nodes and edge probability = {}".format(args.nodes, args.edge_probability))
    graph = nx.fast_gnp_random_graph(args.nodes, args.edge_probability, args.seed, directed=True)

    M = init_matrix(graph, args.dumping_factor)

    I = apply_power_method(M, args.iterations)

    compare_algorithm(graph, args.dumping_factor, args.iterations, I, args.seed)


if __name__ == '__main__':
    main()
