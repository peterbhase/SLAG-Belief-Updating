from types import new_class
from pandas.core.indexing import IndexingError
import torch
import numpy as np
import pandas as pd
import argparse
import utils
import metrics
import time
import os
import jsonlines
from utils import str2bool, Report
from transformers import AutoModelForSeq2SeqLM, AutoModelWithLMHead, AutoModel, AutoTokenizer
from transformers import AdamW, get_scheduler, get_linear_schedule_with_warmup
from torch.optim import SGD, RMSprop
from models.probe import Probe
from models.learned_optimizer import ModelWithLearnedOptimizer
from copy import deepcopy
import networkx as nx
import matplotlib.pyplot as plt

def add_line_breaks_to_sentence(sentence, every_chars=20):
    words_list = sentence.split()
    new_sent = []
    char_counter = 0
    for word_num, word in enumerate(words_list):
        char_counter += len(word)
        if char_counter > every_chars and word_num < len(words_list) - 1: # not last word
            word += '\n'
            char_counter = 0
        new_sent.append(word)
    return ' '.join(new_sent)

def make_graph_df(args, data_loader, tokenizer):
    '''
    returns pandas dataframe with cols: id, in_ids, out_ids, text, label, pred, correct, update_pred
    - id is id of a unique vertex
    - in is list of ids of vertices with in-edges to the vertex
    - text is the text representation of the point
    - label is the label of that data point
    - pred is base model prediction
    - correct is binary correctness of base model prediction
    - update_pred - pred of updated model
    - update_correct - binary correctness of updated pred
    '''

    df_dict = {}
    n_data_points = 0
    for batch_num, batch in enumerate(data_loader):
        batch_size = batch['main_input_ids'].size(0)
        for point_num in range(batch_size):
            point_kwargs = utils.select_point_from_kwargs(batch, point_num)
            point = batch['text_data'][point_num]
            assert f'flipped_points' in point, "need to write flipped ids to file using --write_graph_to_file in main.py, or use fewer data points if these were written for not all points"
            orig_label = point_kwargs['orig_labels']
            # if T/F classification, make label_str into true/false
            if args.probing_style == 'model':
                label_str = 'true' if orig_label else 'false'
            else:
                label_str = orig_label
            name_sentence = add_line_breaks_to_sentence(point['proposition'], every_chars=20)
            df_dict[point['id']] = {
                'id' : point['id'],
                'in_ids' : None, # will make in-edges in another for loop next,
                'out_ids' : point[f'flipped_points'],
                'text' : point['proposition'],
                'label' : point['orig_label'],
                'print_str' : f"{name_sentence}\n[y: {label_str}]",
                'label_str' : label_str,
                'pred' : point[f'prediction'],
                'correct' : metrics.compute_acc_sum(args.probing_style, [point['prediction']], [orig_label], tokenizer),
                'update_pred' : point[f'update_pred'],
                'update_correct' : metrics.compute_acc_sum(args.probing_style, [point['update_pred']], [orig_label], tokenizer),
            }
            new_point = df_dict[point['id']]
        n_data_points += batch_size
    # get in_ids for point
    use_ids = np.arange(n_data_points)
    for i in use_ids:
        if i % 1000 == 0:
            print("points read: ", i)
        into_i_ids = []
        for j in use_ids:
            if i in df_dict[j]['out_ids']:
                into_i_ids.append(j)
        df_dict[i]['in_ids'] = into_i_ids
    df = pd.DataFrame.from_dict(df_dict, "index")
    return df

def nx_graph_from_pd_df(args, pd_df):
    # make a networkx graph object from a pandas dataframe with rows as nodes
    G = nx.DiGraph()
    # first add all rows as nodes
    eligible_ids = pd_df['id'] # don't allow connections to points not among the node ids in pd_df
    for row_num, row in pd_df.iterrows():
        G.add_node(row_num)
        row['in_ids'] = [in_id for in_id in row['in_ids'] if in_id in eligible_ids] 
        row['out_ids'] = [out_id for out_id in row['out_ids'] if out_id in eligible_ids]
        # plotting variables
        row['color'] = '#08af2140' if row['correct'] else '#ef2a3340'
        # lower alpha green: 
        # lower alpha red: 
        G.nodes[row_num].update(row.to_dict()) 
    # then add edges between nodes
    for row_num, row in pd_df.iterrows():
        node_id = row['id']
        out_ids = [out_id for out_id in row['out_ids'] if out_id in eligible_ids] 
        if len(out_ids) > 0:
            for out_id in out_ids:
                G.add_edge(node_id, out_id)
    return G

def proportion_transitive(graph):
    # out of all desendents who are children of children, what proportion are children themselves?
    num_children_of_children = 0
    num_connected = 0
    print("getting proportion transitive...")
    for node_num, node in enumerate(graph.nodes):
        if node_num % 10000 == 0:
            print(f"processed {node_num} children")
        children = graph.nodes[node]['out_ids']
        descendants_of_children = []
        for child in children:
            child_descendants = list(nx.descendants(graph, child))
            descendants_of_children.extend(child_descendants)
        descendant_is_child = [descendant in children for descendant in descendants_of_children]
        num_children_of_children += len(descendants_of_children)
        num_connected += sum(descendant_is_child)
    if num_children_of_children == 0:
        return -1
    else:
        return num_connected / num_children_of_children

def get_most_connected_node_subgraph(graph, max_distance=None):
    # returns subgraph of connected nodes to the most connected node in G, up to a distance of up_to_distance away from this node
    # most connected defined as sum of in and out edges
    in_and_out_edge_counts = [len(node['in_ids']) + len(node['out_ids']) for node_id, node in graph.nodes.data()]
    most_connected_idx = np.argmax(in_and_out_edge_counts)
    sub_graph = nx.generators.ego_graph(graph, most_connected_idx, undirected=True, radius=max_distance)
    return sub_graph

def get_highly_corrupting_points(graph):
    # returns list of tuples: node_id : # corrupted, AND list of (node, # change in acc sum)
    corrupting_list = []
    net_changed_list = []
    for node in graph.nodes:
        children = graph.nodes[node]['out_ids']
        children_corrupted = [graph.nodes[child]['correct']==1 for child in children] # originally was correct, must be incorrect after flipping
        children_fixed = [graph.nodes[child]['correct']==0 for child in children]
        num_corrupted = sum(children_corrupted)
        num_fixed = sum(children_fixed)
        corrupting_list.append( (node, num_corrupted) )
        net_changed_list.append( (node, num_corrupted - num_fixed) ) # will be negative if point is net helpful
    corrupting_list.sort(key = lambda x : x[1], reverse=False)
    net_changed_list.sort(key = lambda x : x[1], reverse=False)
    return corrupting_list, net_changed_list

def get_cycles_subgraphs(graph, top_k = 5):
    cycles = list(nx.simple_cycles(graph))
    sort_cycles = []
    for cycle in cycles:
        if len(cycle) > 2:
            sort_cycles.append((len(cycle), cycle))
    sort_cycles.sort(key = lambda x : x[0], reverse=True)
    cycles = [cycle for idx, cycle in sort_cycles]
    if top_k > 0:
        max_idx = min(len(cycles), top_k)
        cycles = cycles[:max_idx]
    sub_graphs = [nx.subgraph(graph, cycle_idx) for cycle_idx in cycles]
    return sub_graphs

def get_betweenness_subgraphs(graph, top_k=5, max_distance=None):
    betweenness_scores = nx.betweenness_centrality(graph)
    betweenness_scores = [(k,v) for k,v in betweenness_scores.items()]
    betweenness_scores.sort(key = lambda x : x[1], reverse=True)
    if top_k > 0:
        max_idx = min(len(betweenness_scores), top_k)
        betweenness_scores = betweenness_scores[:max_idx]
    join_graphs = []
    for node, val in betweenness_scores:
        new_graph = nx.generators.ego_graph(graph, node, undirected=True, radius=max_distance)
        new_graph.nodes[node]['color'] = 'yellow'
        join_graphs.append(new_graph)
    return join_graphs

def get_chains(graph, top_k=5):
    # find paths in graph which are linear chains, without connections between the nodes
    num_nodes = graph.number_of_nodes()
    chains = []
    seen_pairs = set()
    print("getting chains in graph...")
    for sample in range(10000):
        if sample % 100 == 0:
            print(f"processed {sample} points")
        nodes = np.random.choice(np.arange(num_nodes), size=2, replace=False).tolist()
        if str(nodes) not in seen_pairs:
            simple_paths = nx.all_simple_paths(graph, nodes[0], nodes[1])
            if simple_paths is not None: simple_paths = list(simple_paths)
            for path in simple_paths:
                if len(path) > 2:
                    chains.append(path)
        seen_pairs.add(str(nodes))
    # sort by length
    if len(chains) > 0:
        chains = [(chain, len(chain)) for chain in chains]
        chains.sort(key = lambda x : x[1], reverse=True)
        chains = [chain for chain, length in chains]
        if top_k > 0:
            max_idx = min(len(chains), top_k)
            chains = chains[:max_idx]
        chain_graphs = [nx.subgraph(graph, chain) for chain in chains]
    else:
        chain_graphs = []
    return chain_graphs

def get_longest_path_subgraph(graph, max_distance=1):
    cycles = list(nx.simple_cycles(graph))
    has_cycles = len(cycles) > 0
    # break all the cycles if graph has cycles
    if has_cycles:
        print("Returning graph with cycles pruned")
        raise(NotImplementedError)
    else:
        longest_path = nx.dag_longest_path(graph)
        join_graphs = []
        for node in longest_path:
            join_graphs.append(nx.generators.ego_graph(graph, node, undirected=True, radius=max_distance))
    composed_graphs = join_graphs[0]
    if len(join_graphs) > 1:
        for graph in join_graphs[1:]:
            composed_graphs = nx.compose(composed_graphs, graph)
        # middle_idx = np.floor(len(longest_path)/2)
        # middle_node = longest_path[middle_idx]
        # sub_graph = nx.generators.ego_graph(graph, middle_node, undirected=True, radius=max_distance)
    return composed_graphs
    
def plot_graph(args, graph, save_name=None, save_suffix=None, plot_type='neato'):
    '''
    plot graph given a pd graph df. save pdf in outputs
    plot_type is neato (spring-type layout) or dot (like a flow-chart)
    '''
    # make save_path
    if save_name is None:
        graph_name = f'{args.num_eval_points}'
    if save_suffix is not None:
        graph_name += f'_{save_suffix}'
    if save_name is None:
        graph_name += f'_{plot_type}'
    save_dir = os.path.join('outputs', args.experiment_name)
    save_path = os.path.join(save_dir, graph_name + '.pdf')
    if not os.path.exists(save_dir): os.mkdir(save_dir)
    '''
    filter some points here
    '''
    filter_data = [
        "The Beach was only directed by John Daly.",
        "Most of Ripon College's student body die on campus.",
    ]
    filtered_ids = [node[0] for node in list(graph.nodes.data()) if node[1]['text'] not in filter_data]
    graph = graph.subgraph(filtered_ids)
    pos = nx.nx_pydot.pydot_layout(graph, prog=plot_type)
    # scaling actually not necessary, just pad margins and decrease node_size and font_size
    # now rescale
    pos = {id : (x, y) for id, (x,y) in pos.items()}
    # increase margins by 1 to fit text
    minx = min([x for id, (x,y) in pos.items()])
    maxx = max([x for id, (x,y) in pos.items()])
    miny = min([y for id, (x,y) in pos.items()])
    maxy = max([y for id, (x,y) in pos.items()])
    margin = 90 if plot_type == 'neato' else 150
    minx -= margin
    miny -= margin
    maxx += margin
    maxy += margin
    # get labels and colors
    colors = [node['color'] for node_id, node in graph.nodes.data()]
    labels = {node_id : node['print_str'] for node_id, node in graph.nodes.data()}
    # make and draw figure
    fig = plt.figure()
    fig.set_size_inches(6, 6)
    plt.xlim([minx, maxx])
    plt.ylim([miny, maxy])
    min_distance = min([np.abs(x1-x2) + np.abs(y1-y2) for (x1,y1) in pos.values() for (x2,y2) in pos.values() if not ((x1==x2) and (y1==y2))])
    size_scale = 1
    size_kwargs = {
        'node_size' : 1000 / size_scale,
        'arrowsize' : max(10 / size_scale, 1),
        'width' : max(1 / size_scale, .05),
        'font_size': 6 / size_scale,
    }
    # size_scale = .8 + (30-1)/(148-0)*(graph.number_of_nodes() - 0)
    # size_kwargs = {
    #     'node_size' : 140 / size_scale,
    #     'arrowsize' : max(8 / size_scale, 1),
    #     'width' : max(.5 / size_scale, .05),
    #     'font_size': 4 / size_scale,
    # }
    nx.draw_networkx(graph, pos, labels=labels, node_color=colors, edge_color='#00000080', font_weight='bold', with_labels='true', **size_kwargs)
    fig.tight_layout()
    fig.show()
    fig.savefig(save_path, format="PDF")

def print_graph_summary(args, graph):
    '''
    print a bunch of summary statistics of a graph
    '''
    print("Summary statistics for belief graph: ")
    # import pdb; pdb.set_trace()
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    in_edge_counts = sorted([len(node['in_ids']) for node_id, node in graph.nodes.data()])
    out_edge_counts = sorted([len(node['out_ids']) for node_id, node in graph.nodes.data()])
    in_and_out_edge_counts = [len(node['in_ids']) + len(node['out_ids']) for node_id, node in graph.nodes.data()]
    num_strongly_connected_components = len(list(nx.strongly_connected_components(graph)))
    # get proportion transitive
    if sum(in_and_out_edge_counts) < 1e7:
        prop_transitive = proportion_transitive(graph)
    else:
        prop_transitive = -1
    # weakly connected stats
    weakly_connected_components = list(nx.weakly_connected_components(graph))
    weakly_connected_components = [(len(component), component) for component in weakly_connected_components]
    weakly_connected_components.sort(key=lambda x : x[0], reverse=False)
    weakly_connected_component_sizes = [num_nodes for num_nodes, component in weakly_connected_components]
    num_weakly_connected_components = len(weakly_connected_component_sizes)
    # get proportion atomic and distribution 
    prop_atomic_nodes = sum(np.array(in_and_out_edge_counts) == 0) / n_nodes
    quantiles = [.05, .25, .5, .75, .95]
    in_edge_quantiles = np.quantile(in_edge_counts, quantiles)
    out_edge_quantiles = np.quantile(out_edge_counts, quantiles)
    cc_quantiles = np.quantile(weakly_connected_component_sizes, quantiles)
    # get corrupting points
    corrupting_points, influential_points = get_highly_corrupting_points(graph)
    corrupt_quantiles = np.quantile([point[1] for point in corrupting_points], quantiles) 
    net_corrupt_quantiles = np.quantile([point[1] for point in influential_points], quantiles) 
    top_k = 10
    print(f"Number of nodes: {n_nodes}")
    # print(f"Number of edges: {n_edges}")
    print(f"Number of in-edges: {sum(in_edge_counts)}")
    print(f"Number of out-edges: {sum(out_edge_counts)}")
    print(f"Prop. atomic nodes: {100*prop_atomic_nodes:.2f}")
    print(f"Prop. transitive: {100*prop_transitive:.2f}")
    print(f"num_weakly_connected_components: {num_weakly_connected_components}")
    print(f"num_strongly_connected_components: {num_strongly_connected_components}")
    print(f"top weakly_connected_components: {[x for x in weakly_connected_component_sizes[-top_k:] if x > 1]}")
    print(f"weakly_connected_components distribution: { [f'{quantile} : {quantity:.1f}' for quantile, quantity in zip(quantiles, cc_quantiles)] }")
    print(f"top corrupting points: {[(id, count) for id, count in corrupting_points[-top_k:]]}")
    print(f"top net-corrupting points: {[(id, count) for id, count in influential_points[-top_k:]]}")
    print(f"corrupting distribution: { [f'{quantile} : {quantity:.1f}' for quantile, quantity in zip(quantiles, corrupt_quantiles)] }")
    print(f"net-corrupting distribution: { [f'{quantile} : {quantity:.1f}' for quantile, quantity in zip(quantiles, net_corrupt_quantiles)] }")
    # print(f"Prop. bidirectional edges: {sum(in_and_out_edge)/n_edges :.2f}")
    print(f"Top in-edge counts: ", in_edge_counts[-top_k:])
    print(f"Top out-edge counts: ", out_edge_counts[-top_k:])
    print(f"In-edge distribution: { [f'{quantile} : {quantity:.1f}' for quantile, quantity in zip(quantiles, in_edge_quantiles)] }")
    print(f"Out-edge distribution: { [f'{quantile} : {quantity:.1f}' for quantile, quantity in zip(quantiles, out_edge_quantiles)] }")
    return