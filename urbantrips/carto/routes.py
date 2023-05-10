import networkx as nx
from osmnx import distance


def create_branch_graph(branch_stops):
    metadata = {
        "crs": "epsg:4326",
        "id_linea": branch_stops['id_linea'].unique().item(),
        "id_ramal": branch_stops['id_ramal'].unique().item()
    }
    G = nx.MultiGraph(**metadata)

    nodes_ramal = branch_stops.sort_values(
        'order').reindex(columns=['node_id', 'x', 'y'])
    nodes = [(int(row['node_id']), {'x': row['x'], 'y':row['y']})
             for _, row in nodes_ramal.iterrows()]
    G.add_nodes_from(nodes)

    edges_from = nodes_ramal['node_id'].iloc[:-1].map(int)
    edges_to = nodes_ramal['node_id'].shift(-1).iloc[:-1].map(int)
    edges = [(i, j, 0) for i, j in zip(edges_from, edges_to)]
    G.add_edges_from(edges)

    # add distance in meters
    G = distance.add_edge_lengths(G)

    return G


def create_branch_g_from_stops_df(stops, id_ramal):
    branch_stops = stops.loc[stops.id_ramal == id_ramal, :]
    G = create_branch_graph(branch_stops)
    return G
