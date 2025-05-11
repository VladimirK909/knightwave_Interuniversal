from qiskit_aer.primitives import SamplerV2
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import pandas as pd
import networkx as nx
import numpy as np
import pypfopt
import asset_screening as s
import quantum_subroutine as j


def run(input_data):
    # Load the data (assets is an object required for further processing)
    df, assets = j.data_loader(input_data)

    # We take the entire dataset into consideration in this run.
    # This can be tweaked by changing the following two bounds:
    start_date = df.index.min()
    end_date = df.index.max()

    # We obtain the maximally connected graph using the `preprocessing` function
    # from the jimmy module.
    G = j.preprocessing(df, assets, start_date, end_date)

    B = j.recursive_maxcut_partition(G, depth=1, level=0, path="0")

    asset_lists = [[] for _ in range(len(B))]  # Initializing an empty list with 4 empty lists inside

    for i, graph in enumerate(B):
        sectors_in_graph = list(graph.nodes)
        for asset, data in assets.items():
            if data.get('sector') in sectors_in_graph:
                asset_lists[i].append(asset)  # Adding the asset to the corresponding list within asset_lists

    top_assets_all = []

    for subgraph in asset_lists:
        df_subgraph = df[subgraph]

        # Drop columns with all NaNs (to avoid issues)
        df_subgraph = df_subgraph.dropna(axis=1, how='all')

        if df_subgraph.empty:
            continue  # skip if no valid data

        # Compute daily returns
        returns_df = pypfopt.expected_returns.returns_from_prices(df_subgraph)

        if returns_df.empty or returns_df.shape[1] == 0:
            continue

        # Compute Sortino ratios
        ratios = s.calculate_sortino_ratio(returns_df)

        # Sort and get top 10 assets in this subgraph
        top_10 = ratios.sort_values(ascending=False).head(10).index.tolist()

        top_assets_all.extend(top_10)

    df_selected = df[top_assets_all]

    from pypfopt.expected_returns import mean_historical_return
    from pypfopt.risk_models import CovarianceShrinkage

    mu = mean_historical_return(df_selected)
    S = CovarianceShrinkage(df_selected).ledoit_wolf()


    from pypfopt.efficient_frontier import EfficientFrontier
    from pypfopt import objective_functions

    ef = EfficientFrontier(mu, S, verbose=True, solver="ECOS")
    ef.add_objective(objective_functions.L2_reg, gamma=0.1)
    w = ef.max_sharpe()

    W = ef.clean_weights()
    output = {
        'selected_assets_weights': dict(W),
        'num_selected_assets': len(W)
    }
    return output
