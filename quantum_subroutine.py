import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from collections import defaultdict

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit_optimization.applications import Maxcut
from qiskit_algorithms import SamplingVQE
from qiskit_algorithms.optimizers import NFT
from qiskit.primitives import Sampler


def data_loader(input_data):
    data = input_data["data"]
    input_data = data

    num_assets = input_data['num_assets']
    assets = input_data['assets']
    df = pd.DataFrame()
    series = []
    for asset in assets:
        series.append(pd.Series(assets[asset]['history'], name=asset))
    df = pd.concat(series, axis=1)
    df_cleaned = df
    return df_cleaned, assets


def level_average(df, assets, level):
    level_assets = defaultdict(list)
    level_info = {}

    for ticker, asset_data in assets.items():
        level_value = asset_data.get(level, "Unknown")
        level_assets[level_value].append(ticker)
        level_info[ticker] = asset_data

    level_dataframes = {}
    for level_value, tickers in level_assets.items():
        level_series = []
        for ticker in tickers:
            history = level_info[ticker].get("history", {})
            series = pd.Series(history, name=ticker)
            level_series.append(series)
        df_level = pd.concat(level_series, axis=1).dropna(axis=1)
        level_dataframes[level_value] = df_level

    level_averages = {}
    for level_value, df_level in level_dataframes.items():
        df_level.index = pd.to_datetime(df_level.index)
        level_avg = df_level.mean(axis=1)
        level_averages[level_value] = level_avg

    return level_averages


def correlation_in_period(df, assets, level, start_date, end_date):
    avg_df = pd.DataFrame(level_average(df, assets, level))
    avg_df.index = pd.to_datetime(avg_df.index)
    mask = (avg_df.index >= pd.to_datetime(start_date)) & (avg_df.index <= pd.to_datetime(end_date))
    period_corr = 1 / avg_df.loc[mask].corr()
    return period_corr


def run_vqe_maxcut(G):
    w = nx.to_numpy_array(G)
    max_cut = Maxcut(w)
    qp = max_cut.to_quadratic_program()
    qubitOp, offset = qp.to_ising()

    num_qubits = qubitOp.num_qubits

    hadamards = QuantumCircuit(num_qubits)
    hadamards.h(range(num_qubits))

    two_local = TwoLocal(num_qubits, "ry", "cz", reps=3, entanglement="full")
    ansatz = hadamards.compose(two_local)

    optimizer = NFT(maxiter=600)
    vqe = SamplingVQE(sampler=Sampler(), ansatz=ansatz, optimizer=optimizer)
    result = vqe.compute_minimum_eigenvalue(qubitOp)

    x = max_cut.sample_most_likely(result.eigenstate)
    return x, offset, result


def preprocessing(df, assets, start_date, end_date):
    sector_averages = level_average(df, assets, 'sector')
    avg_df = pd.DataFrame(sector_averages)
    avg_df.index = pd.to_datetime(avg_df.index)

    full_corr = correlation_in_period(df, assets, 'sector', start_date, end_date)

    corr_matrix = full_corr.copy()

    G = nx.Graph()
    sectors = corr_matrix.columns.tolist()
    G.add_nodes_from(sectors)

    edges = []
    for i in range(len(sectors)):
        for j in range(i + 1, len(sectors)):
            sector_i = sectors[i]
            sector_j = sectors[j]
            weight = round(corr_matrix.loc[sector_i, sector_j], 2)
            edges.append((sector_i, sector_j, weight))

    G.add_weighted_edges_from(edges)

    return G


def recursive_maxcut_partition(G, depth=1, level=0, path="0"):
    if len(G.nodes) <= 1:
        return []

    x, offset, result = run_vqe_maxcut(G)
    partition = {node: x[i] for i, node in enumerate(G.nodes())}

    nodes_0 = [node for node in G.nodes() if partition[node] == 0]
    nodes_1 = [node for node in G.nodes() if partition[node] == 1]

    G0 = G.subgraph(nodes_0).copy()
    G1 = G.subgraph(nodes_1).copy()

    if depth == 0:
        return [G0, G1]
    else:
        return (
            recursive_maxcut_partition(G0, depth=depth - 1, level=level + 1, path=f"{path}.0")
            + recursive_maxcut_partition(G1, depth=depth - 1, level=level + 1, path=f"{path}.1")
        )
