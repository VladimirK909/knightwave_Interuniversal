# Knightwave Interuniversal

This repository contains an implementation of a hybrid quantum-classical portfolio optimization algorithm, which includes a quantum asset allocation routine and classical Markovitz portfolio weighting. 

Our model treats the assets as nodes and correlations as edges, so that the diversification problem is mapped as a Max-Cut problem. This hybrid quantum-classical algorithm is capable of efficiently handling large datasets thanks to a linear scaling in the number of qubits with problem size.


## Table of Contents
- [Acknowledgements](#acknowledgements)
- [Overview](#overview)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Usage](#usage)
    - [Training a New Model](#training-a-new-model)
    - [Visually evaluating a Pre-Trained Model](#visually-evaluating-a-pre-trained-model)
    - [Analytically evaluating a Pre-Trained Model](#analytically-evaluating-a-pre-trained-model)
- [Results](#results)

## Acknowledgements
The project showcased in this repository is part of ETH Quantum Hackathon 2025. Challenge was introduced by Alberto and Antonio, from QCentroid. 

The resources and services used in this work were provided by the challenge authors and evaluation was performed on QCentroid's cloud platform.

## Overview
Portfolio Optimization is a widespread problem which has been studied for years and good classical solutions that allow for the asset allocation and the prediction of the portfolio’s returns have been developed. However, there are limitations in the number of assets they can manage before the computational time grows beyond practical applicability. To tackle this problem without renouncing the good performance offered by the existing classical algorithms, we have decided to focus on the asset selection. This way, we can decrease the number of assets we feed into the Markowitz Model by shrinking the large initial dataset to a limited number of assets by leveraging quantum computation. The idea is to preselect the most promising stocks by firstly subdividing them into lowly correlated clusters, which is a classically NP Hard problem, by mapping it to a Max-Cut problem and solving it with Qiskit. After that, we pick the most promising stocks in every cluster according to a chosen metric, which in our case is going to be the Sortino Ratio, as we want to benefit those scenarios with higher returns. 

## Features
When designing our solution, we had more than just portfolio growth in mind. In the real world, financial portfolios have a *specific purpose* (usually by balancing between risk mitigation and growth potential). Taking this into account, we wanted to make sure that our algorithm is **scalable** both in terms of **increased number of qubits and bigger input data**, but also in terms of **different use case and customized added value** with regards to the portfolio's purpose and financial risk strategies.
In other words, our algorithm is not designed to replace a portfolio manager in the real world, but rather to *add value by enhancing their already existing wokrflow*.
In order to ensure this, we chose the following features as must-haves for our algorithm, and we designed it with them in mind:
- **Reliability**: We use a well-known and deeply tested classical portfolio optimization strategy - the **Modern Portfolio Optimization Theory** by Markowitz based on the **Sharpe ratio**.
- **Modularity**: Parameters, metrics, and risk modelling strategies can be **changed in a straightforward and easily implementable** way. We find this very important as the client might not be as well versed in coding as we are and so we need to make it as user friendly as possible [*].
- **Complementary**: As the quantum subroutine focuses on Asset selection, different strategies can be chosen when optimizing portfolio weighting which would make the algorithm more aggressive or more safe.

[*] : *An example of this would be constructing the fully connected asset correlation graph with some other measure for diversification risk. In our implementation correlation risk is modelled using correlation between assets, but if a client would want to use something else they can just change the definition of the graph and the quantum subroutine will work completely seamlessly with zero further editing.*
## Getting Started

### Prerequisites
To use this repository, ensure you have the following:
- Python 3.11 or later

### Installation
1. *Clone the repository*:
```bash
   git clone https://github.com/VladimirK909/knightwave_Interuniversal.git
   cd knightwave_interuniversal
   ```
3. *Install the required dependencies*:

    When installing the dependencies, you can try running the following command:
   ```bash
   pip install -r requirements.txt
   ```
   What is very important is to use the qiskit version specified in the `requirements.txt`. Newer qiskit versions work differently and the quantum subroutine will not run.

## Repository Structure

.
├── INTEGRATED.ipynb # Pre-run notebook with visualizations and explanations of solution.
├── main.py              
├── portfolio.py                
├── quantum_subroutine.py # The quantum module
├── asset_screening.py # The main classical module
├── requirements.txt # Python dependencies
└── README.md # Project documentation



# Quantum-Assisted Portfolio Diversification

This project explores a hybrid classical-quantum pipeline for constructing diversified financial portfolios. The approach leverages quantum algorithms to cluster assets based on correlations and ultimately optimize portfolio performance using classical techniques.

## Overview

The code is organized into four main components:

---

## 1. Classical Pre-Clustering Using Metadata

Due to current QPU (Quantum Processing Unit) hardware limitations—especially the number of available qubits—and the high cost of simulating quantum systems, it's necessary to reduce the problem size before applying quantum methods.

### Implemented Approach:
- *Heuristic Grouping by Sector*: We group stocks by sector using an ad hoc, zero-cost heuristic. This approach assumes that:
  - Stocks within the same sector are *highly correlated*.
  - Stocks across different sectors exhibit *relatively lower correlation*.
- For each sector, we compute the *average behavior* of its stocks.
- We then compute a *correlation matrix* between these sector-level averages.
- This results in a *reduced graph*, where each node represents a sector, decreasing the number of nodes and making the quantum step more tractable.

> This is a practical workaround for current hardware constraints. As QPUs scale, we expect to remove this step and operate directly on full asset sets.

---

## 2. Quantum Clustering Subroutine

The main quantum component involves identifying clusters of weakly correlated assets using *quantum combinatorial optimization*.

### Why Quantum?
- Asset correlations are modeled as a *graph*:
  - *Nodes* = individual assets.
  - *Edges* = pairwise correlations.
- We apply a *Max-Cut algorithm* (implemented via Variational Quantum Eigensolver, VQE) to partition this graph into clusters.
- The Max-Cut problem is *NP-hard, making it well-suited to explore potential **quantum advantage*.
- Quantum optimization benefits from *linear scaling* with the number of qubits, offering a compelling case as quantum hardware improves.

### Other Quantum Approaches Considered:
- *Quantum Machine Learning (QML): Dismissed, as current QML techniques have been shown to be **classically simulatable* and unlikely to offer real quantum advantage at this stage.
- *HHL Algorithm: Considered but deemed **impractical* due to noise sensitivity and strict constraints.
- *Quantum Optimization for Weighting*: A VQE-based approach to portfolio weighting is possible, but classical algorithms are currently more robust and efficient for this task.

---

## 3. Asset Selection via Sortino Ratio

From each cluster, we select the most promising asset(s) based on the *Sortino Ratio, which focuses on **downside volatility*.

### Justification:
- Our clustering already introduces *risk aversion* by design (low intra-cluster correlation).
- Therefore, a more *aggressive selection criterion* is needed to meet the objective of *maximizing one-day returns*.
- The Sortino Ratio rewards upside volatility and penalizes downside, making it a better fit than Sharpe Ratio in this context.

### Flexibility:
- We may select *more than one asset per cluster, depending on the **target portfolio size*.
- The selection metric (e.g., Sortino, Sharpe, Calmar) is modular and can be adapted to match *different investor profiles* and *objectives*.

> In real-world applications, this aggressiveness can be tuned to better fit conservative or long-term strategies.

### Future Work:
- Implementing a *Quantum Monte Carlo* approach to estimate downside variance could improve scalability and efficiency on larger datasets.

---

## 4. Portfolio Optimization with Markowitz Model

Once assets are selected, we assign weights using the *classical Markowitz Mean-Variance Optimization* model.

### Why Classical?
- The Markowitz model is *simple, interpretable, and robust*.
- Although a quantum version could be explored (e.g., using QAOA or VQE), current *classical solvers* are mature and perform effectively.
- The quantum justification for this step is weaker compared to clustering.

> Future directions may involve experimenting with *Quantum Markowitz Models* as quantum techniques in optimization evolve.

---

## Summary

This pipeline combines classical and quantum techniques to:
- Reduce problem complexity via heuristic clustering.
- Leverage quantum optimization for diversified asset grouping.
- Apply targeted, risk-aware asset selection.
- Optimize portfolio weights using classical methods.

The architecture is modular, adaptable, and future-proof, allowing easy integration of new quantum tools and strategies as the technology matures.

## Future Prospects
Given the time constraints of the hackathon, there were some ideas that we could not implement, however we would like to mention them here as they would be relatively easy to implement due to the modularity of our solution.

Firstly, as mentioned above, by having more qubits our quantum subroutine would work better by taking the max-cut of a fully-connected correlation graph of all of the assets.
This would give better results less biased by the sector or industry grouping.

Secondly, we could use Quantum Monte Carlo for the calculation of the Sortino ratio. In this calculation, the denominator consists of the downside deviation. This metric is usually calculated in linear time, but in the real world cross-talk between different stock signals and their derivatives might also be taken into account which would make it NP-hard. This means that instead of calculating the downside deviation for every data point of every stock (like we do here) we could instead sample the distribution using Quantum Monte Carlo. This would make the algorithm more compatible with real world conditions as well as big data implementations.

Lastly, the quantum subroutine is simulated in this solution, but in the real world it would be run with shots. Because of this, and also taking into consideration the noise of a real quantum machine, it is possible that the VQE for the max-cut does not converge well and the eigenstate distribution is still not converged to the correct value. We can however use this to our advantage by taking the first three most likely solutions from the shots histogram and evaluate the value of the max-cut for each of them and take the best one - thereby leveraging the inherent stochasticity of the quantum computer to deal with the noisy, and often unpredictable nature of financial systems.

All of this is possible because of our modularity. We built our solution with the aim of adding value to a real world client in a creative way, and these principles ensure that given more time this could be shown beyond the proof of concept framework of this hackathon and give decent results in the real world as well.
