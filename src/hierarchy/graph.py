from __future__ import annotations

import os
from collections import deque
from itertools import count
from typing import Iterable, Sequence, Union

import networkx as nx
import pandas as pd


class EnronGraphBuilder:
    """
    Parses Kaggle's enron.csv (or synthetic roles) into a directional policy tree.
    Edges represent commands flowing from higher Authority (CEO) to lower.
    """

    def __init__(self, depth_limit: int = 3):
        if depth_limit < 1:
            raise ValueError("depth_limit must be at least 1.")
        self.graph = nx.DiGraph()
        self.depth_limit = depth_limit
        self._node_counter = count(1)

    def _reset_graph(self) -> None:
        self.graph = nx.DiGraph()
        self._node_counter = count(1)

    def _risk_aversion_for_level(self, level: int) -> float:
        return min(0.1 + (0.2 * level), 0.95)

    def _branch_factor_for_level(self, branching_factors: Union[int, Sequence[int]], level: int) -> int:
        if isinstance(branching_factors, int):
            return branching_factors
        if not branching_factors:
            return 0
        index = min(level - 1, len(branching_factors) - 1)
        return int(branching_factors[index])

    def _validate_edge_columns(self, columns: Iterable[str], parent_col: str, child_col: str) -> None:
        missing = {parent_col, child_col} - set(columns)
        if missing:
            raise ValueError(
                f"Dataframe must include columns {parent_col!r} and {child_col!r}; missing {sorted(missing)}."
            )

    def _iter_edge_frames(
        self,
        df: Union[pd.DataFrame, str, os.PathLike, Iterable[pd.DataFrame]],
        parent_col: str,
        child_col: str,
        chunksize: int,
        read_csv_kwargs: dict | None,
    ) -> Iterable[pd.DataFrame]:
        if isinstance(df, pd.DataFrame):
            self._validate_edge_columns(df.columns, parent_col, child_col)
            yield df[[parent_col, child_col]]
            return

        if isinstance(df, (str, os.PathLike)):
            csv_kwargs = dict(read_csv_kwargs or {})
            csv_kwargs.setdefault("usecols", [parent_col, child_col])
            csv_kwargs.setdefault("chunksize", chunksize)
            csv_kwargs.setdefault("dtype", {parent_col: "string", child_col: "string"})

            try:
                reader = pd.read_csv(df, **csv_kwargs)
            except ValueError as exc:
                raise ValueError(
                    f"CSV source must expose columns {parent_col!r} and {child_col!r} for streaming ingestion."
                ) from exc

            for chunk in reader:
                self._validate_edge_columns(chunk.columns, parent_col, child_col)
                yield chunk[[parent_col, child_col]]
            return

        iterator = iter(df)
        saw_chunk = False
        for chunk in iterator:
            if not isinstance(chunk, pd.DataFrame):
                raise TypeError("Chunk iterators passed to build_from_dataframe must yield pandas DataFrames.")
            self._validate_edge_columns(chunk.columns, parent_col, child_col)
            saw_chunk = True
            yield chunk[[parent_col, child_col]]

        if not saw_chunk:
            raise ValueError("Chunk iterator did not yield any dataframes.")

    def _add_edges_from_frame(self, frame: pd.DataFrame, parent_col: str, child_col: str) -> int:
        if frame.empty:
            return 0

        edge_frame = frame[[parent_col, child_col]].dropna(subset=[parent_col, child_col])
        if edge_frame.empty:
            return 0

        parents = edge_frame[parent_col].astype("string").str.strip()
        children = edge_frame[child_col].astype("string").str.strip()
        mask = parents.notna() & children.notna() & parents.ne("") & children.ne("")
        if not mask.any():
            return 0

        self.graph.add_edges_from(zip(parents[mask].astype(str), children[mask].astype(str)))
        return int(mask.sum())

    def build_balanced_hierarchy(
        self,
        branching_factors: Union[int, Sequence[int]] = 2,
        root_name: str = "C-Suite",
    ) -> nx.DiGraph:
        """
        Builds a recursive tree of arbitrary depth_limit using a fixed or per-level
        branching factor specification.
        """
        self._reset_graph()
        self.graph.add_node(root_name, level=0, risk_aversion=self._risk_aversion_for_level(0))

        def add_children(parent: str, level: int) -> None:
            if level >= self.depth_limit:
                return

            branch_factor = self._branch_factor_for_level(branching_factors, level)
            if branch_factor <= 0:
                return

            for _ in range(branch_factor):
                node_name = f"L{level}_{next(self._node_counter)}"
                self.graph.add_node(
                    node_name,
                    level=level,
                    risk_aversion=self._risk_aversion_for_level(level),
                )
                self.graph.add_edge(parent, node_name)
                add_children(node_name, level + 1)

        add_children(root_name, 1)
        return self.graph

    def build_mock_hierarchy(self) -> nx.DiGraph:
        """
        Builds a recursive mock hierarchy. depth_limit controls total tiers and
        can exceed 3 without code changes.
        """
        branching_factors = [2] + ([2] * max(0, self.depth_limit - 2))
        return self.build_balanced_hierarchy(branching_factors=branching_factors, root_name="C-Suite")

    def _assign_levels_from_roots(self, roots: list[str]) -> None:
        queue = deque((root, 0) for root in roots)
        best_level: dict[str, int] = {}

        while queue:
            node, level = queue.popleft()
            prior_level = best_level.get(node)
            if prior_level is not None and prior_level <= level:
                continue

            best_level[node] = level
            self.graph.nodes[node]["level"] = level
            self.graph.nodes[node]["risk_aversion"] = self._risk_aversion_for_level(level)

            for child in self.graph.successors(node):
                queue.append((child, level + 1))

    def build_from_dataframe(
        self,
        df: Union[pd.DataFrame, str, os.PathLike, Iterable[pd.DataFrame]],
        parent_col: str = "parent",
        child_col: str = "child",
        chunksize: int = 50000,
        read_csv_kwargs: dict | None = None,
    ) -> nx.DiGraph:
        """
        Builds a directed hierarchy from explicit parent->child relationships.

        Accepts a concrete dataframe, a CSV path, or a dataframe chunk iterator.
        CSV sources are streamed through pandas with chunksize=50000 by default so
        large edge lists do not need to be materialized in memory.
        """
        self._reset_graph()

        valid_edge_rows = 0
        for chunk in self._iter_edge_frames(
            df=df,
            parent_col=parent_col,
            child_col=child_col,
            chunksize=chunksize,
            read_csv_kwargs=read_csv_kwargs,
        ):
            valid_edge_rows += self._add_edges_from_frame(chunk, parent_col, child_col)

        if valid_edge_rows == 0 or self.graph.number_of_edges() == 0:
            raise ValueError("No valid edges were extracted from the dataframe source.")

        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("Hierarchy extraction requires an acyclic directed graph.")

        roots = [node for node, indegree in self.graph.in_degree() if indegree == 0]
        if not roots:
            raise ValueError("Hierarchy extraction failed because no root nodes were found.")

        self._assign_levels_from_roots(roots)
        return self.graph
