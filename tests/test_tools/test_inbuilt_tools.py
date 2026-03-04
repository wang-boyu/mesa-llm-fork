from __future__ import annotations

from types import SimpleNamespace

import pytest
from mesa.discrete_space import OrthogonalMooreGrid
from mesa.space import ContinuousSpace, MultiGrid, SingleGrid

from mesa_llm.tools.inbuilt_tools import (
    move_one_step,
    speak_to,
    teleport_to_location,
)


class DummyModel:
    def __init__(self):
        self.grid = None
        self.space = None
        self.agents = []


class DummyAgent:
    def __init__(self, unique_id: int, model: DummyModel):
        self.unique_id = unique_id
        self.model = model
        self.pos = None


def test_move_one_step_on_singlegrid():
    model = DummyModel()
    model.grid = SingleGrid(width=5, height=5, torus=False)

    agent = DummyAgent(unique_id=1, model=model)
    model.agents.append(agent)
    model.grid.place_agent(agent, (2, 2))

    result = move_one_step(agent, "North")

    assert agent.pos == (2, 3)
    assert result == "agent 1 moved to (2, 3)."


def test_teleport_to_location_on_multigrid():
    model = DummyModel()
    model.grid = MultiGrid(width=4, height=4, torus=False)

    agent = DummyAgent(unique_id=7, model=model)
    model.agents.append(agent)
    model.grid.place_agent(agent, (0, 0))

    out = teleport_to_location(agent, [3, 2])

    assert agent.pos == (3, 2)
    assert out == "agent 7 moved to (3, 2)."


def test_teleport_to_location_on_orthogonal_grid_without_constructor():
    # Create an instance of a subclass of OrthogonalMooreGrid without invoking its __init__
    class _DummyOrthogonalGrid(OrthogonalMooreGrid):
        pass

    orth_grid = object.__new__(_DummyOrthogonalGrid)
    target = (1, 1)
    dummy_cell = SimpleNamespace(coordinate=target, agents=[])
    orth_grid._cells = {target: dummy_cell}

    model = DummyModel()
    model.grid = orth_grid

    agent = DummyAgent(unique_id=9, model=model)
    model.agents.append(agent)

    out = teleport_to_location(agent, [1, 1])

    assert getattr(agent, "cell", None) is dummy_cell
    assert out == "agent 9 moved to (1, 1)."


def test_move_one_step_on_orthogonal_grid_without_constructor():
    class _DummyOrthogonalGrid(OrthogonalMooreGrid):
        pass

    orth_grid = object.__new__(_DummyOrthogonalGrid)
    start_target = (1, 1)
    # mesa.discrete_space grids use (row, col), so North decrements row.
    end_target = (0, 1)
    start_cell = SimpleNamespace(coordinate=start_target, agents=[])
    end_cell = SimpleNamespace(coordinate=end_target, agents=[])
    orth_grid._cells = {start_target: start_cell, end_target: end_cell}

    model = DummyModel()
    model.grid = orth_grid

    agent = DummyAgent(unique_id=10, model=model)
    agent.cell = start_cell
    model.agents.append(agent)

    out = move_one_step(agent, "North")

    assert getattr(agent, "cell", None) is end_cell
    assert out == "agent 10 moved to (0, 1)."


def test_move_one_step_east_on_orthogonal_grid_without_constructor():
    class _DummyOrthogonalGrid(OrthogonalMooreGrid):
        pass

    orth_grid = object.__new__(_DummyOrthogonalGrid)
    start_target = (1, 1)
    # mesa.discrete_space grids use (row, col), so East increments col.
    end_target = (1, 2)
    start_cell = SimpleNamespace(coordinate=start_target, agents=[])
    end_cell = SimpleNamespace(coordinate=end_target, agents=[])
    orth_grid._cells = {start_target: start_cell, end_target: end_cell}

    model = DummyModel()
    model.grid = orth_grid

    agent = DummyAgent(unique_id=11, model=model)
    agent.cell = start_cell
    model.agents.append(agent)

    out = move_one_step(agent, "East")

    assert getattr(agent, "cell", None) is end_cell
    assert out == "agent 11 moved to (1, 2)."


def test_speak_to_records_on_recipients(mocker):
    model = DummyModel()

    # Sender and two recipients
    sender = DummyAgent(unique_id=10, model=model)
    r1 = DummyAgent(unique_id=11, model=model)
    r2 = DummyAgent(unique_id=12, model=model)

    # Attach mock memories to recipients
    r1.memory = SimpleNamespace(add_to_memory=mocker.Mock())
    r2.memory = SimpleNamespace(add_to_memory=mocker.Mock())

    model.agents = [sender, r1, r2]

    message = "Hello there"
    ret = speak_to(sender, [10, 11, 12], message)

    # Sender should not get message recorded, recipients should
    r1.memory.add_to_memory.assert_called_once()
    r2.memory.add_to_memory.assert_called_once()

    # Verify payload structure for one recipient
    _, kwargs = r1.memory.add_to_memory.call_args
    assert kwargs["type"] == "message"
    content = kwargs["content"]
    assert content["message"] == message
    assert content["sender"] == sender.unique_id
    assert set(content["recipients"]) == {11, 12}

    # Return string contains sender and recipients list
    assert "10" in ret and "11" in ret and "12" in ret and message in ret


def test_move_one_step_invalid_direction():
    model = DummyModel()
    model.grid = MultiGrid(width=4, height=4, torus=False)

    agent = DummyAgent(unique_id=3, model=model)
    model.agents.append(agent)
    model.grid.place_agent(agent, (2, 2))

    with pytest.raises(ValueError):
        move_one_step(agent, "north east")


def test_teleport_to_location_on_continuousspace():
    model = DummyModel()
    model.grid = None
    model.space = ContinuousSpace(x_max=10.0, y_max=10.0, torus=False)

    agent = DummyAgent(unique_id=5, model=model)
    model.agents.append(agent)
    model.space.place_agent(agent, (1.0, 1.0))

    out = teleport_to_location(agent, [5.0, 7.0])

    assert agent.pos == (5.0, 7.0)
    assert out == "agent 5 moved to (5.0, 7.0)."


def test_move_one_step_on_continuousspace():
    """move_one_step delegates to teleport_to_location, verify it works on ContinuousSpace too."""
    model = DummyModel()
    model.grid = None
    model.space = ContinuousSpace(x_max=10.0, y_max=10.0, torus=False)

    agent = DummyAgent(unique_id=6, model=model)
    model.agents.append(agent)
    model.space.place_agent(agent, (2.0, 2.0))

    result = move_one_step(agent, "North")

    assert agent.pos == (2.0, 3.0)
    assert result == "agent 6 moved to (2.0, 3.0)."
