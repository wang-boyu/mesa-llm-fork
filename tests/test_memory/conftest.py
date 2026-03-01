import json
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def episodic_mock_agent(mock_agent):
    """Create an episodic-memory-specific variant of the shared mock agent."""
    agent = mock_agent

    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps({"grade": 3})
    agent.llm.generate.return_value = mock_response

    agent.model.steps = 100
    agent.model.events = []
    return agent
