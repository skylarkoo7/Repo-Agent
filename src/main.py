#!/usr/bin/env python
import sys
from graph import OpensourcerepoGraph
import agentstack
import agentops

agentops.init(default_tags=agentstack.get_tags())

instance = OpensourcerepoGraph()
initial_state = {
    "question": '''Datetime from timestamp , return the issue url with it as well''',
    "generation": "",
    "retrievelink": "",
    "retriever_db": "",
    "issue_url":"https://github.com/fenilfaldu/AI-Agent/issues/6",
}   

def run(inputs=None):
    """
    Run the agent.
    """
    if inputs is None:
        inputs = initial_state  # Use predefined initial state

    # ✅ Debug: Print the initial state
    print("\n🔹 Initial State before Execution:", inputs)

    # ✅ Call the graph execution
    instance.run(inputs=inputs)

    # ✅ Debug: Print state after execution
    print("\n🔹 Final State after Execution:", inputs)


if __name__ == '__main__':
    run()