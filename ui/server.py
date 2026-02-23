"""FastAPI server with WebSocket streaming for the agent chat UI."""

import json
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from agent.graph import graph

app = FastAPI(title="Agentic Helper")

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="ui/static"), name="static")


@app.get("/")
async def root():
    """Redirect to the chat UI."""
    with open("ui/static/index.html") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/tools")
async def list_tools():
    """List all available tools the agent can use."""
    from tools import get_all_tools

    tools = get_all_tools()
    return [
        {"name": t.name, "description": t.description}
        for t in tools
    ]


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for streaming agent responses."""
    await websocket.accept()

    # Each WebSocket connection gets a unique thread_id for memory
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    try:
        while True:
            # Receive user message
            data = await websocket.receive_text()
            message = json.loads(data)
            user_input = message.get("content", "")

            if not user_input.strip():
                continue

            # Send "thinking" indicator
            await websocket.send_text(json.dumps({
                "type": "status",
                "content": "thinking",
            }))

            # Stream the agent response
            try:
                final_response = ""
                tool_calls_made = []

                for event in graph.stream(
                    {"messages": [("user", user_input)]},
                    config=config,
                    stream_mode="values",
                ):
                    messages = event.get("messages", [])
                    if messages:
                        last = messages[-1]

                        # Track tool calls
                        if hasattr(last, "tool_calls") and last.tool_calls:
                            for tc in last.tool_calls:
                                tool_info = {
                                    "name": tc.get("name", "unknown"),
                                    "args": tc.get("args", {}),
                                }
                                tool_calls_made.append(tool_info)
                                await websocket.send_text(json.dumps({
                                    "type": "tool_call",
                                    "content": tool_info,
                                }))

                        # Track tool results
                        if last.type == "tool":
                            await websocket.send_text(json.dumps({
                                "type": "tool_result",
                                "content": {
                                    "name": last.name,
                                    "result": last.content[:500],  # Truncate long results
                                },
                            }))

                        # Final AI response
                        if last.type == "ai" and last.content and not (
                            hasattr(last, "tool_calls") and last.tool_calls
                        ):
                            final_response = last.content

                await websocket.send_text(json.dumps({
                    "type": "response",
                    "content": final_response,
                    "tools_used": [tc["name"] for tc in tool_calls_made],
                }))

            except Exception as e:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "content": f"Agent error: {str(e)}",
                }))

    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
