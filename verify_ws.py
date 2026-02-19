import asyncio
import websockets
import json

async def verify_websocket():
    uri = "ws://127.0.0.1:8000/api/v2/ws/research"
    try:
        async with websockets.connect(uri) as websocket:
            print(f"Connected to {uri}")
            
            # Send research query
            query = {
                "query": "How do LLMs work?",
                "mode": "sequential"
            }
            await websocket.send(json.dumps(query))
            print(f"Sent query: {query}")
            
            # Listen for updates
            while True:
                try:
                    message = await websocket.recv()
                    data = json.loads(message)
                    print(f"Received: {json.dumps(data, indent=2)}")
                    
                    if data["type"] == "research_complete":
                        print("Research complete!")
                        break
                    if data["type"] == "error":
                        print(f"Error: {data['message']}")
                        break
                except websockets.exceptions.ConnectionClosed:
                    print("Connection closed")
                    break
    except Exception as e:
        print(f"Failed to connect or receive: {e}")

if __name__ == "__main__":
    asyncio.run(verify_websocket())
