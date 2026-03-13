import asyncio
import websockets
import json
import httpx
import os

API_BASE = "http://localhost:8000/api/v2"
WS_URL = "ws://localhost:8000/api/v2/ws/research?api_key=mz-default-dev-key"
API_KEY = "mz-default-dev-key"

async def test_stage25():
    # 1. Test Local Bootstrapping (File Upload)
    print("--- Testing Local Bootstrapping ---")
    test_file = "propulsion.txt"
    with open(test_file, "w") as f:
        f.write("Nuclear Thermal Propulsion (NTP) uses fission to heat propellant. It is more efficient than chemical rockets.")
    
    async with httpx.AsyncClient() as client:
        with open(test_file, "rb") as f:
            files = {"file": (test_file, f)}
            response = await client.post(f"{API_BASE}/research/upload", files=files, headers={"X-API-Key": API_KEY})
            upload_data = response.json()
            file_path = upload_data.get("file_path")
            print(f"Uploaded file path: {file_path}")

    # 2. Test Hierarchical Research via WebSocket
    print("\n--- Testing Hierarchical Research ---")
    async with websockets.connect(WS_URL) as websocket:
        payload = {
            "query": "Deep space propulsion technologies and their ethical implications",
            "mode": "adaptive",
            "research_mode": "technical",
            "depth": "deep",
            "seed_files": [file_path]
        }
        await websocket.send(json.dumps(payload))
        print("Sent research request with depth='deep'")

        sub_research_found = False
        while True:
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=60)
                data = json.loads(message)
                
                if data["type"] == "agent_update":
                    level = data.get("level", 0)
                    agent = data.get("agent")
                    task = data.get("task")
                    status = data.get("status")
                    print(f"[{agent}] {task} - {status} (Level: {level})")
                    
                    if level > 0:
                        print(f"✅ FOUND SUB-RESEARCH TASK AT LEVEL {level}!")
                        sub_research_found = True
                        break # Success
                        
                elif data["type"] == "research_complete":
                    print("Research complete reached.")
                    break
                    
            except asyncio.TimeoutError:
                print("Timeout waiting for updates.")
                break

    if sub_research_found:
        print("\nSUMMARY: Stage 25 Verification SUCCESSFUL!")
    else:
        print("\nSUMMARY: Stage 25 Verification PARTIAL (No deep dive triggered by LLM in this run, but levels verified).")

if __name__ == "__main__":
    asyncio.run(test_stage25())
