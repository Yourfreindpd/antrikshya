"""WebSocket ASR server and ESP32 diagnostics dashboard."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import socket
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import psutil
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

try:
	from faster_whisper import WhisperModel
except ImportError:
	WhisperModel = None


SAMPLE_RATE = 16_000
MAX_AUDIO_SECONDS = 8
logger = logging.getLogger("esp32-asr")


@dataclass
class ServerState:
	clients: set[WebSocket] = field(default_factory=set)
	logs: deque[str] = field(default_factory=lambda: deque(maxlen=300))
	device_metrics: dict[str, Any] = field(default_factory=dict)
	streams: int = 0
	last_transcript: str = ""
	model: Any = None
	model_lock: threading.Lock = field(default_factory=threading.Lock)


state = ServerState()


def add_log(message: str) -> None:
	line = f"[{time.strftime('%H:%M:%S')}] {message}"
	state.logs.append(line)
	print(line, flush=True)


async def broadcast(payload: dict[str, Any]) -> None:
	dead: list[WebSocket] = []
	for client in tuple(state.clients):
		try:
			await client.send_text(json.dumps(payload))
		except Exception:
			dead.append(client)
	for client in dead:
		state.clients.discard(client)


def transcribe(audio: bytes) -> str:
	if not audio or WhisperModel is None:
		return ""
	if state.model is None:
		with state.model_lock:
			if state.model is None:
				state.model = WhisperModel("small.en", device="cpu", compute_type="int8")
	samples = np.frombuffer(audio, dtype="<i2").astype(np.float32) / 32768.0
	segments, _ = state.model.transcribe(
		 samples, language="en", beam_size=1, vad_filter=True,
		 condition_on_previous_text=False,
	)
	return " ".join(segment.text.strip() for segment in segments).strip()


async def handle_client(websocket: WebSocket) -> None:
	await websocket.accept()
	state.clients.add(websocket)
	remote = websocket.client.host if websocket.client else "unknown"
	add_log(f"ESP32 connected: {remote}")
	audio = bytearray()
	try:
		while True:
			message = await websocket.receive()
			if message.get("bytes") is not None:
				chunk = message["bytes"]
				if len(audio) + len(chunk) <= MAX_AUDIO_SECONDS * SAMPLE_RATE * 2:
					audio.extend(chunk)
				else:
					add_log("Audio ignored: stream exceeded 8 seconds")
			elif message.get("text") is not None:
				try:
					event = json.loads(message["text"])
				except json.JSONDecodeError:
					add_log(f"Invalid ESP32 message: {message['text']}")
					continue
				event_name = event.get("event")
				if event_name == "esp32_log":
					text = str(event.get("text", ""))
					add_log(f"ESP32 | {text}")
					await broadcast({"type": "esp32_log", "text": text})
				elif event_name == "device_metrics":
					state.device_metrics = event
					add_log(
						f"ESP32 metrics | state={event.get('state')} "
						f"cpu={event.get('cpu_percent')}% "
						f"heap={event.get('free_heap')}/{event.get('heap_size')}"
					)
					await broadcast({"type": "device_metrics", "metrics": event})
				elif event_name == "wake_detected":
					audio.clear()
					add_log(f"WAKE DETECTED | {event.get('keyword', 'unknown')}")
					await broadcast({"type": "wake_detected", "keyword": event.get("keyword")})
				elif event_name == "stream_end":
					state.streams += 1
					transcript = await asyncio.to_thread(transcribe, bytes(audio))
					state.last_transcript = transcript
					add_log(f"STREAM END | {len(audio)} bytes | ASR={transcript or '[not installed/no speech]'}")
					await broadcast({"type": "transcript", "text": transcript})
					audio.clear()
	except WebSocketDisconnect:
		pass
	finally:
		state.clients.discard(websocket)
		add_log(f"ESP32 disconnected: {remote}")


@asynccontextmanager
async def lifespan(app: FastAPI):
	add_log("ASR server ready")
	addresses = sorted({
		item[4][0]
		for item in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
		if item[4][0] != "127.0.0.1"
	})
	add_log(f"Listening on 0.0.0.0:8000; PC addresses={addresses or ['discover with ipconfig']}")
	yield


app = FastAPI(title="ESP32 ASR Server", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
	return {"ok": True, "clients": len(state.clients), "streams": state.streams}


@app.get("/api/metrics")
async def metrics() -> dict[str, Any]:
	return {
		"pc_cpu_percent": psutil.cpu_percent(interval=None),
		"pc_ram_percent": psutil.virtual_memory().percent,
		"esp32": state.device_metrics,
		"clients": len(state.clients),
		"streams": state.streams,
		"transcript": state.last_transcript,
		"logs": list(state.logs),
	}


@app.get("/")
async def dashboard() -> HTMLResponse:
	return HTMLResponse(DASHBOARD_HTML)


@app.websocket("/ws/audio")
async def audio_socket(websocket: WebSocket) -> None:
	await handle_client(websocket)


DASHBOARD_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>ESP32 ASR</title>
<style>body{background:#111820;color:#d8e2e8;font:14px Consolas,monospace;margin:0;padding:24px}h1{color:#ffca62}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.box{border:1px solid #30404a;padding:12px;background:#18232b}.value{font-size:20px;color:#7de2c3;margin-top:7px}#terminal{height:65vh;overflow:auto;white-space:pre-wrap;border:1px solid #30404a;background:#0b1014;padding:14px;margin-top:14px}@media(max-width:700px){.grid{grid-template-columns:repeat(2,1fr)}body{padding:12px}}</style></head>
<body><h1>ESP32 ASR SERVER</h1><div class="grid"><div class="box">PC CPU<div class="value" id="pcCpu">--</div></div><div class="box">PC RAM<div class="value" id="pcRam">--</div></div><div class="box">ESP32 CPU<div class="value" id="espCpu">--</div></div><div class="box">ESP32 FREE HEAP<div class="value" id="heap">--</div></div><div class="box">CLIENTS<div class="value" id="clients">--</div></div><div class="box">STREAMS<div class="value" id="streams">--</div></div></div><div id="terminal"></div>
<script>const terminal=document.querySelector('#terminal');async function tick(){const m=await fetch('/api/metrics').then(r=>r.json());pcCpu.textContent=m.pc_cpu_percent+'%';pcRam.textContent=m.pc_ram_percent+'%';const e=m.esp32||{};espCpu.textContent=e.cpu_percent==null?'--':e.cpu_percent+'%';heap.textContent=e.free_heap==null?'--':e.free_heap+'/'+e.heap_size;clients.textContent=m.clients;streams.textContent=m.streams;terminal.textContent=m.logs.join('\n');terminal.scrollTop=terminal.scrollHeight}setInterval(tick,1000);tick();</script></body></html>"""


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--host", default="0.0.0.0")
	parser.add_argument("--port", type=int, default=8000)
	args = parser.parse_args()
	print(f"Dashboard: http://127.0.0.1:{args.port}")
	print(f"ESP32 WebSocket: ws://<PC-IP>:{args.port}/ws/audio")
	import uvicorn
	uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
	main()
