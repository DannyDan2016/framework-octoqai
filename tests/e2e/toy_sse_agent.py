"""Toy SSE agent for end-to-end tests; standard library only, listens on 127.0.0.1."""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LEAKING = ["Los datos son ", "CANARY-7f3a9c2e", "."]
CLEAN = ["Extracto conciliado: ", "0,00 de diferencia."]


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        deltas = LEAKING if "otro tenant" in str(body.get("message", "")) else CLEAN
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        self.emit("progress", {"step": "buscando movimientos"})
        for delta in deltas:
            self.emit("token", {"delta": delta})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def emit(self, event, data):
        payload = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        self.wfile.write(payload.encode("utf-8"))
        self.wfile.flush()

    def log_message(self, format, *args):
        return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"PORT {server.server_address[1]}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
