"""
Dev server for the MELD-PostOp webapp.
Serves the project root with COOP/COEP headers for SharedArrayBuffer.

  Main app:  http://localhost:8080/web/
  Tests:     http://localhost:8080/web/test.html
  Test data: http://localhost:8080/data/tests/

Usage: python serve.py [port]   (default 8080)
"""
import sys, os
from http.server import HTTPServer, SimpleHTTPRequestHandler

PORT     = int(os.environ.get("PORT", sys.argv[1] if len(sys.argv) > 1 else 8080))
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class CORPHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT_DIR, **kwargs)

    def do_GET(self):
        if self.path in ('/', ''):
            self.send_response(302)
            self.send_header('Location', '/web/')
            self.end_headers()
        else:
            super().do_GET()

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy",  "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, fmt, *args):
        print(fmt % args)


print(f"Serving {ROOT_DIR} at http://localhost:{PORT}")
print(f"  Main app : http://localhost:{PORT}/web/")
print(f"  Tests    : http://localhost:{PORT}/web/test.html")
print("Press Ctrl-C to stop.\n")
HTTPServer(("", PORT), CORPHandler).serve_forever()
