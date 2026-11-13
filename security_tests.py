"""
PhishGuard security suite (49 cases).
Start the app first (python app.py), then run:  python security_suite.py
Writes security_results.csv, which can be used for Appendix D.3.
Needs: pip install requests
"""
import csv
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests

# Local listener: S15 and S16 pass only if no request reaches it.
LISTENER_PORT = 8000
hits = []


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        hits.append(self.path)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"listener reached")

    def log_message(self, *args):
        pass


try:
    _srv = HTTPServer(("127.0.0.1", LISTENER_PORT), _Handler)
    threading.Thread(target=_srv.serve_forever, daemon=True).start()
except OSError:
    raise SystemExit(f"Port {LISTENER_PORT} is busy. Close the "
                     "'python -m http.server' window and run again.")

BASE = "http://127.0.0.1:5000"
PREDICT = BASE + "/predict"
PREVIEW = BASE + "/website-preview"
CALLBACK = BASE + "/auth/google/callback"
TIMEOUT = 60

results = []


def record(case_id, name, expected, actual, passed, note=""):
    results.append([case_id, name, expected, actual,
                    "Pass" if passed else "FAIL", note])
    print(f"{case_id:4} {'Pass' if passed else 'FAIL':4} {name} -> {actual}")


def post(url, **kw):
    t = time.time()
    try:
        r = requests.post(url, timeout=TIMEOUT, **kw)
        return r, f"{time.time() - t:.2f}s"
    except requests.RequestException as e:
        return None, f"request error: {e}"


def get(url, **kw):
    t = time.time()
    try:
        r = requests.get(url, timeout=TIMEOUT, allow_redirects=False, **kw)
        return r, f"{time.time() - t:.2f}s"
    except requests.RequestException as e:
        return None, f"request error: {e}"


def check_json(case_id, name, payload, ok_codes, endpoint=PREDICT):
    r, note = post(endpoint, json=payload)
    code = r.status_code if r is not None else "none"
    record(case_id, name, "/".join(map(str, ok_codes)), code,
           code in ok_codes, note)


# ---------- M: malformed URLs on /predict ----------
m_cases = [
    ("M1", "empty url", "", [400]),
    ("M2", "whitespace only", "   ", [400]),
    ("M3", "scheme only", "http://", [400]),
    ("M4", "javascript: scheme", "javascript:alert(1)", [400]),
    ("M5", "ftp scheme", "ftp://example.com", [400]),
    ("M6", "file scheme", "file:///etc/passwd", [400]),
    ("M7", "bad port (99999)", "http://example.com:99999", [400]),
    ("M8", "non-numeric port", "http://example.com:abc", [400]),
    ("M9", "space in host", "http://exa mple.com", [200, 400]),
    ("M10", "null byte", "http://example.com/\x00", [200, 400]),
    ("M11", "cyrillic look-alike", "http://\u0430pple.com", [200, 400]),
    ("M12", "punycode host", "http://xn--80ak6aa92e.com", [200, 400]),
    ("M13", "script injection",
     "http://example.com/<script>alert(1)</script>", [200, 400]),
    ("M14", "SQL-style string", "http://example.com/' OR 1=1--", [200, 400]),
]
for cid, name, url, ok in m_cases:
    check_json(cid, name, {"url": url}, ok)

# ---------- P: bad request bodies ----------
check_json("P1", "missing url field", {}, [400])
check_json("P2", "url is a number", {"url": 123}, [400])
check_json("P3", "url is null", {"url": None}, [400])
check_json("P4", "url is a list", {"url": ["http://a.com"]}, [400])

# ---------- L: very long URLs ----------
for cid, n in [("L1", 2000), ("L2", 10000), ("L3", 100000), ("L4", 1000000)]:
    check_json(cid, f"{n:,}-char URL", {"url": "http://example.com/" + "a" * n},
               [200, 400, 413])

# ---------- P5-P8: protocol-level ----------
r, note = post(PREDICT, data="{not json",
               headers={"Content-Type": "application/json"})
record("P5", "invalid JSON", "400", r.status_code if r is not None else "none",
       r is not None and r.status_code == 400, note)

r, note = post(PREDICT, data="", headers={"Content-Type": "application/json"})
record("P6", "empty body", "400", r.status_code if r is not None else "none",
       r is not None and r.status_code == 400, note)

r, note = post(PREDICT, data="url=http://a.com",
               headers={"Content-Type": "text/plain"})
record("P7", "wrong content-type", "400/415",
       r.status_code if r is not None else "none",
       r is not None and r.status_code in (400, 415), note)

r, note = get(PREDICT)
record("P8", "GET on POST-only route", "405",
       r.status_code if r is not None else "none",
       r is not None and r.status_code == 405, note)

# ---------- S: SSRF guard on /website-preview ----------
blocked = [
    ("S1", "http://127.0.0.1:5000/health"),
    ("S2", "http://localhost:5000/"),
    ("S3", "http://[::1]:5000/"),
    ("S4", "http://0.0.0.0:5000/"),
    ("S5", "http://2130706433:5000/health"),
    ("S6", "http://0x7f000001:5000/health"),
    ("S7", "http://127.1:5000/health"),
    ("S8", "http://169.254.169.254/latest/meta-data/"),
    ("S9", "http://10.0.0.1/"),
    ("S10", "http://192.168.1.1/"),
    ("S11", "http://172.16.0.1/"),
    ("S12", "http://user:pass@8.8.8.8/"),
    ("S13", "file:///C:/Windows/win.ini"),
    ("S14", "http://localtest.me:5000/health"),
]
for cid, url in blocked:
    r, note = post(PREVIEW, json={"url": url})
    code = r.status_code if r is not None else "none"
    body = (r.text[:80].replace("\n", " ") if r is not None else note)
    record(cid, url, "blocked (400)", code, code == 400, body)

# S15, S16: public site redirects to a private address (needs internet).
# Pass only if the local listener received no request.
redirects = [
    ("S15", "https://httpbin.org/redirect-to?url=http://127.0.0.1:%d/s15"),
    ("S16", "https://httpbingo.org/redirect-to?url=http://localhost:%d/s16"),
]
for cid, tmpl in redirects:
    url = tmpl % LISTENER_PORT
    r, note = post(PREVIEW, json={"url": url})
    time.sleep(2)
    reached = [h for h in hits if h.startswith("/" + cid.lower())]
    code = r.status_code if r is not None else "none"
    record(cid, url, "no request reaches private address",
           f"HTTP {code}; listener hits: {len(reached)}",
           len(reached) == 0, note)

r, note = post(PREVIEW, json={"url": "https://example.com"})
record("S0", "control: public site allowed", "allowed (200)",
       r.status_code if r is not None else "none",
       r is not None and r.status_code == 200, note)

# ---------- A: auth and debug ----------
r, note = get(BASE + "/login")
record("A1", "/login (record behaviour)", "no 500",
       r.status_code if r is not None else "none",
       r is not None and r.status_code != 500, note)

r, note = get(CALLBACK, params={"state": "forged", "code": "x"})
record("A2", "callback with forged state", "rejected (400/401/403)",
       r.status_code if r is not None else "none",
       r is not None and r.status_code in (400, 401, 403), note)

r, note = get(CALLBACK)
record("A3", "callback with no parameters", "rejected (400/401/403)",
       r.status_code if r is not None else "none",
       r is not None and r.status_code in (400, 401, 403), note)

r, note = get(CALLBACK, params={"error": "access_denied"})
record("A4", "callback user denied consent", "no 500",
       r.status_code if r is not None else "none",
       r is not None and r.status_code != 500, note)

r, note = get(BASE + "/console")
record("A5", "Werkzeug debug console exposed", "404",
       r.status_code if r is not None else "none",
       r is not None and r.status_code == 404, note)

r, note = get(BASE + "/no-such-page-xyz")
leak = r is not None and "Traceback" in r.text
record("A6", "404 page leaks no trace", "404, no traceback",
       r.status_code if r is not None else "none",
       r is not None and r.status_code == 404 and not leak, note)

# ---------- save and summarise ----------
with open("security_results.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["ID", "Test", "Expected", "Actual", "Result", "Note"])
    w.writerows(results)

passed = sum(1 for r in results if r[4] == "Pass")
print(f"\n{passed} of {len(results)} passed. Saved security_results.csv")
for r in results:
    if r[4] == "FAIL":
        print("FAILED:", r[0], r[1])