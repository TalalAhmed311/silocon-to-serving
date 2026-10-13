/* R.6 — HTTP and networking basics. Sockets, TCP/DNS/TLS costs, requests, status codes, SSE streaming, retries, concurrency. */
(function () {
  const F = S2S.fmt;
  // monospace-looking lines of coloured text on the stage
  const lines = (G, x, y, rows, dy = 22, size = 13) => rows.forEach(([t, c], i) => G.text(x, y + i * dy, t, { size, color: c || "ink" }));

  S2S.lesson({
    id: "r-6", n: "R.6", title: "HTTP and networking basics",
    subtitle: "Readiness · first principles · T0, any laptop",
    kicker: "Lesson · ≈ 30 min",
    headline: "What happens before the first token",
    intro: `<p><b>Already comfortable with sockets, requests, status codes, streaming responses and retries? Answer the checkpoints. If you get them all right, skip to R.7.</b></p>
<p>Every LLM server in this course talks to its users over HTTP. When someone says "the first token took 700 ms", part of that time is the model and part of it is the network. To tell them apart you need to know what happens between pressing Enter and seeing a word. This lesson follows one request from end to end: how the two programs find each other, what they send, how an answer can arrive in pieces, and what to do when something fails. It sets up P1.3 (measuring TTFT and ITL) and P2.7 (the gateway).</p>
<p>You need to be able to read a function and run a command in a terminal. Scroll: the dark panel redraws for each step.</p>`,
    facts: ["10 steps", "5 checkpoints", "1 simulator", "4 exercises"],
    legend: [["k", "client"], ["v", "server"], ["q", "TLS / security"], ["ok", "success"], ["hot", "error / wait"], ["w", "network"]],
    prev: "r-5", next: "r-7",
    steps: [
      { rail: "client and server", title: "Two programs, one address, one port",
        body: `<p>A <b>server</b> is a program that waits for requests. A <b>client</b> is a program that sends them. They can be on the same computer or on opposite sides of the world.</p>
<p>To find a server you need two things. The <b>IP address</b> names a machine. <code>127.0.0.1</code> always means "this machine", and the name <code>localhost</code> points to it. The <b>port</b> is a number from 0 to 65,535 that picks one program on that machine, because many can run at once. Together they read <code>localhost:8000</code>: "the program listening on port 8000 here".</p>
<p>Once connected, each side holds a <b>socket</b>: one end of the connection. The server listens on a fixed port. The client's operating system picks a random free port for its own end. The course's mock LLM server (<code>platform/mockllm/server.py</code>) is exactly this: a program listening on a port.</p>`,
        scene(G) {
          G.box(24, 150, 176, 110, "", { stroke: "k" });
          G.text(112, 176, "client", { size: 15, weight: 700, anchor: "middle", color: "k" });
          G.label(112, 198, "curl, browser, code", { anchor: "middle", size: 11 });
          G.circle(200, 230, 9, { fill: "k" }); G.label(112, 240, "127.0.0.1 : 50312", { anchor: "middle", size: 11, color: "ink" });
          G.box(430, 60, 180, 300, "", { stroke: "v" });
          G.text(520, 86, "one machine", { size: 13, weight: 700, anchor: "middle", color: "v" });
          G.label(520, 104, "127.0.0.1", { anchor: "middle", size: 11 });
          [["22", "other program"], ["8000", "LLM server"], ["8080", "gateway"], ["9090", "metrics"]].forEach(([p, who], i) => {
            const y = 124 + i * 56, on = p === "8000";
            G.box(444, y, 152, 42, "", { stroke: on ? "v" : "line", fill: on ? "v" : undefined });
            G.text(456, y + 18, "port " + p, { size: 13, weight: 700, color: on ? "bg" : "muted" });
            G.text(456, y + 34, who, { size: 11, color: on ? "bg" : "muted" });
          });
          G.circle(430, 235, 9, { fill: "v" });
          const l = G.line(209, 230, 421, 235, { color: "ok", w: 3 });
          G.label(310, 218, "one connection", { anchor: "middle", size: 12, color: "ok" });
          G.label(310, 256, "socket ⟷ socket", { anchor: "middle", size: 12 });
          G.label(30, 392, "ports shown are examples; the client port is chosen by the OS", { size: 11 });
          G.from(l, { opacity: 0, duration: 0.6 });
          G.caption("an address picks the machine, a port picks the program");
        } },

      { rail: "tcp", title: "TCP gives you a reliable stream, after one round trip",
        body: `<p>The network underneath only moves small, unreliable packets. They can be lost, repeated or reordered. <b>TCP</b> is the layer that hides this. It gives each side a reliable, in-order stream of bytes, like a pipe. HTTP runs on top of it.</p>
<p>Before any data flows, the two ends must agree to talk. The client sends SYN ("can we talk?"), the server answers SYN-ACK ("yes"), and the client replies ACK. The client can send its first bytes along with that last message. The setup costs <b>one round trip</b>: the time for a message to go there and back, the <b>round-trip time</b> (RTT). Two numbers describe a network link:</p>
<ul><li><b>Latency</b>: how long one message takes. A message across an ocean takes tens of milliseconds, whatever its size.</li>
<li><b>Bandwidth</b>: how many bytes per second fit through. A big download is limited by this.</li></ul>
<p>A small request is limited by latency, not bandwidth. That is why a prompt of a few hundred bytes still feels the distance to the server.</p>`,
        scene(G) {
          const cx = 140, sx = 500, y0 = 70, y1 = 340;
          G.text(cx, 50, "client", { size: 14, weight: 700, anchor: "middle", color: "k" }); G.text(sx, 50, "server", { size: 14, weight: 700, anchor: "middle", color: "v" });
          G.line(cx, y0, cx, y1, { color: "k", w: 2 }); G.line(sx, y0, sx, y1, { color: "v", w: 2 });
          const msg = (ya, yb, lab, col, ltr = true) => {
            const a = G.arrow(ltr ? cx : sx, ya, ltr ? sx : cx, yb, { color: col, w: 2.5 });
            G.text(320, (ya + yb) / 2 + (ltr ? -14 : 24), lab, { size: 12, anchor: "middle", color: col });
            return a;
          };
          const a = [msg(90, 120, "SYN", "w"), msg(135, 165, "SYN-ACK", "w", false), msg(198, 228, "ACK + first bytes of the request", "ok"), msg(268, 298, "first bytes of the response", "v", false)];
          G.path("M60 90 L60 165", { color: "hot", w: 2 }); G.path("M54 90 L66 90 M54 165 L66 165", { color: "hot", w: 2 });
          G.text(48, 124, "1 RTT", { size: 12, color: "hot", anchor: "end" }); G.text(48, 142, "setup", { size: 12, color: "hot", anchor: "end" });
          G.path("M60 198 L60 298", { color: "hot", w: 2, dash: "4 3" });
          G.text(48, 242, "1 RTT", { size: 12, color: "hot", anchor: "end" }); G.text(48, 260, "ask + answer", { size: 12, color: "hot", anchor: "end" });
          G.text(24, 372, "RTT = 100 ms (example):  setup 100 ms + request/response 100 ms", { size: 12 });
          G.text(24, 394, "= 200 ms before the first reply byte, plus the server's own work", { size: 12 });
          G.from(a, { opacity: 0, stagger: 0.3, duration: 0.3 });
          G.caption("every back-and-forth costs one round trip");
        } },

      { rail: "dns and tls", title: "DNS and TLS are costs you pay before your request leaves",
        body: `<p>Two more steps sit in front of most real requests.</p>
<p><b>DNS</b> turns a name like <code>api.example.com</code> into an IP address. The client asks a resolver, which costs about one round trip to that resolver. Answers are cached, so you often pay it once and then not for a while.</p>
<p><b>TLS</b> is what makes <code>https</code> private. The two ends agree on keys, and the client checks the server's certificate to be sure it is talking to the right machine. With TLS 1.3 this takes <b>one</b> extra round trip; with TLS 1.2, two.</p>
<p>So a first request on a fresh connection to a named https server pays, in order: DNS, TCP setup, TLS setup, then the request and the first byte of the answer. <b>Keep-alive</b> reuses one connection for many requests, so the next request skips all of it. That is why load tests and gateways keep connections open.</p>`,
        check: { q: "RTT is 80 ms. A client opens a fresh connection, with TLS 1.3, and the server needs 100 ms to produce the first byte. Ignoring DNS, when is the earliest the first byte arrives?",
          options: ["180 ms", "260 ms", "340 ms", "420 ms"], answer: 2,
          why: "TCP setup is one round trip (80), TLS 1.3 is one more (80), and then the request goes out and the reply comes back (80), plus the server's 100 ms. That is 3 × 80 + 100 = 340 ms. With TLS 1.2 it would be 420. With keep-alive it would be 80 + 100 = 180 ms." },
        scene(G) {
          const R = 50, segs = [["DNS", R, "w"], ["TCP", R, "k"], ["TLS 1.3", R, "q"], ["request + reply", R, "v"], ["server work", 100, "hot"]];
          const s = 560 / 300; let x = 40; const bars = [];
          G.text(40, 60, "first request on a fresh https connection", { size: 13, weight: 700 });
          G.label(40, 80, "RTT = 50 ms and server work = 100 ms (examples)", { size: 11 });
          segs.forEach(([n, d, c]) => {
            bars.push(G.rect(x, 100, d * s - 2, 46, { fill: c, rx: 4, opacity: 0.9 }));
            G.text(x + (d * s) / 2, 128, d + " ms", { size: 12, weight: 700, color: "bg", anchor: "middle" });
            G.text(x + (d * s) / 2, 168, n, { size: 11, anchor: "middle", color: c === "w" ? "muted" : c });
            x += d * s;
          });
          G.path("M40 192 L600 192", { color: "line" });
          G.text(40, 214, "first byte at 300 ms", { size: 14, weight: 700, color: "hot" });
          G.text(40, 236, "150 ms of it passes before the request even leaves", { size: 12, color: "muted" });
          G.text(40, 280, "second request, same connection (keep-alive)", { size: 13, weight: 700 });
          const w2 = [["request + reply", R, "v"], ["server work", 100, "hot"]]; x = 40;
          w2.forEach(([n, d, c]) => {
            bars.push(G.rect(x, 300, d * s - 2, 46, { fill: c, rx: 4, opacity: 0.9 }));
            G.text(x + (d * s) / 2, 328, d + " ms", { size: 12, weight: 700, color: "bg", anchor: "middle" });
            G.text(x + (d * s) / 2, 366, n, { size: 11, anchor: "middle", color: c });
            x += d * s;
          });
          G.text(40 + 150 * s + 14, 328, "← first byte at 150 ms", { size: 13, weight: 700, color: "ok" });
          G.label(40, 400, "DNS is often cached, and TLS 1.2 costs one more round trip than 1.3", { size: 11 });
          G.from(bars, { scaleX: 0, transformOrigin: "left center", stagger: 0.12, duration: 0.3 });
          G.caption("setup costs are paid once per connection, not once per request");
        } },

      { rail: "http", title: "HTTP is text with a simple shape",
        body: `<p>Once the connection exists, the two programs speak <b>HTTP</b>, which is plain text with a fixed layout.</p>
<p>A <b>request</b> has: a <b>method</b> (<code>GET</code> reads, <code>POST</code> sends data), a <b>path</b> (what you want), <b>headers</b> (<code>Name: value</code> lines about the request), a blank line, and an optional <b>body</b>.</p>
<p>A <b>response</b> has: a <b>status code</b> (a number and a word: how it went), headers, a blank line and a body.</p>
<p>The tool <code>curl -v</code> prints both sides. Lines starting <code>&gt;</code> were sent; lines starting <code>&lt;</code> came back. This is the shape of a real exchange with the course's mock server (the exact headers vary):</p>
<div class="eq">&gt; GET /v1/models HTTP/1.1
&gt; Host: localhost:8000
&gt; Accept: */*
&lt; HTTP/1.1 200 OK
&lt; content-type: application/json
&lt; content-length: 87
{"object":"list","data":[{"id":"mock-llama-8b", ...}]}</div>`,
        scene(G) {
          G.text(24, 46, "request  (client → server)", { size: 13, weight: 700, color: "k" });
          G.rect(20, 56, 600, 128, { stroke: "k", rx: 8 });
          lines(G, 36, 82, [["GET /v1/models HTTP/1.1", "k"], ["Host: localhost:8000", "ink"], ["Accept: */*", "ink"], ["(blank line, then an optional body)", "muted"]], 24, 14);
          G.text(318, 82, "← method, path, version", { size: 11, color: "k" }); G.text(318, 106, "← headers: Name: value", { size: 11, color: "muted" });
          G.text(24, 226, "response  (server → client)", { size: 13, weight: 700, color: "v" });
          G.rect(20, 236, 600, 156, { stroke: "v", rx: 8 });
          lines(G, 36, 262, [["HTTP/1.1 200 OK", "ok"], ["content-type: application/json", "ink"], ["content-length: 87", "ink"], ["", "ink"], ["{\"object\":\"list\",\"data\":[ ... ]}", "v"]], 24, 14);
          G.text(318, 262, "← status code and word", { size: 11, color: "ok" }); G.text(318, 286, "← headers", { size: 11, color: "muted" }); G.text(318, 334, "← body: the payload", { size: 11, color: "v" });
          G.caption("request: method, path, headers, body. Response: status, headers, body");
        } },

      { rail: "json api", title: "LLM servers speak JSON over HTTP",
        body: `<p>The body is just bytes, so client and server must agree on a format. For APIs the usual choice is <b>JSON</b>: nested lists and key-value pairs as text. Most LLM servers copy the OpenAI-style layout, and this course does too. To ask for a completion you <code>POST</code> to <code>/v1/chat/completions</code> with a JSON body:</p>
<div class="eq">{
  "model": "mock-llama-8b",
  "messages": [{"role": "user", "content": "Say hi"}],
  "max_tokens": 16,
  "stream": true
}</div>
<ul><li><code>model</code>: which model to run (a gateway also uses it to pick a backend).</li>
<li><code>messages</code>: the conversation so far, each with a role and text.</li>
<li><code>max_tokens</code>: the longest answer to generate.</li>
<li><code>stream</code>: <code>false</code> means one JSON reply at the end; <code>true</code> means pieces as they are made (step 7).</li></ul>
<p>Headers carry <code>Content-Type: application/json</code> and usually <code>Authorization: Bearer &lt;key&gt;</code>. The <code>Content-Type</code> header tells the receiver how to read the body. A body that does not parse is the caller's mistake.</p>`,
        scene(G) {
          G.text(24, 44, "POST /v1/chat/completions", { size: 14, weight: 700, color: "k" });
          G.label(24, 64, "Authorization: Bearer <key>     Content-Type: application/json", { size: 11 });
          G.rect(20, 78, 380, 190, { stroke: "k", rx: 8 });
          const body = [["{", "ink"], ["  \"model\": \"mock-llama-8b\",", "k"], ["  \"messages\": [{\"role\": \"user\",", "v"], ["                \"content\": \"Say hi\"}],", "v"], ["  \"max_tokens\": 16,", "q"], ["  \"stream\": true", "hot"], ["}", "ink"]];
          lines(G, 34, 104, body, 24, 13);
          const notes = [[130, "k", "which model"], [180, "v", "the conversation"], [228, "q", "answer length cap"], [252, "hot", "pieces, or one reply?"]];
          notes.forEach(([y, c, t]) => G.text(414, y, "← " + t, { size: 11, color: c }));
          G.text(24, 306, "response if \"stream\": false", { size: 13, weight: 700, color: "v" });
          G.rect(20, 316, 600, 84, { stroke: "v", rx: 8 });
          lines(G, 34, 342, [["{\"object\": \"chat.completion\", \"model\": \"mock-llama-8b\",", "ink"], ["  \"choices\": [{\"message\": {\"content\": \"Hi!\"}, ...}]}", "ink"]], 22, 12);
          G.label(34, 392, "abridged: real replies also carry ids and usage counts", { size: 11 });
          G.caption("a JSON body in, a JSON body (or a stream) out");
        } },

      { rail: "status codes", title: "The status code says who failed and whether to retry",
        body: `<p>Every response begins with a three-digit <b>status code</b>. The first digit is the class: <b>2xx</b> success, <b>4xx</b> the caller's request is the problem, <b>5xx</b> the server's. A gateway reads the code and decides what to do:</p>
<table><tr><th>code</th><th>meaning</th><th>what a gateway does</th></tr>
<tr><td>200</td><td>OK</td><td>pass it through</td></tr>
<tr><td>400</td><td>malformed request</td><td>return it; retrying cannot help</td></tr>
<tr><td>401</td><td>missing or wrong key</td><td>reject; do not forward</td></tr>
<tr><td>404</td><td>unknown path or model</td><td>return it</td></tr>
<tr><td>429</td><td>too many requests</td><td>tell the caller to wait (<code>Retry-After</code>)</td></tr>
<tr><td>500</td><td>server bug or crash</td><td>maybe retry elsewhere</td></tr>
<tr><td>503</td><td>overloaded or draining</td><td>try another backend, shed load</td></tr></table>
<p>Your own validation errors also land here. A body that is not valid JSON is a 400. A body that parses but has the wrong fields is often a 422 (FastAPI's default). Exercise 3 builds exactly that.</p>`,
        check: { q: "A request arrives at your gateway with a wrong API key. What should the gateway answer, and should it try another backend?",
          options: ["401, and no: the request never reaches a backend", "503, and yes: another backend may accept it", "429, and yes after a short pause", "404, and no: the key does not exist as a path"], answer: 0,
          why: "A wrong key is an authentication failure: 401. The caller must fix it, so retrying anywhere gives the same answer, and the gateway should refuse before spending any backend capacity. 429 and 503 are about load, not credentials." },
        scene(G) {
          const rows = [["200", "ok", "OK", "pass through"], ["400", "hot", "malformed", "return, no retry"], ["401", "hot", "bad key", "reject, no forward"], ["404", "hot", "not found", "return"], ["429", "v", "slow down", "wait, Retry-After"], ["500", "hot", "server error", "retry if safe"], ["503", "v", "overloaded", "other backend"]];
          G.label(24, 40, "code", { size: 11 }); G.label(100, 40, "meaning", { size: 11 }); G.label(250, 40, "gateway does", { size: 11 });
          const bx = [];
          rows.forEach(([c, col, m, a], i) => {
            const y = 52 + i * 48;
            bx.push(G.box(20, y, 64, 38, c, { fill: col, size: 16, weight: 700 }));
            G.text(100, y + 24, m, { size: 14 }); G.text(250, y + 24, a, { size: 14, color: "muted" });
          });
          G.rect(440, 52, 180, 38, { stroke: "ok", rx: 6 }); G.text(530, 76, "2xx  success", { size: 13, anchor: "middle", color: "ok" });
          G.rect(440, 100, 180, 38, { stroke: "hot", rx: 6 }); G.text(530, 124, "4xx  caller's fault", { size: 13, anchor: "middle", color: "hot" });
          G.rect(440, 148, 180, 38, { stroke: "v", rx: 6 }); G.text(530, 172, "5xx  server's side", { size: 13, anchor: "middle", color: "v" });
          G.label(440, 214, "429 and 503 are", { size: 11 }); G.label(440, 230, "\"not now\": try again.", { size: 11 });
          G.label(440, 252, "400, 401, 404 are", { size: 11 }); G.label(440, 268, "\"not ever\": don't.", { size: 11 });
          G.from(bx, { opacity: 0, x: -14, stagger: 0.06, duration: 0.25 });
          G.caption("the first digit tells you who has to change something");
        } },

      { rail: "streaming", title: "Server-sent events stream tokens as they are made",
        body: `<p>A model makes its answer one token at a time. Waiting for all of them is slow: a 500-token answer at 30 ms per token is 15 seconds of blank screen. With <b>streaming</b>, the server keeps the response open and sends each piece as soon as it exists.</p>
<p>The format is <b>server-sent events</b> (SSE). Each event is one or more <code>data:</code> lines followed by a <b>blank line</b>. The end is marked by <code>data: [DONE]</code>:</p>
<div class="eq">data: {"choices":[{"delta":{"content":"Hel"}}]}

data: {"choices":[{"delta":{"content":"lo"}}]}

data: [DONE]</div>
<p>The client notes the time each event arrives. <b>TTFT</b> (time to first token) is the wait from sending the request to the first event. <b>ITL</b> (inter-token latency) is the gap between consecutive events. Without streaming, the user sees nothing until the last token, so the first token and the full answer arrive together. P1.3 turns these two numbers into percentiles and SLOs.</p>`,
        check: { q: "A client sends a request at t = 10.00 s. Chunks arrive at 10.40, 10.45, 10.50 and 10.70 s. What are the TTFT and the mean ITL?",
          options: ["0.40 s and 0.10 s", "0.70 s and 0.10 s", "0.40 s and 0.075 s", "0.30 s and 0.05 s"], answer: 0,
          why: "TTFT is first arrival minus send: 10.40 − 10.00 = 0.40 s. The gaps are 0.05, 0.05 and 0.20, three of them, so the mean is 0.30 / 3 = 0.10 s. Dividing by 4 gives 0.075, but four chunks have only three gaps. The 0.20 s gap is a stall that the mean hides." },
        scene(G) {
          const X = (t) => 40 + t * 700, tk = [0.5, 0.55, 0.6, 0.65, 0.8];
          G.line(40, 330, 610, 330, { color: "line" });
          [0, 0.2, 0.4, 0.6, 0.8].forEach((t) => { G.line(X(t), 326, X(t), 334, { color: "line" }); G.label(X(t), 350, t.toFixed(1) + " s", { anchor: "middle", size: 11 }); });
          G.text(24, 44, "streaming: the user sees tokens arrive", { size: 13, weight: 700, color: "ok" });
          G.line(X(0), 150, X(0.8) + 10, 150, { color: "line" });
          G.circle(X(0), 150, 5, { fill: "k" }); G.label(X(0) - 8, 134, "request sent", { size: 11 });
          const t1 = tk.map((t, i) => G.rect(X(t) - 5, 138, 10, 24, { fill: i === 0 ? "ok" : "v", rx: 3 }));
          G.path(`M${X(0)} 112 L${X(0.5)} 112 M${X(0)} 106 L${X(0)} 118 M${X(0.5)} 106 L${X(0.5)} 118`, { color: "ok", w: 2 });
          G.text((X(0) + X(0.5)) / 2, 100, "TTFT 0.50 s", { size: 13, anchor: "middle", color: "ok", weight: 700 });
          G.label(24, 190, "ITL gaps, ms:", { size: 11 });
          [["50", 0.5, 0.55], ["50", 0.55, 0.6], ["50", 0.6, 0.65], ["150", 0.65, 0.8]].forEach(([d, a, b], i) => G.text((X(a) + X(b)) / 2, 190, d, { size: 12, anchor: "middle", color: i === 3 ? "hot" : "muted", weight: i === 3 ? 700 : 400 }));
          G.text((X(0.65) + X(0.8)) / 2, 208, "an ITL stall", { size: 11, color: "hot", anchor: "middle" });
          G.text(24, 244, "no streaming: nothing until the end", { size: 13, weight: 700, color: "hot" });
          G.line(X(0), 280, X(0.8) + 10, 280, { color: "line" });
          G.circle(X(0), 280, 5, { fill: "k" });
          G.rect(X(0.8) - 5, 268, 10, 24, { fill: "hot", rx: 3 });
          G.text(X(0.8) + 5, 262, "all tokens at once", { size: 11, anchor: "end", color: "hot" });
          G.label(24, 392, "the first token the user sees comes at 0.50 s vs 0.80 s", { size: 12, color: "ink" });
          G.from(t1, { scaleY: 0, transformOrigin: "center bottom", stagger: 0.2, duration: 0.2 });
          G.caption("streaming makes TTFT and every ITL gap visible");
        } },

      { rail: "retries", title: "A retry is only safe if repeating the request is harmless",
        body: `<p>Things fail. A server can answer with an error, or not answer at all. A <b>timeout</b> is how long the client waits before giving up. After a failure the natural move is to <b>retry</b>, but that is only safe when the request is <b>idempotent</b>: doing it twice has the same effect as doing it once. A <code>GET</code> is. "Create this order" is not, unless it carries an <i>idempotency key</i> the server can use to spot the repeat.</p>
<p>The status code and the kind of failure tell you what happened:</p>
<ul><li><b>429, 503</b>: the server refused before doing any work. Safe to retry, even for a non-idempotent request.</li>
<li><b>Timeout, 500, 502, 504</b>: the work may have happened and the reply been lost. Retry only if idempotent.</li>
<li><b>400, 401, 404</b>: the request is wrong. Never retry.</li></ul>
<p>When you do retry, do not retry at once. Wait longer after each failure: <b>exponential backoff</b>, e.g. <code>0.1 s, 0.2 s, 0.4 s, ...</code> up to a cap. And pick the actual wait at random between 0 and that bound (<b>jitter</b>), so a thousand clients that failed together do not all come back together.</p>`,
        check: { q: "A client POSTs a request that is NOT idempotent and gets no answer before its timeout. What is the safe policy?",
          options: ["Retry at once: the server probably failed", "Do not blindly retry: the work may already have happened. Use an idempotency key or ask the user", "Retry three times with backoff", "Treat it as a 400"], answer: 1,
          why: "Silence does not mean failure. The server may have finished the work and the reply got lost, so a retry could do it twice (two orders, two charges). Retry only if the request is idempotent or carries an idempotency key." },
        scene(G) {
          G.text(24, 44, "what to do after a failure", { size: 13, weight: 700 });
          const R = [["429 / 503", "refused, no work done", "retry", "ok"], ["timeout / 500 / 502 / 504", "may have run", "retry only if idempotent", "v"], ["400 / 401 / 404", "request is wrong", "never retry", "hot"]];
          R.forEach(([c, why, act, col], i) => {
            const y = 60 + i * 66;
            G.box(20, y, 176, 50, c, { stroke: col, color: col, size: 11 });
            G.arrow(198, y + 25, 220, y + 25, { color: "muted" });
            G.box(224, y, 160, 50, why, { stroke: "line", size: 11, color: "muted" });
            G.arrow(386, y + 25, 410, y + 25, { color: "muted" });
            G.text(416, y + 22, act.split(" ").slice(0, 2).join(" "), { size: 12, color: col, weight: 700 }); if (act.split(" ").length > 2) G.text(416, y + 38, act.split(" ").slice(2).join(" "), { size: 12, color: col, weight: 700 });
          });
          G.text(24, 276, "backoff bound: min(cap, 0.1 · 2^(n−1)) seconds, cap 5 s", { size: 12 });
          const b = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2];
          const bars = G.bars(40, 394, b, { w: 54, gap: 16, h: 84, max: 3.2, fill: "q" });
          b.forEach((v, i) => { G.text(40 + i * 70 + 27, 388 - (84 * v) / 3.2, v + " s", { size: 11, anchor: "middle" }); G.label(40 + i * 70 + 27, 410, "try " + (i + 1), { size: 11, anchor: "middle" }); });
          G.text(470, 330, "jitter picks a", { size: 12, color: "muted" }); G.text(470, 348, "random wait between", { size: 12, color: "muted" }); G.text(470, 366, "0 and the bar", { size: 12, color: "muted" });
          G.from(bars, { scaleY: 0, transformOrigin: "center bottom", stagger: 0.1, duration: 0.25 });
          G.caption("retry what is safe, wait longer each time, add randomness");
        } },

      { rail: "concurrency", title: "A server holds many connections open at once",
        body: `<p>A streaming answer keeps its connection open for the whole generation. That changes how you size a server. The number of open connections is the arrival rate times how long each stays open (Little's law):</p>
<div class="eq">connections = requests per second × seconds open
500 tokens × 30 ms = 15 s per answer
100 requests/s × 15 s = 1,500 open connections</div>
<p>(Both figures are example values.) Most of those connections are idle at any instant, waiting for the next token. A design that dedicates one operating-system thread to each one wastes memory and time switching between them. Two common answers:</p>
<ul><li><b>Async I/O</b>: one thread runs an <b>event loop</b>. It watches thousands of sockets and handles whichever one is ready, then moves on. FastAPI and the course's gateway work this way.</li>
<li><b>Many worker processes</b>, each handling a share.</li></ul>
<p>Either way, the question for an LLM gateway is "how many connections can I hold?", not only "how many requests per second?". P2.7 builds one, and P2 returns to batching many users on one GPU.</p>`,
        check: { q: "Each streamed answer takes 15 s. Requests arrive at 100 per second. About how many connections are open at once?",
          options: ["15", "100", "1,500", "15,000"], answer: 2,
          why: "Open connections = arrival rate × duration = 100 × 15 = 1,500. Every second adds 100 new streams, and each lives for 15 seconds, so 15 seconds' worth are alive together." },
        scene(G) {
          G.text(160, 44, "one thread per connection", { size: 13, weight: 700, anchor: "middle", color: "hot" });
          G.text(480, 44, "one event loop, many sockets", { size: 13, weight: 700, anchor: "middle", color: "ok" });
          G.line(320, 30, 320, 330, { color: "line", dash: "4 4" });
          const ths = [];
          for (let i = 0; i < 6; i++) { const y = 66 + i * 44; G.circle(50, y + 14, 6, { fill: "k" }); G.line(58, y + 14, 90, y + 14, { color: "w", w: 2 }); ths.push(G.box(94, y, 90, 30, "thread " + (i + 1), { stroke: "hot", size: 11, color: "hot" })); G.label(190, y + 19, "waiting", { size: 11 }); }
          const ls = [];
          for (let i = 0; i < 6; i++) { const y = 66 + i * 44; G.circle(380, y + 14, 6, { fill: "k" }); ls.push(G.line(388, y + 14, 480, 80 + 70, { color: "w", w: 2 })); }
          G.box(480, 100, 120, 100, "", { fill: "ok" }); G.text(540, 142, "event loop", { size: 13, weight: 700, color: "bg", anchor: "middle" }); G.text(540, 162, "1 thread", { size: 12, color: "bg", anchor: "middle" });
          G.text(540, 184, "serves whichever", { size: 10, color: "bg", anchor: "middle" });
          G.rect(20, 346, 600, 66, { stroke: "line", rx: 8 });
          G.text(36, 372, "connections = rate × duration = 100/s × 15 s = 1,500", { size: 13, weight: 700 });
          G.text(36, 396, "memory and switching cost scale with threads, not with idle sockets", { size: 12, color: "muted" });
          G.from(ls, { opacity: 0, stagger: 0.08, duration: 0.2 });
          G.caption("idle connections are cheap for an event loop, costly as threads");
        } },

      { rail: "tools", title: "Three tools: curl, curl -N and TestClient",
        body: `<p>You now know what to look at. These tools let you look.</p>
<ul><li><code>curl -v URL</code> prints the request and response headers, the same view as step 4.</li>
<li><code>curl -N URL</code> turns off output buffering. With a streaming endpoint you see each <code>data:</code> line appear as the server sends it, so you can <i>watch</i> TTFT and ITL happen.</li>
<li>FastAPI's <code>TestClient</code> calls a Python app directly, with no network and no port. It is how the exercises, and the course's own gateway tests, check status codes and bodies quickly.</li></ul>
<div class="eq">curl -N -X POST \\
  http://localhost:8000/v1/chat/completions \\
  -H 'Content-Type: application/json' \\
  -d '{"model":"mock-llama-8b","stream":true,
       "messages":[{"role":"user","content":"hi"}]}'</div>
<p>When a test needs a real socket (for example to measure streaming), <code>course/common/python/s2s/gatewaytest.py</code> and <code>mockserver.py</code> start a server on a free localhost port inside the test. Nothing leaves your machine.</p>`,
        scene(G) {
          G.rect(20, 40, 600, 112, { stroke: "k", rx: 8 });
          G.text(34, 62, "$ curl -v http://localhost:8000/v1/models", { size: 12, color: "k" });
          G.text(34, 84, "> GET /v1/models HTTP/1.1", { size: 12 }); G.text(34, 104, "< HTTP/1.1 200 OK", { size: 12, color: "ok" }); G.text(34, 124, "< content-type: application/json", { size: 12, color: "muted" });
          G.text(330, 104, "headers, both directions", { size: 11, color: "muted" });
          G.rect(20, 166, 600, 112, { stroke: "v", rx: 8 });
          G.text(34, 188, "$ curl -N -X POST .../v1/chat/completions -d '{... \"stream\": true}'", { size: 12, color: "v" });
          const ev = ['data: {"choices":[{"delta":{"content":"Hel"}}]}', 'data: {"choices":[{"delta":{"content":"lo"}}]}', "data: [DONE]"];
          const evs = ev.map((t, i) => G.text(34, 212 + i * 22, t, { size: 12, color: i === 2 ? "ok" : "ink" }));
          G.text(470, 250, "each line appears live", { size: 11, color: "muted" });
          G.rect(20, 292, 600, 112, { stroke: "q", rx: 8 });
          lines(G, 34, 314, [["client = TestClient(app)", "q"], ["r = client.post(\"/v1/echo\", json={\"text\": \"hi\"})", "ink"], ["assert r.status_code == 200", "ink"], ["# no network, no port: calls the app in-process", "muted"]], 22, 12);
          G.from(evs, { opacity: 0, stagger: 0.45, duration: 0.2 });
          G.caption("look at the wire with curl, test it without the wire using TestClient");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>A client reaches a server by IP address and port. Each end holds a socket.</li>
<li>TCP costs one round trip to set up, TLS one or two more, and DNS a lookup. Keep-alive pays them once. Small requests are limited by latency, big ones by bandwidth.</li>
<li>An HTTP request is a method, a path, headers and a body; a response is a status code, headers and a body. LLM servers use JSON bodies in the OpenAI style.</li>
<li>400, 401 and 404 mean the request is wrong. 429 and 503 mean "not now". 500 and timeouts mean "maybe it ran".</li>
<li>Server-sent events stream tokens, which makes TTFT (first arrival) and ITL (gaps between arrivals) measurable.</li>
<li>Retry only what is idempotent or was refused, with backoff and jitter. Streaming holds connections open, so servers need async I/O or many workers.</li></ul>
<p><b>Next:</b> R.7 (a safe cloud account), then P0. P1.3 and P2.7 build directly on this page.</p>
<h2>Go further</h2><ul>
<li><a href="https://developer.mozilla.org/en-US/docs/Web/HTTP/Overview">MDN, An overview of HTTP</a>: requests, responses, headers and methods in more detail.</li>
<li><a href="https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events">MDN, Using server-sent events</a>: the full event format, including fields this lesson skipped.</li></ul>`,
    sim: {
      title: "Request timeline",
      intro: "One request to an LLM server on a brand-new connection. Set the network round-trip time, whether TLS is used, how long the server takes to make the first token, how many tokens come back and the gap between them. The bar shows where the time goes. Compare when the user first sees a token with streaming on and off.",
      height: 270,
      controls: [
        { id: "rtt", label: "network round-trip time, ms", min: 5, max: 300, step: 5, value: 60 },
        { id: "tls", label: "TLS", type: "select", value: 1, options: [[1, "on (TLS 1.3)"], [0, "off (plain http)"]] },
        { id: "ttft", label: "server time to first token, ms", min: 20, max: 3000, step: 10, value: 400 },
        { id: "n", label: "tokens in the answer", min: 1, max: 1000, step: 1, value: 200 },
        { id: "itl", label: "inter-token latency, ms", min: 5, max: 200, step: 5, value: 30 },
        { id: "stream", label: "streaming", type: "select", value: 1, options: [[1, "on"], [0, "off"]] },
      ],
      draw(G, v) {
        const R = v.rtt, tls = v.tls, sv = v.ttft, N = v.n, itl = v.itl, st = v.stream;
        const tcp = R, tl = tls * R, up = R / 2, gen = (N - 1) * itl, down = R / 2;
        const tStart = tcp + tl + up;                  // the server has the request
        const first = tStart + sv + down;              // first token reaches the client (streaming)
        const full = tStart + sv + gen + down;         // last token reaches the client
        const seen1 = st ? first : full, seenAll = full;
        const scale = 560 / full, X = (t) => 40 + t * scale;
        const segs = [["TCP", tcp, "k"], ["TLS", tl, "q"], ["request", up, "w"], ["server: 1st token", sv, "hot"], ["tokens", gen, "ok"], ["reply", down, "w"]];
        const lane = (y, title, showStream) => {
          G.label(40, y - 10, title, { size: 11, color: "ink" });
          let t = 0;
          segs.forEach(([n, d, c]) => {
            if (d <= 0) return;
            const col = !showStream && n === "tokens" ? "hot" : c;
            G.rect(X(t), y, Math.max(1, d * scale - 1), 34, { fill: col, rx: 3, opacity: !showStream && n === "tokens" ? 0.35 : 0.9 });
            if (d * scale > 46) G.text(X(t) + (d * scale) / 2, y + 21, n === "server: 1st token" ? "server" : !showStream && n === "tokens" ? "generating, not sent yet" : n, { size: 10, color: "bg", anchor: "middle", weight: 700 });
            t += d;
          });
        };
        lane(54, "streaming: tokens flow as they are made", true);
        lane(160, "no streaming: the server sends everything at the end", false);
        const mark = (t, y, lab, col) => { G.line(X(t), y - 6, X(t), y + 46, { color: col, w: 2, dash: "3 3" }); G.text(X(t) + 4 > 520 ? X(t) + 4 : X(t) + 4, y + 58, lab, { size: 10, color: col, anchor: X(t) > 450 ? "end" : "start" }); };
        mark(first, 54, "first token seen", "ok");
        mark(full, 160, "answer complete", "hot");
        G.label(40, 250, "scale: 0 to " + F.ms(full / 1000) + ".  DNS is assumed cached; the model is the same in both lanes.", { size: 10 });
        const pre = tcp + tl + up;
        return [
          { title: "When the user sees it", rows: [["first token, streaming on", F.ms(first / 1000)], ["first token, streaming off", F.ms(full / 1000)], ["full answer (both)", F.ms(full / 1000)], ["this setting's first token", F.ms(seen1 / 1000)]],
            chip: [st === 1, st ? "streaming: first token " + (full / first).toFixed(1) + "× sooner" : "no streaming: blank until the end"] },
          { title: "Before the server starts", rows: [["TCP + TLS + request flight", F.ms(pre / 1000)], ["share of first-token time", (100 * pre / first).toFixed(0) + "%"], ["round trips paid", String(1 + tls) + " + half a trip"]],
            gauge: [[Math.min(1, pre / first), "q"], [Math.max(0, 1 - pre / first), "hot"]], gaugeText: "network setup vs server wait (streaming)" },
          { title: "Reading the result", html: `<p class="note">First token = (TCP + TLS) round trips + half a trip out + server TTFT + half a trip back. The full answer adds (tokens − 1) × ITL. A kept-alive connection skips TCP and TLS, so try setting the round-trip time high and see how much of the first token is network. A model, not a measurement: it ignores queueing, slow start and packet loss.</p>` },
        ];
      },
    },
    practice: {
      intro: `Run these from the repository root. Each exercise has a starter file and a test; <code>S2S_SOLUTIONS=1</code> runs the same test against the reference solution. Run all four with <code>uv run pytest course/R-readiness/R.6-http-and-networking/exercises</code>. Nothing needs the internet: exercise 3 uses an in-process test client.`,
      items: [
        { title: "Parse a server-sent-events stream", tier: "T0 · medium", goal: "Turn raw bytes into events: multi-line data, comments, event names, CRLF line ends, [DONE] and a half-received last event.",
          cmd: "uv run pytest course/R-readiness/R.6-http-and-networking/exercises/01-parse-sse" },
        { title: "TTFT and inter-token latency", tier: "T0 · easy", goal: "From arrival timestamps compute TTFT, mean and p90 ITL, and end-to-end time; handle the one-chunk case.",
          cmd: "uv run pytest course/R-readiness/R.6-http-and-networking/exercises/02-ttft-itl" },
        { title: "A tiny validated JSON API", tier: "T0 · medium", goal: "Write POST /v1/echo with pydantic: 400 for bad JSON or blank text, 422 for a wrong shape, 200 otherwise. Tested with FastAPI's TestClient.",
          cmd: "uv run pytest course/R-readiness/R.6-http-and-networking/exercises/03-tiny-api" },
        { title: "A retry policy", tier: "T0 · easy", goal: "should_retry(status, attempt, idempotent) over a table of cases, plus exponential backoff with full jitter.",
          cmd: "uv run pytest course/R-readiness/R.6-http-and-networking/exercises/04-retry-policy" },
      ],
      labs: [
        { label: "Module README: the ten ideas in one page", path: "course/R-readiness/R.6-http-and-networking/README.md" },
        { label: "Self-check questions", path: "course/R-readiness/R.6-http-and-networking/quiz.md" },
        { label: "The mock LLM server: a real SSE endpoint to curl -N", path: "platform/mockllm/server.py" },
        { label: "How the course's tests start a real localhost server", path: "course/common/python/s2s/gatewaytest.py" },
      ],
    },
  });
})();
