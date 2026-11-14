// Call the Wan worker from your Node backend (Node 18+). Videos take minutes: create a job, then poll (or store the id
// and check later). Keep the API key on the server only; never send it to the browser.
//
//   RUNPOD_API_KEY=...  RUNPOD_ENDPOINT_ID=...
const BASE = `https://api.runpod.ai/v2/${process.env.RUNPOD_ENDPOINT_ID}`;
const headers = { Authorization: `Bearer ${process.env.RUNPOD_API_KEY}`, "Content-Type": "application/json" };

// input: { prompt, logo_url?, aspect?: "9:16"|"16:9", duration_s?: 1..5, quality?: "fast"|"high", logo_target?, seed? }
export async function startVideo(input) {
  const r = await fetch(`${BASE}/run`, { method: "POST", headers, body: JSON.stringify({ input }) });
  if (!r.ok) throw new Error(`start failed: HTTP ${r.status}`);
  return (await r.json()).id;                       // store this job id with the user's request
}

// returns { status: "IN_QUEUE"|"IN_PROGRESS"|"COMPLETED"|"FAILED", video_url?, logo?, error? }
export async function videoStatus(jobId) {
  const r = await fetch(`${BASE}/status/${jobId}`, { headers });
  if (!r.ok) throw new Error(`status failed: HTTP ${r.status}`);
  const j = await r.json();
  if (j.status === "COMPLETED") {
    if (j.output?.error) return { status: "FAILED", error: j.output.error };   // worker rejected the input
    return { status: "COMPLETED", ...j.output };
  }
  return { status: j.status, error: j.error };
}

// Example (Express):
//   app.post("/api/videos",  async (req, res) => res.json({ jobId: await startVideo({ prompt: req.body.prompt, logo_url: req.body.logoUrl }) }));
//   app.get("/api/videos/:id", async (req, res) => res.json(await videoStatus(req.params.id)));
