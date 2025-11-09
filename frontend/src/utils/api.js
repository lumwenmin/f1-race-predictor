const API_BASE_URL = "http://localhost:8000";

export async function fetchRaceData() {
  const response = await fetch(`${API_BASE_URL}/race-data`);
  if (!response.ok) {
    throw new Error("Failed to fetch race data");
  }
  return response.json();
}

export async function predictRaceOutcome(formData) {
  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      driverId: parseInt(formData.driverId),
      circuitId: parseInt(formData.circuitId),
      grid: parseInt(formData.grid),
    }),
  });
  if (!response.ok) throw new Error("API error");
  const json = await response.json();
  return json;
}
