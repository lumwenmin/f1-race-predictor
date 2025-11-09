import { useState, useEffect } from "react";
import { fetchRaceData, predictRaceOutcome } from "../utils/api";

export default function PredictionForm({ onResult }) {
  const [drivers, setDrivers] = useState([]);
  const [circuits, setCircuits] = useState([]);
  const [form, setForm] = useState({
    driverId: "",
    circuitId: "",
    grid: 5,
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function loadRaceData() {
      try {
        const data = await fetchRaceData();
        setDrivers(data.drivers);
        setCircuits(data.circuits);
        setForm((form) => ({
          ...form,
          driverId: data.drivers[0]?.driverId || "",
          circuitId: data.circuits[0]?.circuitId || "",
        }));
      } catch {
        setError("Failed to load race data");
      }
    }
    loadRaceData();
  }, []);

  const handleChange = (e) => {
    const value =
      e.target.name === "grid" ? parseInt(e.target.value) : e.target.value;
    setForm({ ...form, [e.target.name]: value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const result = await predictRaceOutcome({
        driverId: form.driverId,
        circuitId: form.circuitId,
        grid: form.grid,
      });
      onResult(result);
    } catch {
      setError("Failed to get prediction");
    }
    setLoading(false);
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="max-w-md mx-auto p-6 bg-gray-800 rounded-lg shadow-md text-white"
    >
      <h2 className="text-2xl font-semibold mb-4">Race Prediction Input</h2>

      <label className="block mb-2">
        Driver:
        <select
          name="driverId"
          value={form.driverId}
          onChange={handleChange}
          required
          className="mt-1 w-full rounded-md bg-gray-700 border border-gray-600 px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
        >
          {drivers.map((driver) => (
            <option key={driver.driverId} value={driver.driverId}>
              {driver.name}
            </option>
          ))}
        </select>
      </label>

      <label className="block mb-2">
        Circuit:
        <select
          name="circuitId"
          value={form.circuitId}
          onChange={handleChange}
          required
          className="mt-1 w-full rounded-md bg-gray-700 border border-gray-600 px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
        >
          {circuits.map((circuit) => (
            <option key={circuit.circuitId} value={circuit.circuitId}>
              {circuit.name}
            </option>
          ))}
        </select>
      </label>

      <label className="block mb-4">
        Grid Position:
        <input
          name="grid"
          type="number"
          min={1}
          max={20}
          value={form.grid}
          onChange={handleChange}
          required
          className="mt-1 w-full rounded-md bg-gray-700 border border-gray-600 px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
        />
      </label>

      <button
        type="submit"
        disabled={loading}
        className="w-full py-2 rounded-md bg-indigo-600 hover:bg-indigo-700 transition duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {loading ? "Predicting..." : "Predict"}
      </button>

      {error && <p className="mt-3 text-red-500">{error}</p>}
    </form>
  );
}
