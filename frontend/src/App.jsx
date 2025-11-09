import { useState } from "react";
import PredictionForm from "./components/PredictionForm";
import ResultDisplay from "./components/ResultDisplay";

export default function App() {
  const [result, setResult] = useState(null);

  return (
    <div className="min-h-screen bg-gray-900 text-white p-4">
      <h1 className="text-4xl font-bold mb-4">🏎️ F1 Race Predictor</h1>
      <PredictionForm onResult={setResult} />
      <ResultDisplay result={result} />
    </div>
  );
}
