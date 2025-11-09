export default function ResultDisplay({ result }) {
  if (!result) return null;
  return (
    <div className="max-w-md mx-auto mt-6 p-6 bg-gray-900 rounded-lg text-white shadow-lg text-center">
      <h3 className="text-2xl font-bold mb-2">
        Prediction:{" "}
        <span
          className={
            result.prediction === 1 ? "text-green-400" : "text-red-400"
          }
        >
          {result.message}
        </span>
      </h3>
      <p className="text-lg">
        Confidence: <strong>{(result.confidence * 100).toFixed(1)}%</strong>
      </p>
    </div>
  );
}
