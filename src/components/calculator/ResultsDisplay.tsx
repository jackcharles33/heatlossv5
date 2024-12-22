import { CalculationResults } from '../../types/calculator';

interface ResultsDisplayProps {
  results: CalculationResults;
}

export function ResultsDisplay({ results }: ResultsDisplayProps) {
  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">{results.totalHeatLoss.toFixed(1)} kW</h2>
        <p className="text-sm text-gray-500">Total Heat Loss</p>
        
        <div className="mt-4 grid grid-cols-2 gap-4">
          <div>
            <h3 className="font-medium">Walls</h3>
            <p>{results.breakdown.walls} kW</p>
          </div>
          <div>
            <h3 className="font-medium">Windows</h3>
            <p>{results.breakdown.windows} kW</p>
          </div>
          <div>
            <h3 className="font-medium">Floor</h3>
            <p>{results.breakdown.floor} kW</p>
          </div>
          <div>
            <h3 className="font-medium">Roof</h3>
            <p>{results.breakdown.roof} kW</p>
          </div>
        </div>
      </div>
    </div>
  );
}