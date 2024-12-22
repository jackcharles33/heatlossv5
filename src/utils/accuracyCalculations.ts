import { getRecentPredictions } from '../db/database';

export async function getAverageAccuracy(): Promise<number> {
  const predictions = await getRecentPredictions();
  const validPredictions = predictions.filter(p => p.accuracy !== undefined);
  
  if (validPredictions.length === 0) return 85; // Default ±15% accuracy if no data

  const totalAccuracy = validPredictions.reduce((sum, p) => sum + (p.accuracy || 0), 0);
  return totalAccuracy / validPredictions.length;
}

export function calculateErrorMargin(prediction: number, accuracy: number): number {
  const errorPercentage = (100 - accuracy) / 100;
  return prediction * errorPercentage;
}