import { BaseHouseData } from '../types/HouseData';
import { dbPromise } from './database';

export async function getPredictionsForTraining(limit: number = 1000): Promise<BaseHouseData[]> {
  const db = await dbPromise;
  const predictions = await db.getAllFromIndex(
    'predictions',
    'by-timestamp',
    IDBKeyRange.lowerBound(0)
  );
  
  return predictions
    .filter(p => p.krakenValue && p.isAccurate && p.confidenceScore >= 0.8)
    .slice(0, limit)
    .map(({ krakenValue, size, age, propertyType, wallType, floorType, windowType, roofType }) => ({
      heatLoss: krakenValue!,
      size,
      age,
      propertyType,
      wallType,
      floorType,
      windowType,
      roofType
    }));
}